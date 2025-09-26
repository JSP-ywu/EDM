from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import lightning.pytorch as pl
from matplotlib import pyplot as plt

from src.edm import EDM
from src.edm.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
from src.losses.edm_loss import EDMLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.edm.neck.neck import DepthAnythingFeatureExtractor


class PL_EDM(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        # print(_config.keys())
        print('Current epi loss weight: ', _config['edm']['loss']['epi_weight'])
        print('Current epi loss tau: ', _config['edm']['loss']['epi_tau'])

        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        # Matcher: EDM
        self.matcher = EDM(config=_config["edm"])
        self.loss = EDMLoss(_config)

        # Optional depth feature extractor outside checkpoint
        use_hidden = _config["edm"].get("use_hidden", False)
        use_extract = _config["edm"].get("depth_from_extract", False)
        if use_hidden and use_extract:
            extractor = DepthAnythingFeatureExtractor()
            extractor.requires_grad_(False)
            extractor.eval()
            # 중요: __setattr__ 우회하여 등록/체크포인트 제외
            self.__dict__["_depth_extractor"] = extractor
        else:
            self.__dict__["_depth_extractor"] = None
     
        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location="cpu")[
                "state_dict"]
            self.matcher.load_state_dict(state_dict, strict=True)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.warmup = False
        self.reparameter = False
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        self.dump_dir = dump_dir

        # outputs
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == "linear":
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + (
                    self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif self.config.TRAINER.WARMUP_TYPE == "constant":
                pass
            else:
                raise ValueError(
                    f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):
        batch["global_step"] = int(self.global_step)
        # (optional) compute depth hidden features outside EDM
        if getattr(self, "_depth_extractor", None) is not None:
            # print('Extract hidden state....')
            with torch.no_grad():
                img0 = batch.get("depth_feat_image0", batch.get("image0"))
                img1 = batch.get("depth_feat_image1", batch.get("image1"))
                feat0, feat1 = self.__dict__["_depth_extractor"](img0, img1)
                dev = batch["image0"].device
                feat0 = feat0.to(dev, non_blocking=True)
                feat1 = feat1.to(dev, non_blocking=True)
                batch["depth_feat0"] = feat0
                batch["depth_feat1"] = feat1
        with self.profiler.profile("Compute coarse supervision"):
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("EDM"):
            with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                batch = self.matcher(batch)

        # with self.profiler.profile("Compute fine supervision"):
        #     with torch.autocast(enabled=False, device_type='cuda'):
        #         compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                self.loss(batch)

    def _compute_metrics(self, batch):
        # compute epi_errs for each match
        compute_symmetrical_epipolar_errors(batch)

        compute_pose_errors(
            batch, self.config
        )  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch["pair_names"]))
        bs = batch["image0"].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
            "epi_errs": [
                (batch["epi_errs"].reshape(-1, 1))[batch["m_bids"] == b]
                .reshape(-1)
                .cpu()
                .numpy()
                for b in range(bs)
            ],
            "R_errs": batch["R_errs"],
            "t_errs": batch["t_errs"],
            "inliers": batch["inliers"],
            "num_matches": [batch["mconf"].shape[0]],  # batch size = 1 only
        }
        ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.log(
                    {f"train/{k}": v}, step=self.global_step
                )
            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                self._filter_and_compute_final_matches(batch)
                compute_symmetrical_epipolar_errors(
                    batch
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.log(
                        {f"train_match/{k}": v}, step=self.global_step
                    )

        out = {"loss": batch["loss"]}
        self.log("loss", batch["loss"], prog_bar=True, rank_zero_only=True)

        # avoid significant memory growth
        # self.train_step_outputs.append(out)
        return out

    def on_after_backward(self) -> None:
        # (선택) Make parameter set which is registered to optimizer in real
        opt_params = set()
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                opt_params.update(group["params"])

        for n, p in self.named_parameters():
            # Pass frozen parameters
            if not p.requires_grad:
                continue
            # Pass non-registered modules
            if len(opt_params) and p not in opt_params:
                continue
            # When required gradient but not train
            if p.grad is None:
                print(n)

        return super().on_after_backward()

    def on_train_epoch_end(self):
        pass  # avoid significant memory growth during training
        # outputs = self.train_step_outputs
        # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # if self.trainer.global_rank == 0:
        #     self.logger.experiment.add_scalar(
        #         'train/avg_loss_on_epoch', avg_loss,
        #         global_step=self.current_epoch)
        # self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)

        # self._filter_and_compute_final_matches(batch)

        ret_dict, _ = self._compute_metrics(batch)

        val_plot_interval = max(
            self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0:
            figures = make_matching_figures(
                batch, self.config, mode=self.config.TRAINER.PLOT_MODE
            )

        out = {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            "figures": figures,
        }
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        # handle multiple validation sets
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if self.trainer.ckpt_path is None and self.trainer.sanity_checking:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(
                    flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(
                    val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            _figures = [o["figures"] for o in outputs]
            figures = {
                k: flattenList(
                    gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }

            # wandb records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.log(
                        {f"val_{valset_idx}/avg_{k}": mean_v}, step=self.global_step
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.log(
                        {f"metrics_{valset_idx}/{k}": v}, step=self.global_step
                    )

                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.log(
                                {f"val_match_{valset_idx}/{k}/pair-{plot_idx}": fig},
                                step=self.global_step,
                            )
                            plt.close(fig)  # close the figure to free memory
            plt.close("all")

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(
                f"auc@{thr}",
                torch.tensor(np.mean(multi_val_metrics[f"auc@{thr}"])),
                sync_dist=True,
            )  # ckpt monitors on this
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        if self.config.EDM.HALF:
            self.matcher = self.matcher.eval().half()

        # Following EfficientLoFTR
        if not self.warmup:
            if self.config.EDM.HALF:
                for i in range(50):
                    batch = self.matcher(batch)
            else:
                with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                    for i in range(50):
                        batch = self.matcher(batch)
            self.warmup = True

        torch.cuda.synchronize()
        if self.config.EDM.HALF:
            self.start_event.record()
            batch = self.matcher(batch)
            self.end_event.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_event.elapsed_time(self.end_event)
        else:
            with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
                self.start_event.record()
                batch = self.matcher(batch)
                self.end_event.record()
                torch.cuda.synchronize()
                self.total_ms += self.start_event.elapsed_time(self.end_event)

        # --- Analysis Data Preparation ---
        # 1. Store initial top-k coarse matches before filtering
        initial_mkpts0_c = batch['mkpts0_c'].clone()
        initial_mkpts1_c = batch['mkpts1_c'].clone()
        initial_b_ids = batch['b_ids'].clone()
        initial_i_ids = batch['i_ids'].clone()
        initial_j_ids = batch['j_ids'].clone()

        # 2. Compute final matches and get the filter mask
        final_keep_mask = self._post_process_and_filter(batch)

        # 3. Compute metrics on the FINAL filtered matches
        ret_dict, rel_pair_names = self._compute_metrics(batch) # This now uses the filtered matches

        # 4. Calculate Coarse Precision using INITIAL matches
        conf_matrix_gt = batch['conf_matrix_gt']
        gt_vals = conf_matrix_gt[initial_b_ids, initial_i_ids, initial_j_ids]
        coarse_precision = gt_vals.mean() if len(gt_vals) > 0 else 0.
        ret_dict['metrics']['coarse_precision'] = [coarse_precision.cpu().numpy()]

        if self.dump_dir is not None:
            compute_symmetrical_epipolar_errors(batch)
            with self.profiler.profile("dump_results"):
                # dump results for further analysis
                keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ["R_errs", "t_errs", "inliers"]:
                        item[key] = batch[key][b_id]

                    initial_b_mask = initial_b_ids == b_id
                    final_b_mask = batch['m_bids'] == b_id # m_bids is now filtered

                    item['initial_mkpts0_c'] = initial_mkpts0_c[initial_b_mask].cpu().numpy()
                    item['initial_mkpts1_c'] = initial_mkpts1_c[initial_b_mask].cpu().numpy()
                    item['rejected_mask'] = ~final_keep_mask[initial_b_mask].cpu().numpy()
                    
                    item['final_mkpts0_f'] = batch['mkpts0_f'][final_b_mask].cpu().numpy()
                    item['final_mkpts1_f'] = batch['mkpts1_f'][final_b_mask].cpu().numpy()
                    item['final_epi_errs'] = batch['epi_errs'][final_b_mask].cpu().numpy()
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        self.test_step_outputs.append(ret_dict)
        return ret_dict

    # ------- Original test_step
    # def test_step(self, batch, batch_idx):
    #     if self.config.EDM.HALF:
    #         self.matcher = self.matcher.eval().half()

    #     # Following EfficientLoFTR
    #     if not self.warmup:
    #         if self.config.EDM.HALF:
    #             for i in range(50):
    #                 batch = self.matcher(batch)
    #         else:
    #             with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
    #                 for i in range(50):
    #                     batch = self.matcher(batch)
    #         self.warmup = True

    #     torch.cuda.synchronize()
    #     if self.config.EDM.HALF:
    #         self.start_event.record()
    #         batch = self.matcher(batch)
    #         self.end_event.record()
    #         torch.cuda.synchronize()
    #         self.total_ms += self.start_event.elapsed_time(self.end_event)
    #     else:
    #         with torch.autocast(enabled=self.config.EDM.MP, device_type="cuda"):
    #             self.start_event.record()
    #             batch = self.matcher(batch)
    #             self.end_event.record()
    #             torch.cuda.synchronize()
    #             self.total_ms += self.start_event.elapsed_time(self.end_event)

    #     self._filter_and_compute_final_matches(batch)
    #     ret_dict, rel_pair_names = self._compute_metrics(batch)

    #     if self.dump_dir is not None:
    #         with self.profiler.profile("dump_results"):
    #             # dump results for further analysis
    #             keys_to_save = {"mkpts0_f", "mkpts1_f", "mconf", "epi_errs"}
    #             pair_names = list(zip(*batch["pair_names"]))
    #             bs = batch["image0"].shape[0]
    #             dumps = []
    #             for b_id in range(bs):
    #                 item = {}
    #                 mask = batch["m_bids"] == b_id
    #                 item["pair_names"] = pair_names[b_id]
    #                 item["identifier"] = "#".join(rel_pair_names[b_id])
    #                 for key in keys_to_save:
    #                     item[key] = batch[key][mask].cpu().numpy()
    #                 for key in ["R_errs", "t_errs", "inliers"]:
    #                     item[key] = batch[key][b_id]
    #                 dumps.append(item)
    #             ret_dict["dumps"] = dumps

    #     self.test_step_outputs.append(ret_dict)
    #     return ret_dict

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]

        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # dump
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"]
                                 for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        # [{key: [{...}, *#bs]}, *#batch]
        if self.trainer.global_rank == 0:
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
            )
            if 'coarse_precision' in metrics:
                val_metrics_4tb['coarse_precision'] = np.mean(metrics['coarse_precision'])

            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            
            print(
                "Averaged Matching time over 1500 pairs: {:.2f} ms".format(
                    self.total_ms / 1500
                )
            )
            if self.dump_dir is not None:
                for i in range(min(self.n_vals_plot, len(dumps))):
                    dump_item = dumps[i]
                    batch_for_plot = {
                        'image0': torch.from_numpy(dump_item['image0']), # Assuming you save images in dump
                        'image1': torch.from_numpy(dump_item['image1']),
                        'dataset_name': [self.config.DATASET.TEST_DATASET]
                    }
                    
                    # Plot rejected matches
                    fig_rejected = make_matching_figures(
                        batch_for_plot, self.config, mode='rejected',
                        mkpts0=dump_item['initial_mkpts0_c'],
                        mkpts1=dump_item['initial_mkpts1_c'],
                        mask=dump_item['rejected_mask']
                    )
                    self.logger.experiment.log(
                        {f"test_analysis/rejected/pair-{i}": fig_rejected['rejected'][0]},
                        step=self.global_step
                    )

                    # Plot failure cases
                    fig_failure = make_matching_figures(
                        batch_for_plot, self.config, mode='failure',
                        mkpts0=dump_item['final_mkpts0_f'],
                        mkpts1=dump_item['final_mkpts1_f'],
                        epi_errs=dump_item['final_epi_errs']
                    )
                    self.logger.experiment.log(
                        {f"test_analysis/failure/pair-{i}": fig_failure['failure'][0]},
                        step=self.global_step
                    )
                np.save(Path(self.dump_dir) / "EDM_pred_eval", dumps)

        self.test_step_outputs.clear()
    
    # ----------- Original on_test_epoch_end
    # def on_test_epoch_end(self):
    #     outputs = self.test_step_outputs
    #     # metrics: dict of list, numpy
    #     _metrics = [o["metrics"] for o in outputs]

    #     metrics = {
    #         k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
    #         for k in _metrics[0]
    #     }

    #     # dump
    #     if self.dump_dir is not None:
    #         Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
    #         _dumps = flattenList([o["dumps"]
    #                              for o in outputs])  # [{...}, #bs*#batch]
    #         dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
    #         logger.info(
    #             f"Prediction and evaluation results will be saved to: {self.dump_dir}"
    #         )

    #     # [{key: [{...}, *#bs]}, *#batch]
    #     if self.trainer.global_rank == 0:
    #         val_metrics_4tb = aggregate_metrics(
    #             metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
    #         )

    #         logger.info("\n" + pprint.pformat(val_metrics_4tb))
    #         print(
    #             "Averaged Matching time over 1500 pairs: {:.2f} ms".format(
    #                 self.total_ms / 1500
    #             )
    #         )
    #         if self.dump_dir is not None:
    #             np.save(Path(self.dump_dir) / "EDM_pred_eval", dumps)

    #     self.test_step_outputs.clear()

    # For debugging
    @torch.no_grad()
    def _post_process_and_filter(self, data):
        """
        Takes raw model predictions from the data dict and computes the final,
        filtered matches. This logic is moved from the original FineMatching module.
        """
        # Get raw predictions
        pred_coord = data['pred_coord']
        mconf = data['mconf']
        
        # De-normalize offset to pixel scale
        offset = pred_coord * self.matcher.local_resolution
        
        if self.config['edm']['fine']['bi_directional_refine']:
            offset_01, offset_10 = torch.chunk(offset, 2, dim=0)
            score_01, score_10 = torch.chunk(data['pred_score'], 2, dim=0)

            # --- Bi-directional Consistency Check ---
            use_01_mask = score_01 > score_10
            mkpts0_f = torch.where(use_01_mask.unsqueeze(1), data['mkpts0_c'], data['mkpts0_c'] + offset_10)
            mkpts1_f = torch.where(use_01_mask.unsqueeze(1), data['mkpts1_c'] + offset_01, data['mkpts1_c'])
            final_score = torch.where(use_01_mask, score_01, score_10)
        else:
            mkpts0_f = data['mkpts0_c']
            mkpts1_f = data['mkpts1_c'] + offset
            final_score = data['pred_score']

        # --- Filtering ---
        final_mask = mconf > self.config['edm']['coarse']['mconf_thr']
        final_mask &= final_score > self.config['edm']['fine']['sigma_thr']

        border_rm = self.config['edm']['coarse']['border_rm']
        h0, w0 = data['hw0_i']
        h1, w1 = data['hw1_i']
        final_mask &= (mkpts0_f[:, 0] >= border_rm) & (mkpts0_f[:, 0] < w0 - border_rm) & \
                      (mkpts0_f[:, 1] >= border_rm) & (mkpts0_f[:, 1] < h0 - border_rm) & \
                      (mkpts1_f[:, 0] >= border_rm) & (mkpts1_f[:, 0] < w1 - border_rm) & \
                      (mkpts1_f[:, 1] >= border_rm) & (mkpts1_f[:, 1] < h1 - border_rm)

        # Update data dictionary with the final, filtered matches
        data.update({
            'm_bids': data['b_ids'][final_mask],
            'mkpts0_f': mkpts0_f[final_mask],
            'mkpts1_f': mkpts1_f[final_mask],
            'mconf': mconf[final_mask]
        })
        
        return final_mask
    # @torch.no_grad()
    # def _filter_and_compute_final_matches(self, data):
    #     """
    #     Applies filters to the raw fine-level predictions to get the final match set.
    #     This version correctly handles the symmetrical bi-directional predictions.
    #     """
    #     # Get raw predictions from the data dictionary
    #     pred_offset_px = data['pred_offset_fine_px'] * self.matcher.local_resolution
    #     pred_score = data['pred_score'] # This is 1 - sigma
    #     mconf = data['mconf']
    #     mkpts0_c = data['mkpts0_c']
    #     mkpts1_c = data['mkpts1_c']

    #     # The predictions are concatenated [0->1, 1->0]
    #     offset_01, offset_10 = torch.chunk(pred_offset_px, 2, dim=0)
    #     score_01, score_10 = torch.chunk(pred_score, 2, dim=0)

    #     # --- Symmetrical Bi-directional Check ---
    #     mkpts0_f_from_10 = mkpts0_c + offset_10 # Refined point in 0, from 1's perspective
    #     mkpts1_f_from_10 = mkpts1_c            # Anchor in 1 (coarse center)

    #     mkpts0_f_from_01 = mkpts0_c            # Anchor in 0 (coarse center)
    #     mkpts1_f_from_01 = mkpts1_c + offset_01 # Refined point in 1, from 0's perspective

    #     # Choose the entire coordinate pair based on the more confident direction
    #     use_01_mask = score_01 > score_10
    #     mkpts0_f = torch.where(use_01_mask.unsqueeze(1), mkpts0_f_from_01, mkpts0_f_from_10)
    #     mkpts1_f = torch.where(use_01_mask.unsqueeze(1), mkpts1_f_from_01, mkpts1_f_from_10)
        
    #     # Final confidence scores
    #     final_score = torch.where(use_01_mask, score_01, score_10)
        
    #     # --- Filtering ---
    #     # 1. Coarse-level confidence threshold
    #     conf_mask = mconf > self.config['edm']['coarse']['mconf_thr']
        
    #     # 2. Fine-level confidence threshold (from sigma)
    #     conf_mask &= final_score > self.config['edm']['fine']['sigma_thr']

    #     # 3. Border removal
    #     border_rm = self.config['edm']['coarse']['border_rm']
    #     h0, w0 = data['hw0_i']
    #     h1, w1 = data['hw1_i']
    #     conf_mask &= (mkpts0_f[:, 0] >= border_rm) & (mkpts0_f[:, 0] < w0 - border_rm) & \
    #                   (mkpts0_f[:, 1] >= border_rm) & (mkpts0_f[:, 1] < h0 - border_rm) & \
    #                   (mkpts1_f[:, 0] >= border_rm) & (mkpts1_f[:, 0] < w1 - border_rm) & \
    #                   (mkpts1_f[:, 1] >= border_rm) & (mkpts1_f[:, 1] < h1 - border_rm)

    #     # Update data dictionary with the final, filtered matches
    #     data.update({
    #         'm_bids': data['b_ids'][conf_mask],
    #         'mkpts0_f': mkpts0_f[conf_mask],
    #         'mkpts1_f': mkpts1_f[conf_mask],
    #         'mconf': mconf[conf_mask],
    #         'mconf_fine': final_score[conf_mask]
    #     })

    # def on_fit_start(self):
    #     # Ensure depth extractor is on the same device as the module
    #     if getattr(self, "_depth_extractor", None) is not None:
    #         try:
    #             self._depth_extractor.to(self.device)
    #         except Exception as e:
    #             print(f"[on_fit_start] failed to move depth extractor to {self.device}: {e}")
    #     try:
    #         ws = getattr(self.trainer, "world_size", None)
    #         nd = getattr(self.trainer, "num_nodes", None)
    #         ndv = getattr(self.trainer, "num_devices", None)
    #         print(f"[on_fit_start] world_size={ws}, num_devices={ndv}, num_nodes={nd}, "
    #             f"global_rank={self.global_rank}, local_rank={self.local_rank}")
    #         if ws is not None:
    #             self.config.TRAINER.WORLD_SIZE = int(ws)
    #     except Exception:
    #         pass
