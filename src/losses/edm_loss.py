from loguru import logger

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# === Epipolar utilities (loss-only) ===
def _skew(x: torch.Tensor) -> torch.Tensor:
    """
    x: [3] -> [3,3] skew-symmetric matrix
    """
    return x.new_tensor([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

@torch.no_grad()
def _compute_f_from_rt_k(R: torch.Tensor, t: torch.Tensor,
                         K0: torch.Tensor, K1: torch.Tensor) -> torch.Tensor:
    """
    **Assume all priors are given**
    Compute fundamental matrix F = K1^{-T} [t]_x R K0^{-1}
    R: [B,3,3], t: [B,3], K0,K1: [B,3,3] -> F: [B,3,3]
    """
    B = R.size(0)
    E = torch.stack([_skew(t[b]) @ R[b] for b in range(B)], dim=0)  # [B,3,3]
    K0inv = torch.inverse(K0)
    K1invT = torch.inverse(K1).transpose(1, 2)
    F = torch.einsum('bij,bjk,bkl->bil', K1invT, E, K0inv)
    return F

@torch.no_grad()
def _sampson_distance_points(x0_xy: torch.Tensor, x1_xy: torch.Tensor,
                             F_sel: torch.Tensor) -> torch.Tensor:
    """x0_xy,x1_xy: [M,2], F_sel: [M,3,3] -> Sampson distance [M]"""
    ones = x0_xy.new_ones(x0_xy.size(0), 1)
    x0h = torch.cat([x0_xy, ones], dim=1)  # [M,3]
    x1h = torch.cat([x1_xy, ones], dim=1)
    Fx0 = torch.einsum('mij,mj->mi', F_sel, x0h)
    Ftx1 = torch.einsum('mji,mj->mi', F_sel, x1h)
    x1Fx0 = (x1h * Fx0).sum(dim=1).abs()
    denom = Fx0[:, 0]**2 + Fx0[:, 1]**2 + Ftx1[:, 0]**2 + Ftx1[:, 1]**2 + 1e-9
    return x1Fx0 / denom

class EDMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.coord_length = config["edm"]["fine"]["coord_length"]

        self.loss_config = config["edm"]["loss"]
        self.sparse_spvs = self.loss_config["sparse_spvs"]

        # coarse-level
        self.c_pos_w = self.loss_config["pos_weight"]
        self.c_neg_w = self.loss_config["neg_weight"]
        # fine-level
        self.q_distribution = self.loss_config["q_distribution"]
        # self.fine_type = self.loss_config["fine_type"]
        # self.fine_loss = [nn.L1Loss(), nn.MSELoss(), nn.SmoothL1Loss()][1]

        # Epipolar loss-only regularization
        self.lambda_epi = float(self.loss_config.get("epi_weight", 0.0))
        self.epi_tau = float(self.loss_config.get("epi_tau", 1.0))

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        if self.loss_config["coarse_type"] == "cross_entropy":
            assert (
                not self.sparse_spvs
            ), "Sparse Supervision for cross-entropy not implemented!"
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[pos_mask])
            loss_neg = -torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()

        elif self.loss_config["coarse_type"] == "focal":
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config["focal_alpha"]
            gamma = self.loss_config["focal_gamma"]

            if self.sparse_spvs:
                pos_conf = conf[pos_mask]
                loss_pos = -alpha * \
                    torch.pow(1 - pos_conf, gamma) * pos_conf.log()

                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]

                loss = c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:
                loss_pos = (
                    -alpha
                    * torch.pow(1 - conf[pos_mask], gamma)
                    * (conf[pos_mask]).log()
                )

                loss_neg = (
                    -alpha
                    * torch.pow(conf[neg_mask], gamma)
                    * (1 - conf[neg_mask]).log()
                )
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]

                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError(
                "Unknown coarse loss: {type}".format(
                    type=self.loss_config["coarse_type"]
                )
            )

    def logQ(self, gt_uv, pred_jts, sigma):
        assert self.q_distribution in ["laplace", "gaussian"]

        error = (pred_jts - gt_uv) / (sigma + 1e-9)

        if self.q_distribution == "laplace":
            loss_q = torch.log(sigma * 2) + torch.abs(error)
        else:
            loss_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

        return loss_q

    def compute_rle_loss(self, data, f_weight=1):
        gt_uv = data["target_uv"]
        gt_uv_weight = data["target_uv_weight"]

        if gt_uv_weight.sum() == 0:
            if (
                self.training
            ):  # this seldomly happen when training, since we pad prediction with gt
                logger.warning(
                    "assign a false supervision to avoid ddp deadlock")
                gt_uv_weight[0] = True
                f_weight = 0.0
            else:
                return None

        residual = True
        if residual:
            Q_logprob = self.logQ(
                gt_uv[gt_uv_weight], data["mask_coord"], data["mask_sigma"]
            )
            loss = Q_logprob + data["nf_loss"]

        return loss.mean() * f_weight

    # def compute_fine_loss(self, data, f_weight=1):
    #     pred_jts = data["pred_coord"]
    #     gt_uv = data["target_uv"]
    #     gt_uv_weight = data["target_uv_weight"]

    #     if gt_uv_weight.sum() == 0:
    #         if (
    #             self.training
    #         ):  # this seldomly happen when training, since we pad prediction with gt
    #             logger.warning("assign a false supervision to avoid ddp deadlock")
    #             gt_uv_weight[0] = True
    #             f_weight = 0.0
    #         else:
    #             return None

    #     return self.fine_loss(gt_uv[gt_uv_weight], pred_jts[gt_uv_weight]) * f_weight

    @torch.no_grad()
    def compute_c_weight(self, data):
        """compute element-wise weights for computing coarse-level loss."""
        if "mask0" in data:
            c_weight = (
                data["mask0"].flatten(-2)[..., None]
                * data["mask1"].flatten(-2)[:, None]
            ).float()
        else:
            c_weight = None
        return c_weight
    
    def compute_epi_loss(self, data):
        if self.lambda_epi <= 0:
            return None
        if ("mkpts0_f" not in data) or (data["mkpts0_f"].numel() == 0):
            return None

        mk0, mk1, b_ids = data["mkpts0_f"], data["mkpts1_f"], data["m_bids"]
        R = data["T_0to1"][:, :3, :3]; t = data["T_0to1"][:, :3, 3]
        K0, K1 = data["K0"], data["K1"]

        F_all = _compute_f_from_rt_k(R, t, K0, K1)   # [B,3,3]
        F_sel = F_all[b_ids]                          # [M,3,3]
        d = _sampson_distance_points(mk0, mk1, F_sel) # [M], in pixels

        # --- Robust weighting by confidence (if present) ---
        w = data.get("mconf", None)
        if w is not None:
            w = w.clamp_min(1e-3)  # avoid zero
        else:
            w = torch.ones_like(d)

        # --- Robust scale normalization (median) ---
        s = d.detach().median()
        s = torch.clamp(s, min=1e-3)
        d_norm = d / s

        # --- Robust penalty: log1p vs softplus ---
        # loss_per = torch.log1p(d_norm / self.epi_tau)
        loss_per = F.softplus(d_norm / self.epi_tau)

        # Weighted mean
        loss = (w * loss_per).sum() / (w.sum() + 1e-9)
        return loss


    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data["conf_matrix"],
            data["conf_matrix_gt"],
            weight=c_weight,
        )
        loss = loss_c * self.loss_config["coarse_weight"]
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_rle_loss(
            data=data,
            f_weight=self.loss_config["fine_weight"],
        )
        if loss_f is not None:
            loss += loss_f
            loss_scalars.update(
                {"loss_f": min(loss_f.clone().detach().cpu(),
                               torch.tensor(1.0))}
            )
        else:
            assert self.training is False
            # 1 is the upper bound
            loss_scalars.update({"loss_f": torch.tensor(1.0)})
            
        # 3. epipolar loss-only regularization (does not change forward graph)
        loss_epi_val = None
        if self.lambda_epi > 0:
            loss_epi_val = self.compute_epi_loss(data)
            if loss_epi_val is not None:
                loss = loss + self.lambda_epi * loss_epi_val
                loss_scalars.update({"loss_epi": loss_epi_val.clone().detach().cpu()})

        loss_scalars.update({"loss": loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
