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

        # ADDED: Hyperparameter for the new BCE loss weight
        self.bce_weight = self.loss_config.get("bce_weight", 0.5)

        # Epipolar loss-only regularization
        self.bi_directional_refine = self.config['edm']['fine']['bi_directional_refine']
        self.lambda_epi = float(self.loss_config.get("epi_weight", 0.0))
        self.epi_tau = float(self.loss_config.get("epi_tau", 1.0))
    
        # EPI robustification defaults
        self.epi_min_parallax_deg = float(self.loss_config.get("epi_min_parallax_deg", 0.5))
        self.epi_gate_mult = float(self.loss_config.get("epi_gate_mult", 5.0))
        self.cycle_weight = float(self.loss_config.get("cycle_weight", 0.2))

        self.epi_warmup_steps = int(self.loss_config.get("epi_warmup_steps", 6900))
        self.epi_full_steps   = int(self.loss_config.get("epi_full_steps", 69000))

        # EMA scale for epi residual normalization
        self.register_buffer("epi_s_ema", torch.tensor(1.0))

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
        # gt_uv = data["target_uv"]
        # gt_uv_weight = data["target_uv_weight"]

        # if gt_uv_weight.sum() == 0:
        #     if (
        #         self.training
        #     ):  # this seldomly happen when training, since we pad prediction with gt
        #         logger.warning(
        #             "assign a false supervision to avoid ddp deadlock")
        #         gt_uv_weight[0] = True
        #         f_weight = 0.0
        #     else:
        #         return None

        # residual = True
        # if residual:
        #     Q_logprob = self.logQ(
        #         gt_uv[gt_uv_weight], data["mask_coord"], data["mask_sigma"]
        #     )
        #     loss = Q_logprob + data["nf_loss"]

        # return loss.mean() * f_weight
        """
        Computes the RLE loss based on outputs from FineMatchingV2.
        """
        gt_uv = data["target_uv"]
        gt_uv_weight = data["target_uv_weight"]
        
        if gt_uv_weight.sum() == 0:
            if self.training:
                logger.warning("Assigning a false supervision to avoid DDP deadlock in RLE loss.")
                return torch.tensor(0.0, device=gt_uv.device, requires_grad=True)
            return None

        # These are the concatenated predictions for both directions
        pred_offset = data["pred_offset"]
        pred_sigma = data["pred_sigma"]
        
        # The GT for RLE is the offset in the local window, normalized
        gt_offset_norm = gt_uv / self.config['edm']['local_resolution']
        
        # The prediction for RLE should also be the normalized offset in the local window
        pred_offset_norm = pred_offset / self.config['edm']['local_resolution']

        # Select training samples
        pred_offset_masked = pred_offset_norm[gt_uv_weight]
        gt_offset_masked = gt_offset_norm[gt_uv_weight]
        sigma_masked = pred_sigma[gt_uv_weight]

        # The normalizing flow part (assuming flow is part of the fine_matching module)
        # This part requires access to the flow model, which is tricky from the loss module.
        # A simple solution is to pre-calculate nf_loss in the fine_matching module.
        # Here we assume it's pre-calculated and named 'fine_nf_loss'.
        nf_loss = data['fine_nf_loss'] # This needs to be calculated and added to `data`
        
        Q_logprob = self.logQ(gt_offset_masked, pred_offset_masked, sigma_masked)
        loss = Q_logprob + nf_loss

        return loss.mean() * f_weight

    # ADDED: New function for BCE loss
    def compute_bce_loss(self, data, f_weight=1.0):
        """
        Computes the BCE loss for match confidence.
        """
        if 'fine_match_logits' not in data:
            return None
        
        logits = data['fine_match_logits']
        # The GT labels are the mask indicating if the match is an inlier
        gt_labels = data['target_uv_weight'].float()
        
        if gt_labels.sum() == 0:
             if self.training:
                logger.warning("Assigning a false supervision to avoid DDP deadlock in BCE loss.")
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
             return None
        
        # The GT from supervision is for both directions, so it matches the concatenated logits
        loss = F.binary_cross_entropy_with_logits(logits, gt_labels)
        return loss * f_weight


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
        # Need coarse selections and fine offsets to define correspondences
        if ("i_ids" not in data) or (data["i_ids"].numel() == 0):
            return None

        # --- Build differentiable matched points from fine head outputs ---
        # Base (coarse-cell centers) in image coords
        mkpts0_c = data.get("mkpts0_c", None)
        mkpts1_c = data.get("mkpts1_c", None)
        if mkpts0_c is None or mkpts1_c is None:
            return None

        # Predicted local offsets in [-0.5, 0.5], shape [M,2] or [2M,2] if bi-directional
        pred_coord = data.get("pred_coord", None)
        if pred_coord is None or pred_coord.numel() == 0:
            return None

        b_ids = data["b_ids"]               # [M]
        R = data["T_0to1"][:, :3, :3]
        t = data["T_0to1"][:, :3, 3]
        K0, K1 = data["K0"], data["K1"]

        # Scale from local window units -> image pixels (scalar)
        # Prefer deriving from current batch if available; fallback to config
        try:
            lr0 = float(data["hw0_i"][0]) / float(data["hw0_c"][0])
            lr1 = float(data["hw1_i"][0]) / float(data["hw1_c"][0])
            local_res = (lr0 + lr1) * 0.5  # scalar
        except Exception:
            local_res = float(self.config["edm"]["local_resolution"])  # scalar
        
        # Shapes
        M = mkpts0_c.shape[0]
        P = pred_coord.shape[0]
        # print('mkpts0_c: ', mkpts0_c)
        # print('mkpts1_c: ', mkpts1_c)
        # print('scale0:', scale0)
        # print('scale1:', scale1)


        # Use the original M-length indices for scales/weights (b_ids may be duplicated after final selection)
        b_ids_all = b_ids
        b_ids_M = b_ids_all[:M]
        scale0_M = data["scale0"][b_ids_M] if "scale0" in data else mkpts0_c.new_ones(M, 2)
        scale1_M = data["scale1"][b_ids_M] if "scale1" in data else mkpts1_c.new_ones(M, 2)

        # Optional confidence of length M
        w_all = data.get("mconf", None)
        w_M = None
        if w_all is not None and w_all.shape[0] >= M:
            w_M = w_all[:M]


        if P == M:
            # One-direction (0->1)
            mk0 = mkpts0_c                          # [M,2]
            mk1 = mkpts1_c + pred_coord * local_res * scale1_M  # [M,2]
            m_bids = b_ids_M
            w = w_M
        elif P == 2 * M:
            # Bi-directional (0->1 and 1->0). First M correspond to 0->1, last M to 1->0
            pred01 = pred_coord[:M]
            pred10 = pred_coord[M:]
            mk0_a = mkpts0_c
            # print('pred01: ', pred01)
            # print('pred10: ', pred10)
            mk1_a = mkpts1_c + pred01 * local_res * scale1_M
            mk0_b = mkpts0_c + pred10 * local_res * scale0_M
            mk1_b = mkpts1_c
            mk0 = torch.cat([mk0_a, mk0_b], dim=0)
            mk1 = torch.cat([mk1_a, mk1_b], dim=0)
            m_bids = torch.cat([b_ids_M, b_ids_M], dim=0)
            w = torch.cat([w_M, w_M], dim=0) if w_M is not None else None
        else:
            # Unexpected shape; fall back to one-direction using the first M rows
            mk0 = mkpts0_c
            mk1 = mkpts1_c + pred_coord[:M] * local_res * scale1_M
            m_bids = b_ids_M
            w = w_M

        # --- Fundamental matrices per batch ---
        F_all = _compute_f_from_rt_k(R, t, K0, K1)      # [B,3,3], no gradients
        F_sel = F_all[m_bids]                           # [*,3,3]
        # Align F to the (scaled) pixel coordinate system used by mk0/mk1.
        # If x' = S x  (S = diag(sx, sy, 1)), then  F' = S1^{-T} * F * S0^{-1}.
        # Here (sx, sy) are the per-pair scales used when forming mkpts*_c.
        if P == 2 * M:
            s0_pair = torch.cat([scale0_M, scale0_M], dim=0)  # [2M, 2]
            s1_pair = torch.cat([scale1_M, scale1_M], dim=0)  # [2M, 2]
        else:
            s0_pair = scale0_M                                 # [M, 2]
            s1_pair = scale1_M                                 # [M, 2]

        eps = 1e-12 # For preventing divided-by-zero
        sx0 = s0_pair[:, 0].clamp_min(eps)
        sy0 = s0_pair[:, 1].clamp_min(eps)
        sx1 = s1_pair[:, 0].clamp_min(eps)
        sy1 = s1_pair[:, 1].clamp_min(eps)

        S0_inv = torch.zeros(F_sel.size(0), 3, 3, device=F_sel.device, dtype=F_sel.dtype)
        S1_inv = torch.zeros_like(S0_inv)
        S0_inv[:, 0, 0] = 1.0 / sx0
        S0_inv[:, 1, 1] = 1.0 / sy0
        S0_inv[:, 2, 2] = 1.0
        S1_inv[:, 0, 0] = 1.0 / sx1
        S1_inv[:, 1, 1] = 1.0 / sy1
        S1_inv[:, 2, 2] = 1.0

        F_img = torch.einsum('mij,mjk,mkl->mil', S1_inv.transpose(1, 2), F_sel, S0_inv)
        # --- Sampson distance with gradients ---
        d = _sampson_distance_points(mk0, mk1, F_img)   # [*]

        # === Train-time gating for stability ===
        # Confidence mask
        if w is not None:
            w = w.clamp_min(1e-6)
            mask_conf = w > 0.0
        else:
            w = torch.ones_like(d)
            mask_conf = torch.ones_like(d, dtype=torch.bool)

        # Parallax gate (skip ill-conditioned pairs: near-pure rotation / very low parallax)
        try:
            ones = mk0.new_ones(mk0.size(0), 1)
            x0h = torch.cat([mk0, ones], dim=1)
            x1h = torch.cat([mk1, ones], dim=1)
            # bearings in each camera frame
            K0_inv_sel = torch.inverse(K0)[m_bids]
            K1_inv_sel = torch.inverse(K1)[m_bids]
            v0 = torch.einsum('mij,mj->mi', K0_inv_sel, x0h)
            v1 = torch.einsum('mij,mj->mi', K1_inv_sel, x1h)
            v0 = v0 / (v0.norm(dim=1, keepdim=True) + 1e-9)
            v1 = v1 / (v1.norm(dim=1, keepdim=True) + 1e-9)
            # rotate v0 into cam1 frame
            R_sel = R[m_bids]
            v0_to_1 = torch.einsum('mij,mj->mi', R_sel, v0)
            cosang = (v0_to_1 * v1).sum(dim=1).clamp(-1.0, 1.0)
            parallax_deg = torch.rad2deg(torch.acos(cosang))
            mask_parallax = parallax_deg > self.epi_min_parallax_deg
        except Exception:
            # if anything goes wrong, don't drop by parallax
            mask_parallax = torch.ones_like(d, dtype=torch.bool)

        # Distance gate (adaptive, median-based)
        d_med = d.detach().median()
        gate_thr = d_med * self.epi_gate_mult
        mask_dist = d < gate_thr

        valid = mask_conf & mask_parallax & mask_dist
        if valid.sum() < 16:
            return None

        d = d[valid]
        w = w[valid]

        # === Robust scale (EMA of median) and Charbonnier penalty ===
        s_now = d.detach().median().clamp_min(1e-3)
        # EMA update (no grad)
        self.epi_s_ema = 0.99 * self.epi_s_ema + 0.01 * s_now
        s = float(self.epi_s_ema)

        d_norm = d / (s + 1e-9)
        eps = 1e-6
        rho = torch.sqrt((d_norm / self.epi_tau) ** 2 + eps)  # Charbonnier

        loss = (w * rho).sum() / (w.sum() + 1e-9)
        return loss

    def compute_cycle_loss(self, data):
        """Train-only: encourage 0->1 and 1->0 fine offsets to cancel (same pixel units).
            Returns: scalar loss or None
        """
        if not self.bi_directional_refine or self.cycle_weight <= 0:
            return None
        pred = data.get("pred_coord", None)
        if pred is None:
            return None
        M = data["mkpts0_c"].shape[0]
        if pred.shape[0] != 2 * M:
            return None
        
        pred01, pred10 = pred[:M], pred[M:]
        b_ids = data["b_ids"][:M]

        # local offset -> pixels
        try:
            lr0 = float(data["hw0_i"][0]) / float(data["hw0_c"][0])
            lr1 = float(data["hw1_i"][0]) / float(data["hw1_c"][0])
            local_res = (lr0 + lr1) * 0.5
        except Exception:
            local_res = float(self.config["edm"]["local_resolution"])  # fallback

        scale0_M = data["scale0"][b_ids] if "scale0" in data else pred01.new_ones(M, 2)
        scale1_M = data["scale1"][b_ids] if "scale1" in data else pred01.new_ones(M, 2)

        off01_pix = pred01 * local_res * scale1_M
        off10_pix = pred10 * local_res * scale0_M

        # cycle consistency in the same reference frame (approximate)
        res = off01_pix + off10_pix  # [M,2]
        # weight by confidence if available
        w = data.get("mconf", None)
        if w is not None and w.shape[0] >= M:
            w = w[:M].clamp_min(1e-6)
            loss_vec = F.smooth_l1_loss(res, res.new_zeros(res.shape), reduction="none").sum(dim=1)
            loss = (w * loss_vec).sum() / (w.sum() + 1e-9)
        else:
            loss = F.smooth_l1_loss(res, res.new_zeros(res.shape), reduction="mean")
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

        # # 2. fine-level loss
        # loss_f = self.compute_rle_loss(
        #     data=data,
        #     f_weight=self.loss_config["fine_weight"],
        # )
        # if loss_f is not None:
        #     loss += loss_f
        #     loss_scalars.update(
        #         {"loss_f": min(loss_f.clone().detach().cpu(),
        #                        torch.tensor(1.0))}
        #     )
        # else:
        #     assert self.training is False
        #     # 1 is the upper bound
        #     loss_scalars.update({"loss_f": torch.tensor(1.0)})
            
        # 2. fine-level loss (RLE) - MODIFIED
        loss_f_rle = self.compute_rle_loss(data, self.loss_config["fine_weight"])
        if loss_f_rle is not None:
            loss += loss_f_rle
            loss_scalars.update({"loss_f_rle": loss_f_rle.clone().detach().cpu()})

        # 3. fine-level loss (BCE) - ADDED
        loss_f_bce = self.compute_bce_loss(data, self.bce_weight)
        if loss_f_bce is not None:
            loss += loss_f_bce
            loss_scalars.update({"loss_f_bce": loss_f_bce.clone().detach().cpu()})

        # 3. cycle consistency (train-only, optional)
        cycle_log = torch.tensor(0.0)
        if self.cycle_weight > 0:
            cycle_loss = self.compute_cycle_loss(data)
            if cycle_loss is not None:
                loss = loss + self.cycle_weight * cycle_loss
                cycle_log = cycle_loss.detach()
            loss_scalars.update({"loss_cycle": cycle_log.clone().cpu()})

        # 4. epipolar loss-only regularization (does not change forward graph)
        epi_log = torch.tensor(0.0)
        gs = int(data.get("global_step", getattr(self, "_internal_step", 0)))
        warm_ratio = min(1.0, gs / max(1, self.epi_warmup_steps))
        lambda_epi_now = float(self.lambda_epi) * warm_ratio
        if self.lambda_epi > 0:
            loss_epi_val = self.compute_epi_loss(data)
            if loss_epi_val is not None:
                loss = loss + lambda_epi_now * loss_epi_val
                epi_log = loss_epi_val.detach()
            loss_scalars.update({"loss_epi": epi_log.clone().cpu(),
                             "lambda_epi": torch.tensor(lambda_epi_now).cpu()})
        
        data.update({"loss": loss, "loss_scalars": loss_scalars})
