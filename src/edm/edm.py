from ..utils.misc import detect_NaN
from .head.fine_matching import FineMatching, FineMatchingV2
from .head.coarse_matching import CoarseMatching
from .neck.neck import CIM, DepthAnythingFeatureExtractor
from .backbone.resnet import ResNet18
from einops.einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
torch.set_float32_matmul_precision("highest")  # highest (defualt) high medium

# ADDED: A simple head to predict matchability from fine features
class SaliencyHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class EDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.local_resolution = self.config["local_resolution"]
        self.bi_directional_refine = self.config["fine"]["bi_directional_refine"]
        self.deploy = self.config["deploy"]
        self.topk = config["coarse"]["topk"]

        # Modules
        self.backbone = ResNet18(config)
        self.neck = CIM(config)
        self.coarse_matching = CoarseMatching(config)
        self.fine_matching = FineMatching(config)
        # self.fine_matching = FineMatchingV2(config)

        # ADDED: Saliency head to predict matchability from fine features
        fine_feature_dim = config["backbone"]["block_dims"][-3]
        self.saliency_head = SaliencyHead(fine_feature_dim)

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        if self.deploy:
            image0, image1 = data.split(1, 1)
            data = {"image0": image0, "image1": image1}

        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        # 1. Feature Extraction
        if data["hw0_i"] == data["hw1_i"]:
            # faster & better BN convergence
            feats = self.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0))
            f8, f16, f32, f8_fine = feats
            ms_feats = f8, f16, f32
            feat_f0, feat_f1 = f8_fine.chunk(2)
        else:
            # handle different input shapes
            # raise ValueError("image0 and image1 should have the same shape.")
            feats0, feats1 = self.backbone(data["image0"]), self.backbone(
                data["image1"]
            )
            f8_0, f16_0, f32_0, feat_f0 = feats0
            f8_1, f16_1, f32_1, feat_f1 = feats1
            ms_feats = f8_0, f16_0, f32_0, f8_1, f16_1, f32_1

        # --- ADDED: Predict Saliency Score Map ---
        # Predict the "matchability" score for each fine-level pixel
        saliency_map0 = self.saliency_head(feat_f0) # Shape: [B, 1, H/8, W/8]
        saliency_map1 = self.saliency_head(feat_f1)

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        # 2.  Feature Interaction & Multi-Scale Fusion
        # Optional train-time-only hidden-state injection (e.g., Depth Anything v2 hidden)
        if self.config['use_hidden']:
            if self.config['depth_from_extract']:
                hidden0 = data.get("depth_feat0", None)
                hidden1 = data.get("depth_feat1", None)
            else:
                hidden0 = data.get("da_hidden0", None)
                hidden1 = data.get("da_hidden1", None)
        else:
            hidden0=hidden1=None
        feat_c0, feat_c1 = self.neck(ms_feats, mask_c0, mask_c1,
                                     hidden0=hidden0, hidden1=hidden1,
                                     inject_hidden=self.config['use_hidden'])
        
        # ADDED: Store original 2D feature shapes before flattening
        h_c0, w_c0 = feat_c0.shape[2:]
        h_c1, w_c1 = feat_c1.shape[2:]
        
        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_c0.shape[2:] * self.config["local_resolution"],
                "hw1_f": feat_c1.shape[2:] * self.config["local_resolution"],
            }
        )
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")
        feat_f0 = rearrange(feat_f0, "n c h w -> n (h w) c")
        feat_f1 = rearrange(feat_f1, "n c h w -> n (h w) c")

        # detect NaN during mixed precision training
        if self.config["mp"] and (
            torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))
        ):
            detect_NaN(feat_c0, feat_c1)

        # 3. Coarse-Level Matching
        conf_matrix = self.coarse_matching(
            feat_c0,
            feat_c1,
            data,
            mask_c0=(
                mask_c0.view(mask_c0.size(0), -
                             1) if mask_c0 is not None else mask_c0
            ),
            mask_c1=(
                mask_c1.view(mask_c1.size(0), -
                             1) if mask_c1 is not None else mask_c1
            ),
        )

        if self.deploy:
            k = self.topk
            row_max_val, row_max_idx = torch.max(conf_matrix, dim=2)
            topk_val, topk_idx = torch.topk(row_max_val, k, dim=1)

            b_ids = (
                torch.arange(conf_matrix.shape[0], device=conf_matrix.device)
                .unsqueeze(1)
                .repeat(1, k)
                .flatten()
            )
            i_ids = topk_idx.flatten()
            j_ids = row_max_idx[b_ids, i_ids].flatten()
            mconf = conf_matrix[b_ids, i_ids, j_ids]
   
            scale = data["hw0_i"][0] / data["hw0_c"][0]
            scale0 = scale * \
                data["scale0"][b_ids] if "scale0" in data else scale
            scale1 = scale * \
                data["scale1"][b_ids] if "scale1" in data else scale
            mkpts0_c = (
                torch.stack(
                    [
                        i_ids % data["hw0_c"][1],
                        torch.div(i_ids, data["hw0_c"][1],
                                  rounding_mode="floor"),
                    ],
                    dim=1,
                )
                * scale0
            )
            mkpts1_c = (
                torch.stack(
                    [
                        j_ids % data["hw1_c"][1],
                        torch.div(j_ids, data["hw1_c"][1],
                                  rounding_mode="floor"),
                    ],
                    dim=1,
                )
                * scale1
            )

            data.update(
                {
                    "mconf": mconf,
                    "mkpts0_c": mkpts0_c,
                    "mkpts1_c": mkpts1_c,
                    "b_ids": b_ids,
                    "i_ids": i_ids,
                    "j_ids": j_ids,
                }
            )
        # -------
        # 4. Fine-Level Matching
        # K0 = data["i_ids"].shape[0] // data["bs"]
        # K1 = data["j_ids"].shape[0] // data["bs"]
        # feat_f0 = feat_f0[data["b_ids"], data["i_ids"]
        #                   ].reshape(data["bs"], K0, -1)
        # feat_f1 = feat_f1[data["b_ids"], data["j_ids"]
        #                   ].reshape(data["bs"], K1, -1)
        # feat_c0 = feat_c0[data["b_ids"], data["i_ids"]
        #                   ].reshape(data["bs"], K0, -1)
        # feat_c1 = feat_c1[data["b_ids"], data["j_ids"]
        #                   ].reshape(data["bs"], K1, -1)

        # if self.bi_directional_refine:
        #     # Bidirectional Refinement
        #     offset, score = self.fine_matching(
        #         torch.cat([feat_f0, feat_f1], dim=1),
        #         torch.cat([feat_f1, feat_f0], dim=1),
        #         torch.cat([feat_c0, feat_c1], dim=1),
        #         torch.cat([feat_c1, feat_c0], dim=1),
        #         data,
        #     )
        # else:
        #     offset, score = self.fine_matching(
        #         feat_f0, feat_f1, feat_c0, feat_c1, data)

        # if self.deploy:
        #     if self.bi_directional_refine:
        #         fine_offset01, fine_offset10 = offset.chunk(2)
        #         fine_score01, fine_score10 = score.unsqueeze(dim=1).chunk(2)
        #         output = torch.cat(
        #             [mkpts0_c, mkpts1_c, fine_offset01, fine_offset10, fine_score01, fine_score10, mconf.unsqueeze(dim=1)], 1) # [K, 11]
        #     else:
        #         output = torch.cat(
        #             [mkpts0_c, mkpts1_c, offset, score, mconf.unsqueeze(dim=1)], 1)
        #     return output
        # -------

        # -------
        # # Do not re-index features. Instead, reshape the original fine features to 2D
        # h_f0, w_f0 = data['hw0_i'][0] // 8, data['hw0_i'][1] // 8
        # feat_f0_2d = rearrange(feat_f0, 'n (h w) c -> n c h w', h=h_f0, w=w_f0)
        
        # h_f1, w_f1 = data['hw1_i'][0] // 8, data['hw1_i'][1] // 8
        # feat_f1_2d = rearrange(feat_f1, 'n (h w) c -> n c h w', h=h_f1, w=w_f1)
        
        # # Prepare fine-level masks
        # mask_f0, mask_f1 = None, None
        # if 'mask0' in data:
        #     mask_f0 = F.interpolate(data['mask0'].unsqueeze(1).float(), size=(h_f0, w_f0), mode='nearest').squeeze(1).bool()
        #     mask_f1 = F.interpolate(data['mask1'].unsqueeze(1).float(), size=(h_f1, w_f1), mode='nearest').squeeze(1).bool()
        
        # # Call the new fine_matching module
        # data = self.fine_matching(feat_f0_2d, feat_f1_2d, data, mask_f0, mask_f1)
        
        # # The old bi-directional logic here is removed, as it's now handled internally.
        
        # if self.deploy:
        #     # Deployment logic needs to be updated based on the new outputs
        #     # This is a simplified example of how to reconstruct the final matches
        #     mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']
        #     pred_offset = data['pred_offset']
        #     pred_score = data['pred_score']
        #     mconf = data['mconf']
            
        #     offset_01, offset_10 = torch.chunk(pred_offset, 2, dim=0)
        #     score_01, score_10 = torch.chunk(pred_score, 2, dim=0)
            
        #     mkpts0_f = mkpts0_c
        #     mkpts1_f = mkpts1_c + offset_01

        #     # A simple filtering for deployment
        #     mask = mconf > self.config['coarse']['mconf_thr']
        #     # You can add more filtering based on `score_01` here
            
        #     return torch.cat(
        #         [mkpts0_f[mask], mkpts1_f[mask], offset_01, score_01, mconf[mask].unsqueeze(dim=1)], 1)
        # return data
                # --- Dynamic Feature Selection Logic ---
        # Instead of just taking the center point's feature, we find a better one
        # using the saliency map. This logic is fully vectorized (no Python loops).
        # ---------

        # a. Get coarse match indices and feature map dimensions
        b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']
        bs = data['bs']
        h_c, w_c = data['hw0_c']
        h_f, w_f = h_c * self.local_resolution, w_c * self.local_resolution

        # b. Get the 8x8 saliency patches corresponding to each coarse match
        # Unfold the saliency map into non-overlapping 8x8 patches
        saliency_patches0 = F.unfold(saliency_map0, kernel_size=self.local_resolution, stride=self.local_resolution)
        # Shape: [B, 1 * 8 * 8, Hc * Wc] -> [B, 64, L]
        saliency_patches1 = F.unfold(saliency_map1, kernel_size=self.local_resolution, stride=self.local_resolution)

        # c. Find the location of the max value (local offset) within each patch
        local_argmax_idx0 = torch.argmax(saliency_patches0, dim=1) # Shape: [B, L]
        local_argmax_idx1 = torch.argmax(saliency_patches1, dim=1) # Shape: [B, L]
        
        # d. Gather the local offsets for our M coarse matches
        M = b_ids.shape[0]
        local_offset_idx0 = local_argmax_idx0[b_ids, i_ids] # Shape: [M]
        local_offset_idx1 = local_argmax_idx1[b_ids, j_ids] # Shape: [M]

        # Convert 1D local offset index to 2D local offset (dy, dx)
        local_offset_y0 = torch.div(local_offset_idx0, self.local_resolution, rounding_mode='floor')
        local_offset_x0 = local_offset_idx0 % self.local_resolution
        
        local_offset_y1 = torch.div(local_offset_idx1, self.local_resolution, rounding_mode='floor')
        local_offset_x1 = local_offset_idx1 % self.local_resolution

        # e. Calculate the new global 1D indices for the fine feature map
        coarse_coords0_x = i_ids % w_c
        coarse_coords0_y = torch.div(i_ids, w_c, rounding_mode='floor')
        new_fine_coords_x0 = coarse_coords0_x * self.local_resolution + local_offset_x0
        new_fine_coords_y0 = coarse_coords0_y * self.local_resolution + local_offset_y0
        new_fine_indices0 = new_fine_coords_y0 * w_f + new_fine_coords_x0
        
        coarse_coords1_x = j_ids % w_c
        coarse_coords1_y = torch.div(j_ids, w_c, rounding_mode='floor')
        new_fine_coords_x1 = coarse_coords1_x * self.local_resolution + local_offset_x1
        new_fine_coords_y1 = coarse_coords1_y * self.local_resolution + local_offset_y1
        new_fine_indices1 = new_fine_coords_y1 * w_f + new_fine_coords_x1

        # f. Gather the features from the new, dynamically selected points
        # feat_f0 and feat_f1 are flat [B, L_fine, C]
        feat_f0_dynamic = torch.gather(feat_f0, 1, new_fine_indices0.unsqueeze(1).unsqueeze(2).expand(-1, -1, feat_f0.shape[2]))
        feat_f1_dynamic = torch.gather(feat_f1, 1, new_fine_indices1.unsqueeze(1).unsqueeze(2).expand(-1, -1, feat_f1.shape[2]))

        # Also get coarse features for the original FineMatching module
        K = M // bs
        feat_c0_original = feat_c0[b_ids, i_ids].view(bs, K, -1)
        feat_c1_original = feat_c1[b_ids, j_ids].view(bs, K, -1)
        
        feat_f0_dynamic = feat_f0_dynamic.view(bs, K, -1)
        feat_f1_dynamic = feat_f1_dynamic.view(bs, K, -1)

        # Call the original fine_matching module with the 'upgraded' feature vectors
        if self.bi_directional_refine:
            offset, score = self.fine_matching(
                torch.cat([feat_f0_dynamic, feat_f1_dynamic], dim=1),
                torch.cat([feat_f1_dynamic, feat_f0_dynamic], dim=1),
                torch.cat([feat_c0_original, feat_c1_original], dim=1),
                torch.cat([feat_c1_original, feat_c0_original], dim=1),
                data,
            )
        else:
            offset, score = self.fine_matching(
                feat_f0_dynamic, feat_f1_dynamic, feat_c0_original, feat_c1_original, data)
        return data # Return data for the loss function
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)