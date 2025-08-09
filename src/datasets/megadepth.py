import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, read_megadepth_rgb, read_megadepth_depth_feature, read_megadepth_depth_fusion


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        root_dir,
        npz_path,
        mode="train",
        min_overlap_score=0.4,
        img_resize=None,
        df=None,
        img_padding=False,
        depth_padding=False,
        augment_fn=None,
        fp16=False,
        pre_extracted_depth=False,
        depth_map_fusion=False,
        **kwargs
    ):
        """
        Manage one scene(npz_path) of MegaDepth dataset.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split(".")[0]

        # prepare scene_info and pair_info
        if mode == "test" and min_overlap_score != 0:
            logger.warning(
                "You are using `min_overlap_score`!=0 in test mode. Set to 0."
            )
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info["pair_infos"].copy()

        del self.scene_info["pair_infos"]
        self.pair_infos = [
            pair_info
            for pair_info in self.pair_infos
            if pair_info[1] > min_overlap_score
        ]

        # parameters for image resizing, padding and depthmap padding
        if mode == "train":
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = (
            2000 if depth_padding else None
        )  # the upperbound of depthmaps size in megadepth.

        # for training EDM
        self.augment_fn = augment_fn if mode == "train" else None
        self.coarse_scale = getattr(kwargs, "coarse_scale", 0.125)

        self.fp16 = fp16
        self.pre_extracted_depth = pre_extracted_depth
        self.depth_map_fusion = depth_map_fusion

    def _norm(self, d, target_size=None):
        if d.dim() == 2:
            d = d.unsqueeze(0) # (h, w) -> (1, h, w)
        d = d.float()
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = (d > 0) & torch.isfinite(d)
        valid_pixels = d[valid_mask]
        if valid_pixels.numel() == 0:
            # No valid pixels, return zero tensor of same shape
            d_norm = torch.zeros_like(d)
        else:
            low = torch.quantile(valid_pixels, 0.02)
            high = torch.quantile(valid_pixels, 0.98)
            d_clipped = torch.clamp(d, low, high)
            d_norm = (d_clipped - low) / (high - low + 1e-6)
            d_norm = 1.0 - d_norm  # invert to match GT depth convention (close=0, far=1)

        # Resize if requested (e.g. depth padded to 2000×2000)
        if target_size is not None and d_norm.shape[-2:] != target_size:
            d_norm = F.interpolate(d_norm.unsqueeze(0), size=target_size,
                              mode="bilinear", align_corners=False)[0]

        return d_norm.float()       # Save VRAM


    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(
            self.root_dir, self.scene_info["image_paths"][idx0])
        img_name1 = osp.join(
            self.root_dir, self.scene_info["image_paths"][idx1])

        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None
        )
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None
        )
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        if not self.pre_extracted_depth:
            image0_rgb, _, _ = read_megadepth_rgb(
                img_name0, self.img_resize, self.df, self.img_padding, None
            )
            image1_rgb, _, _ = read_megadepth_rgb(
                img_name1, self.img_resize, self.df, self.img_padding, None
            )       

        # read depth. shape: (h, w)
        if self.mode in ["train", "val", "test"] and self.depth_map_fusion:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info["depth_paths"][idx0]),
                pad_to=self.depth_max_size,
            )
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info["depth_paths"][idx1]),
                pad_to=self.depth_max_size,
            )
        else:
            depth0 = depth1 = torch.tensor([])

        # read depth anything v2 depth map. shape: (h, w)
        if self.mode in ["train", "val", "test"] and self.depth_map_fusion:
            depth0_da = read_megadepth_depth_fusion(
                osp.join(self.root_dir, self.scene_info["image_paths"][idx0]),
                pad_to=self.depth_max_size,
            )
            depth1_da = read_megadepth_depth_fusion(
                osp.join(self.root_dir, self.scene_info["image_paths"][idx1]),
                pad_to=self.depth_max_size,
            )
        else:
            depth0_da = depth1_da = torch.tensor([])

        # read pre-extracted depth features: 1, 1369, 384 (fixed from DepthAnything-v2-small)
        if self.mode in ["train", "val", "test"] and self.pre_extracted_depth:
            depth_feat0 = read_megadepth_depth_feature(img_name0)
            depth_feat1 = read_megadepth_depth_feature(img_name1)

        # read intrinsics of original size
        K_0 = torch.tensor(
            self.scene_info["intrinsics"][idx0].copy(), dtype=torch.float
        ).reshape(3, 3)
        K_1 = torch.tensor(
            self.scene_info["intrinsics"][idx1].copy(), dtype=torch.float
        ).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info["poses"][idx0]
        T1 = self.scene_info["poses"][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[
            :4, :4
        ]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        if self.fp16:
            image0, image1, depth0, depth1, scale0, scale1 = map(
                lambda x: x.half(), [image0, image1,
                                     depth0, depth1, scale0, scale1]
            )
        data = {
            "image0": image0,  # (1, h, w)
            "depth0": depth0,  # (h, w)
            "image1": image1,
            "depth1": depth1,
            "T_0to1": T_0to1,  # (4, 4)
            "T_1to0": T_1to0,
            "K0": K_0,  # (3, 3)
            "K1": K_1,
            "scale0": scale0,  # [scale_w, scale_h]
            "scale1": scale1,
            "dataset_name": "MegaDepth",
            "scene_id": self.scene_id,
            "pair_id": idx,
            "pair_names": (
                self.scene_info["image_paths"][idx0],
                self.scene_info["image_paths"][idx1],
            ),
        }
        # for training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(
                    torch.stack([mask0, mask1], dim=0)[None].float(),
                    scale_factor=self.coarse_scale,
                    mode="nearest",
                    recompute_scale_factor=False,
                )[0].bool()
            data.update({"mask0": ts_mask_0, "mask1": ts_mask_1})
        if self.depth_map_fusion:
            target_size = image0.shape[-2:]  # (H, W) of the grayscale image
            data.update({
                "depth0_norm": self._norm(depth0_da, target_size),
                "depth1_norm": self._norm(depth1_da, target_size),
            })
            return data
        # If using pre-extracted depth features, add them to data
        if self.pre_extracted_depth:
            data.update({"depth_feat0": depth_feat0, "depth_feat1": depth_feat1})

        # Or add RGB images for depth extraction, it will be read when needed
        elif not self.depth_map_fusion:
            data.update({
                "depth_feat_image0": image0_rgb,
                "depth_feat_image1": image1_rgb,
            })
            
        return data
