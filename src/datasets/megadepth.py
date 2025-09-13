import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from typing import Optional

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth

 # ---- Optional hidden-state utils (train-time only injection) ----

def _tokens_to_feature_map(hidden_seq: torch.Tensor, maybe_has_cls: bool = False) -> torch.Tensor:
    """
    Convert ViT-like hidden sequence
    Input-hidden state  : [B, N, C]
    Return-4D map       : [B, C, H, W]
    If maybe_has_cls and N-1 is a perfect square, drop the first token.
    """
    assert hidden_seq.dim() == 3, f"Expected [B,N,C], got {hidden_seq.shape}"
    B, N, C = hidden_seq.shape
    tokens = hidden_seq
    if maybe_has_cls and int((N - 1) ** 0.5) ** 2 == (N - 1):
        tokens = hidden_seq[:, 1:, :]
        N = N - 1
    s = int(N ** 0.5)
    assert s * s == N, f"Token count {N} is not a perfect square."
    maps = tokens.view(B, s, s, C).permute(0, 3, 1, 2).contiguous()  # [B,C,s,s]
    return maps


def _hidden_sidecar_path(img_path: str, *, replace_ext: bool, suffix: str, ext: str) -> str:
    """
    Build sidecar path for hidden.
    - If replace_ext=True: replace basename extension with `suffix` (e.g., .png -> .pth)
    - Else: append `ext` to the original path (legacy behavior)
    """
    import os.path as osp
    if replace_ext:
        root, _ = osp.splitext(img_path)
        return root + suffix
    return img_path + ext


def _load_hidden_sidecar(img_path: str, *, replace_ext: bool, suffix: str, ext: str) -> Optional[torch.Tensor]:
    """
    Load sidecar tensor via torch.load using the constructed path.
    """
    sidecar = _hidden_sidecar_path(img_path, replace_ext=replace_ext, suffix=suffix, ext=ext)
    try:
        return torch.load(sidecar, map_location="cpu")
    except Exception:
        return None
    

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
        use_hidden=False,
        hidden_ext=".hidden.pt",            # legacy: appended
        hidden_suffix=".pth",               # new: replace original image ext with this suffix
        hidden_replace_ext=True,             # honor the user's rule by default
        hidden_maybe_has_cls=False,
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
        # optional train-time hidden prior
        self.use_hidden = use_hidden
        self.hidden_ext = hidden_ext
        self.hidden_suffix = hidden_suffix
        self.hidden_replace_ext = hidden_replace_ext
        self.hidden_maybe_has_cls = hidden_maybe_has_cls

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

        # --- Optional: load train-time hidden states (Depth Anything v2 etc.) ---
        da_hidden0 = da_hidden1 = None
        if self.mode == "train" and self.use_hidden:
            # Support both legacy and new hidden sidecar naming
            raw_h0 = _load_hidden_sidecar(
                img_name0,
                replace_ext=self.hidden_replace_ext,
                suffix=self.hidden_suffix,
                ext=self.hidden_ext,
            )
            raw_h1 = _load_hidden_sidecar(
                img_name1,
                replace_ext=self.hidden_replace_ext,
                suffix=self.hidden_suffix,
                ext=self.hidden_ext,
            )
            if isinstance(raw_h0, torch.Tensor) and isinstance(raw_h1, torch.Tensor):
                # Accept either [B,N,C] or [C,H,W]; normalize to [B,C,H,W]
                def _to_4d(x: torch.Tensor) -> torch.Tensor:
                    if x.dim() == 3:  # [B,N,C]
                        return _tokens_to_feature_map(x, maybe_has_cls=self.hidden_maybe_has_cls).float()
                    elif x.dim() == 4:  # [B,C,H,W]
                        return x.float()
                    else:
                        raise ValueError(f"Unsupported hidden shape: {tuple(x.shape)}")
                try:
                    da_hidden0 = _to_4d(raw_h0)
                    da_hidden1 = _to_4d(raw_h1)
                except Exception as e:
                    logger.warning(f"Failed to format hidden maps: {e}")
                    da_hidden0 = da_hidden1 = None
            else:
                # silently skip if sidecars not present
                pass

        # read depth. shape: (h, w)
        if self.mode in ["train", "val"]:
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
        # inject train-time hidden maps if available
        if da_hidden0 is not None and da_hidden1 is not None:
            if self.fp16:
                da_hidden0 = da_hidden0.half()
                da_hidden1 = da_hidden1.half()
            data["da_hidden0"] = da_hidden0
            data["da_hidden1"] = da_hidden1
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

        return data
