# extract_depth_feature.py  (torchrun-compatible, saves depth maps)
# ------------------------------------------------------------------
# example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   torchrun --standalone --nproc_per_node=4 extract_depth_feature.py \
#            --image_dir /path/to/images --batch_size 8
# ------------------------------------------------------------------
import os, argparse, h5py
from pathlib import Path
from PIL import Image

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ------------------ model wrapper ------------------
class DepthAnythingDepthEstimator(torch.nn.Module):
    """Return depth maps [B, H, W] resized to processor input size"""
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        if dist.get_rank() == 0:
            print(f"[INFO] Loading depth estimator: {model_name} …")
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, pil_imgs):                       # list[PIL.Image.Image]
        # Prepare inputs
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        device = next(self.model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)             # [B,3,H,W]
        # Forward pass
        outputs = self.model(pixel_values=pixel_values)
        pred = outputs.predicted_depth                               # [B,h,w]
        # Upsample to match input resolution
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=pixel_values.shape[2:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)                                                 # [B,H,W]
        return pred


# ------------------ io utils -----------------------
def save_pth(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), path)

def save_h5(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("depth", data=tensor.cpu().numpy(), compression="gzip")


# ------------------ worker logic -------------------
def worker(rank, world_size, args):
    if rank == 0:
        print(f"[INFO] Process-group initialised (world_size={world_size}) — saving depth maps")
    torch.cuda.set_device(rank)

    extractor = DepthAnythingDepthEstimator().to(rank)
    extractor = DDP(extractor, device_ids=[rank])

    # Glob all image paths and split
    all_imgs = sorted(
        p for p in Path(args.image_dir).rglob("*")
        if p.suffix.lower() == ".jpg" and not p.with_suffix(".npy").exists()  # adjust suffix as needed
    )
    print(f"[INFO] Found {len(all_imgs)} images to process.")
    shard = all_imgs[rank::world_size]     # round-robin

    # Process images in batches
    B = args.batch_size
    from tqdm import tqdm
    for i in tqdm(range(0, len(shard), B), desc="Processing batches"):
        batch_paths = shard[i : i + B]
        # For MegaDepth training setup (matches default 832×832 in config)
        pil_imgs = [Image.open(p).convert("RGB").resize((832, 832), Image.BILINEAR) for p in batch_paths]
        # If using ScanNet, you may prefer (480, 640)
        # pil_imgs = [Image.open(p).convert("RGB").resize((480, 640), Image.BILINEAR) for p in batch_paths]
        depths   = extractor(pil_imgs)          # [B,H,W], float32
        for depth, pth in zip(depths, batch_paths):
            base = pth.with_suffix("")
            # Save as .npy for easy loading elsewhere; change to .pth/.h5 if preferred
            npy_path = base.with_suffix(".npy")
            pth_path = base.with_suffix(".pth")
            # Numpy
            import numpy as _np
            # _np.save(npy_path, depth.cpu().numpy())
            # Torch (optional)
            # save_pth(depth, pth_path)
            # HDF5 (optional, comment out if not needed)
            save_h5(depth, base.with_suffix(".h5"))

    dist.barrier()
    if rank == 0:
        print("[INFO] All ranks finished.")


# --------------------------- main -------------------
def main():
    import lovely_tensors
    lovely_tensors.monkey_patch()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # torchrun 이 주입한 env
    rank        = int(os.environ["RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])
    local_rank  = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    worker(local_rank, world_size, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()