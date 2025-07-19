# extract_depth_feature.py  (torchrun-compatible)
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
class DepthAnythingTokenExtractor(torch.nn.Module):
    """Return [B, 1369, 384] token tensor without CLS"""
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        if dist.get_rank() == 0:
            print(f"[INFO] Loading {model_name} …")
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, pil_imgs):                       # list[PIL]
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outs   = self.model(**inputs, output_hidden_states=True)
        return outs.hidden_states[-1][:, 1:]           # [B,1370,384]


# ------------------ io utils -----------------------
def save_pth(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), path)

def save_h5(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens", data=tensor.cpu().numpy(), compression="gzip")


# ------------------ worker logic -------------------
def worker(rank, world_size, args):
    if rank == 0:
        print(f"[INFO] Process-group initialised (world_size={world_size})")
    torch.cuda.set_device(rank)

    extractor = DepthAnythingTokenExtractor().to(rank)
    extractor = DDP(extractor, device_ids=[rank])

    # Glob all image paths and split
    all_imgs = sorted(
        p for p in Path(args.image_dir).rglob("*")
        if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
    )
    shard = all_imgs[rank::world_size]     # round-robin

    # Process images in batches
    B = args.batch_size
    for i in range(0, len(shard), B):
        batch_paths = shard[i : i + B]
        pil_imgs = [Image.open(p).convert("RGB").resize((832, 832), Image.BILINEAR) for p in batch_paths]
        tokens   = extractor(pil_imgs)          # [B,1369,384]
        # print(tokens)
        # print(pil_imgs[0])
        # print(pil_imgs)

        for tok, pth in zip(tokens, batch_paths):
            base = pth.with_suffix("")
            save_pth(tok.unsqueeze(0), base.with_suffix(".pth"))
            save_h5(tok.unsqueeze(0), base.with_suffix(".h5"))

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