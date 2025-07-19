# multi_gpu_extract_depth_tokens.py
import os, argparse, h5py
from pathlib import Path
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# --------------------- model wrapper ---------------------
class DepthAnythingTokenExtractor(torch.nn.Module):
    """
    outputs.hidden_states[-1] → [B, 577, 384]  (CLS + 576 tokens)
    After remove cls token, return [B, 1370, 384]
    """
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        super().__init__()
        print(f"Loading model {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, pil_imgs):                          # list[ PIL ]
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outs = self.model(**inputs, output_hidden_states=True)
        tokens = outs.hidden_states[-1][:, 1:]             # remove CLS
        return tokens  # [B, 1370, 384]

# ------------------ save utils ------------------------
def save_pth(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), path)

def save_h5(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens", data=tensor.cpu().numpy(), compression="gzip")

# ------------------ worker process --------------------
def worker(rank, world_size, args):
    # 1) Init distribution
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2) Prepare model
    print(f"Rank {rank} initializing model...")
    extractor = DepthAnythingTokenExtractor().to(rank)
    extractor = DDP(extractor, device_ids=[rank])

    # 3) Image path shard
    print(f"Rank {rank} preparing image paths...")
    all_imgs = sorted(
        [p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    )
    shard = all_imgs[rank::world_size]  # round-robin

    # 4) Extract & save
    batch_size = args.batch_size
    for i in range(0, len(shard), batch_size):
        batch_paths = shard[i : i + batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        tokens = extractor(pil_imgs)  # [B, 1370, 384]
        print(pil_imgs[0])
        print(pil_imgs)
        print(tokens)
        break

        for tok, pth in zip(tokens, batch_paths):
            base = pth.with_suffix("")
            save_pth(tok.unsqueeze(0), base.with_suffix(".pth"))
            save_h5(tok.unsqueeze(0), base.with_suffix(".h5"))

    dist.destroy_process_group()

# ------------------------ main -----------------------
def main():
    import lovely_tensors
    lovely_tensors.monkey_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True, help="Image root path")
    ap.add_argument("--batch_size", type=int, default=8, help="Images per GPU forward pass")
    args = ap.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("GPU not detected")
    mp.spawn(worker, args=(world_size, args), nprocs=world_size)

if __name__ == "__main__":
    main()