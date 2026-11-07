"""Re-implementation of the dataset paper's baseline (MONAI residual UNet, 512x512 resize,
dual-expert MaskedDiceLoss, Adam 1e-4, 25 epochs, batch 32) evaluated two ways:

  as_published : thresholds RAW LOGITS at 0.5 (no softmax), nearest-neighbour resize, compares the
                 predicted 'wall-only' class with the FULL expert wall - reproduces Table 5 behaviour
  corrected    : softmax probabilities, bilinear resize to native size, hierarchical decoding,
                 full-wall vs full-wall comparison (same evaluator as all our models)

python baselines/official_unet_fixed.py [--epochs 25] [--folds 0 1 2 3 4]
-> results/official_unet_corrected/oof_predictions.csv, results/official_unet_corrected/as_published.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frpv.config import load_config, resolve_path  # noqa: E402
from frpv.data.meta import decode_labels, load_meta  # noqa: E402
from frpv.eval.infer import evaluate_image  # noqa: E402
from frpv.eval.metrics import dice, fi  # noqa: E402

SIZE = 512


class Resized(Dataset):
    def __init__(self, meta, cache):
        self.meta, self.cache = meta.reset_index(drop=True), cache

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        img = torch.from_numpy(np.load(self.cache / "img" / f"{r.stem}.npy")).permute(2, 0, 1).float()
        d = decode_labels(np.load(self.cache / "lab" / f"{r.stem}.npy"))
        x = F.interpolate(img[None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
        x = (x - x.mean()) / (x.std() + 1e-8)                     # MONAI NormalizeIntensityd
        roi = torch.from_numpy(d["roi"].astype(np.float32))

        def tgt(w, f):
            w, f = torch.from_numpy(w.astype(np.float32)), torch.from_numpy(f.astype(np.float32))
            t = torch.stack([(roi - w).clamp(0, 1), (w - f).clamp(0, 1), f])
            return F.interpolate(t[None], size=(SIZE, SIZE), mode="nearest")[0]
        return {"x": x, "t1": tgt(d["w1"], d["f1"]), "t2": tgt(d["w2"], d["f2"]),
                "roi": F.interpolate(roi[None, None], size=(SIZE, SIZE), mode="nearest")[0], "stem": r.stem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_images", type=int, default=None, help="debug")
    a = ap.parse_args()
    from monai.losses import MaskedDiceLoss
    from monai.networks.nets import UNet
    cfg = load_config("configs/base.yaml")
    cache = resolve_path(cfg.paths.cache_dir)
    meta = load_meta(cache)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = resolve_path(cfg.paths.results_dir) / "official_unet_corrected"
    out.mkdir(parents=True, exist_ok=True)
    rows_fixed, rows_pub = [], []
    for k in a.folds:
        tr, te = meta[meta.fold_official != k], meta[meta.fold_official == k]
        if a.max_images:
            tr, te = tr.head(a.max_images), te.head(a.max_images)
        torch.manual_seed(42)
        net = UNet(spatial_dims=2, in_channels=3, out_channels=3, channels=(32, 64, 128, 256, 512),
                   strides=(2, 2, 2, 2), num_res_units=2).to(dev)
        crit = MaskedDiceLoss(softmax=True, to_onehot_y=False)
        opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
        dl = DataLoader(Resized(tr, cache), batch_size=a.batch, shuffle=True, num_workers=a.workers)
        for ep in range(a.epochs):
            net.train()
            tot = 0
            for b in tqdm(dl, desc=f"fold {k} ep {ep+1}", leave=False):
                x, roi = b["x"].to(dev), b["roi"].to(dev)
                o = net(x)
                loss = (crit(o, b["t1"].to(dev), mask=roi) + crit(o, b["t2"].to(dev), mask=roi)) / 2
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            print(f"fold {k} epoch {ep+1}: loss {tot/len(dl):.4f}")
        net.eval()
        with torch.no_grad():
            for b in tqdm(DataLoader(Resized(te, cache), batch_size=1), desc=f"eval fold {k}"):
                stem = b["stem"][0]
                r = meta[meta.stem == stem].iloc[0]
                lab = np.load(cache / "lab" / f"{stem}.npy")
                H, W = lab.shape
                logits = net(b["x"].to(dev))
                # ---- as published
                raw = F.interpolate(logits.float(), size=(H, W), mode="nearest")[0].cpu().numpy()
                d = decode_labels(lab)
                roi = d["roi"]
                pw_only, pfib = (raw[1] > 0.5) & roi, (raw[2] > 0.5) & roi
                e1w, e1f = d["w1"] & roi, d["f1"] & roi
                pfull = (pw_only | pfib)
                gt_fi = np.nanmean([fi(e1w, e1f), fi(d["w2"] & roi, d["f2"] & roi)])
                rows_pub.append(dict(stem=stem, fold=k, dice_wall_e1=dice(pw_only, e1w), dice_fib_e1=dice(pfib, e1f),
                                     FI_pred=fi(pfull, pfib), FI_gt=gt_fi,
                                     FI_abs_err=abs(fi(pfull, pfib) - gt_fi)))
                # ---- corrected
                p = torch.softmax(logits.float(), 1)
                p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0]
                v = torch.stack([p[1] + p[2], p[2]], 0).unsqueeze(0).cpu()
                row, _ = evaluate_image([v], lab, "flat", 1)
                row.update(file=r.file, stem=stem, animal=r.animal, pc=r.pc, gn=r.gn, group=r.group, role=r.role,
                           level=r.level, fold=k, exp="official_unet_corrected")
                rows_fixed.append(row)
    fx, pb = pd.DataFrame(rows_fixed), pd.DataFrame(rows_pub)
    fx.to_csv(out / "oof_predictions.csv", index=False)
    pb.to_csv(out / "as_published.csv", index=False)
    print(f"AS PUBLISHED : dice_wall(wall-only vs full)={pb.dice_wall_e1.mean():.3f} dice_fib={pb.dice_fib_e1.mean():.3f} "
          f"FI_MAE={pb.FI_abs_err.mean():.3f}")
    print(f"CORRECTED    : dice_wall={fx.dice_wall_e1.mean():.3f} dice_fib={fx.dice_fib_e1.mean():.3f} "
          f"FI_MAE={(fx.FI_hat - fx.FI_mean).abs().mean():.4f}")


if __name__ == "__main__":
    main()
