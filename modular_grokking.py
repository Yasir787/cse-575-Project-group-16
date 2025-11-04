"""
Modular Grokking Experiment (PyTorch)
-------------------------------------
Task: predict (a + b) mod N from integer pairs.
Logs per-step train/val accuracy to CSV and saves curves as PNG.
Supports MLP and tiny Transformer, with/without weight decay.

Usage examples:
python modular_grokking.py --model mlp --modulo 97 --steps 15000 --weight_decay 0.0
python modular_grokking.py --model transformer --modulo 97 --steps 20000 --weight_decay 1e-3

Outputs (in ./outputs/mod<N>_<model>_wd<wd>/):
  - train_log.csv, val_log.csv
  - curve.png
"""

import argparse, os, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# ---------------- Dataset ----------------
class ModAddDataset(Dataset):
    def __init__(self, N:int, split:str="train", split_frac:float=0.5, size:int=None, seed:int=1337):
        assert split in ["train", "val"]
        self.N = N
        rng = np.random.default_rng(seed)
        # generate all pairs (a,b)
        pairs = [(a,b) for a in range(N) for b in range(N)]
        rng.shuffle(pairs)
        n_total = len(pairs)
        n_train = int(split_frac * n_total)
        if split == "train":
            pairs = pairs[:n_train]
        else:
            pairs = pairs[n_train:]
        if size is not None:
            pairs = pairs[:size]
        self.x = torch.tensor(pairs, dtype=torch.long)
        self.y = (self.x[:,0] + self.x[:,1]) % N
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------------- Models ----------------
class MLP(nn.Module):
    def __init__(self, N, emb=128):
        super().__init__()
        self.emb_a = nn.Embedding(N, emb)
        self.emb_b = nn.Embedding(N, emb)
        self.net = nn.Sequential(
            nn.Linear(2*emb, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, N)
        )
    def forward(self, x):
        a, b = x[:,0], x[:,1]
        ea = self.emb_a(a)
        eb = self.emb_b(b)
        h = torch.cat([ea, eb], dim=-1)
        return self.net(h)

class TinyTransformer(nn.Module):
    def __init__(self, N, d_model=128, nhead=4, num_layers=2, dim_ff=256):
        super().__init__()
        self.token_emb = nn.Embedding(N, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, N)
        self.pos = nn.Parameter(torch.zeros(1, 2, d_model))
    def forward(self, x):
        # x: [B,2] integer tokens
        h = self.token_emb(x) + self.pos
        h = self.encoder(h)           # [B,2,d]
        h = h.mean(dim=1)             # pool
        return self.cls(h)

# ---------------- Utils ----------------
def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def log_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(list(row.keys()))
        w.writerow(list(row.values()))

def plot_curves(train_csv, val_csv, out_png):
    import pandas as pd
    tr = pd.read_csv(train_csv)
    va = pd.read_csv(val_csv)
    plt.figure()
    plt.plot(tr["step"], tr["acc"], label="train_acc")
    plt.plot(va["step"], va["acc"], label="val_acc")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Grokking Curve: Train vs Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------------- Train ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp","transformer"], default="mlp")
    p.add_argument("--modulo", type=int, default=97)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--train_subset", type=int, default=None, help="optional: limit train set size for speed")
    p.add_argument("--val_subset", type=int, default=None, help="optional: limit val set size for speed")
    args = p.parse_args()

    N = args.modulo
    outdir = Path(f"./outputs/mod{N}_{args.model}_wd{args.weight_decay}")
    outdir.mkdir(parents=True, exist_ok=True)
    train_log = outdir / "train_log.csv"
    val_log = outdir / "val_log.csv"
    curve_png = outdir / "curve.png"

    train_ds = ModAddDataset(N, "train", split_frac=0.5, size=args.train_subset)
    val_ds   = ModAddDataset(N, "val",   split_frac=0.5, size=args.val_subset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "mlp":
        model = MLP(N)
    else:
        model = TinyTransformer(N)
    model.to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    step = 0
    model.train()
    while step < args.steps:
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % args.eval_every == 0:
                # train acc on current batch
                tr_acc = accuracy(logits.detach(), yb)
                log_row(train_log, {"step": step, "acc": tr_acc, "loss": float(loss.detach().cpu())})

                # full val
                model.eval()
                tot_corr, tot = 0, 0
                with torch.no_grad():
                    for xv, yv in val_loader:
                        xv = xv.to(args.device); yv = yv.to(args.device)
                        lv = model(xv)
                        tot_corr += (lv.argmax(-1) == yv).sum().item()
                        tot += yv.numel()
                va_acc = tot_corr / tot
                log_row(val_log, {"step": step, "acc": va_acc, "loss": float("nan")})
                model.train()

            step += 1
            if step >= args.steps:
                break

    # plot curves
    try:
        plot_curves(str(train_log), str(val_log), str(curve_png))
        print(f"Saved curve to {curve_png}")
    except Exception as e:
        print("Plotting failed:", e)

    print("Done. Logs at:", outdir)

if __name__ == "__main__":
    main()
