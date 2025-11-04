
"""
MNIST → INR Pipeline (Enhanced)
-------------------------------
- Fits a small SIREN INR per image (subset of MNIST).
- Builds a dataset of INR parameter vectors (theta) + labels.
- Trains a simple classifier on theta to test discriminative power.
- Saves:
    * inrs.pt (X, y)
    * recon_samples.png
    * clf_curve.png
    * clf_log.csv (epoch, train_acc, val_acc)
    * inr_losses.png (optional: per-epoch INR losses for first K samples)
- Resumable: if inrs.pt exists, skips INR fitting (use --force_refit to redo).

Quick start:
python mnist_inr_pipeline.py --subset 200 --epochs_per_inr 200 --classifier_epochs 20

"""

import argparse, os, math
from pathlib import Path
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

try:
    from torchvision import datasets, transforms
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

# ---------------- SIREN ----------------
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def siren_linear(in_f, out_f, is_first=False, w0=30.0):
    layer = nn.Linear(in_f, out_f)
    with torch.no_grad():
        if is_first:
            layer.weight.uniform_(-1/in_f, 1/in_f)
        else:
            bound = math.sqrt(6/in_f) / w0
            layer.weight.uniform_(-bound, bound)
        nn.init.zeros_(layer.bias)
    return layer

class SIREN(nn.Module):
    def __init__(self, in_f=2, hidden=64, depth=3, out_f=1, w0=30.0):
        super().__init__()
        layers = []
        for i in range(depth):
            is_first = (i == 0)
            layers += [siren_linear(in_f if is_first else hidden, hidden, is_first=is_first, w0=w0), Sine()]
        layers += [nn.Linear(hidden, out_f)]
        self.net = nn.Sequential(*layers)
    def forward(self, coords):
        return self.net(coords)

# ---------------- Utilities ----------------
def make_coord_grid(H, W, device):
    ys = torch.linspace(-1, 1, steps=H, device=device)
    xs = torch.linspace(-1, 1, steps=W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).view(-1, 2)  # [H*W,2]
    return coords

def load_mnist_subset(n=200, seed=0, device="cpu"):
    if HAS_TORCHVISION:
        tfm = transforms.Compose([transforms.ToTensor()])
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        # Roughly class-balanced subset
        per = max(1, n // 10)
        got = {i:0 for i in range(10)}
        imgs, labs = [], []
        for img, lab in ds:
            if got[lab] < per:
                imgs.append(img[0]); labs.append(lab); got[lab]+=1
            if sum(got.values()) >= per*10: break
        import torch as T
        imgs = T.stack(imgs)[:n].to(device)  # [n,28,28]
        labs = T.tensor(labs)[:n].to(device)
    else:
        # Fallback: synthetic blobs; adequate for pipeline checks
        import torch as T
        imgs = T.zeros(n, 28, 28)
        labs = T.randint(0, 10, (n,))
        yy, xx = T.meshgrid(T.arange(28), T.arange(28), indexing="ij")
        for i in range(n):
            cx = T.randint(6, 22, (1,)).item()
            cy = T.randint(6, 22, (1,)).item()
            r = T.randint(3, 6, (1,)).item()
            mask = (xx-cx).float().pow(2) + (yy-cy).float().pow(2) < r*r
            imgs[i][mask] = 1.0
        imgs = imgs.to(device); labs = labs.to(device)
    return imgs, labs

def train_single_inr(img, coords, epochs=200, hidden=64, depth=3, lr=1e-3, weight_decay=0.0, device="cpu", log_losses=False):
    target = img.view(-1,1).to(device)
    net = SIREN(in_f=2, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for ep in range(epochs):
        pred = net(coords)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_losses:
            losses.append(loss.item())
    theta = torch.cat([p.detach().flatten() for p in net.parameters()])
    with torch.no_grad():
        recon = net(coords).view(img.shape[0], img.shape[1]).clamp(0,1).detach().cpu()
    return net, theta.cpu(), recon, losses

class INRParamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

def plot_grid(imgs, recons, path):
    k = min(8, imgs.shape[0])
    plt.figure(figsize=(8,4))
    for i in range(k):
        plt.subplot(2,k,i+1); plt.imshow(imgs[i].cpu(), cmap="gray"); plt.axis("off"); plt.title("GT")
        plt.subplot(2,k,k+i+1); plt.imshow(recons[i].cpu(), cmap="gray"); plt.axis("off"); plt.title("Recon")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def train_classifier(ds, epochs=20, batch_size=32, lr=1e-3, device="cpu", out_png=None, out_csv=None):
    import pandas as pd
    n = len(ds)
    n_val = max(1, int(0.2*n))
    train_set, val_set = random_split(ds, [n-n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    in_dim = ds[0][0].numel()
    model = SimpleClassifier(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    tr_hist, va_hist = [], []

    rows = []
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb)
            loss.backward(); opt.step()
        # eval
        def eval(loader):
            model.eval(); tot=0; corr=0
            with torch.no_grad():
                for xb,yb in loader:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb).argmax(-1)
                    corr += (pred==yb).sum().item()
                    tot += yb.numel()
            return corr/tot
        tr_acc = eval(train_loader); va_acc = eval(val_loader)
        tr_hist.append(tr_acc); va_hist.append(va_acc)
        rows.append({"epoch": ep+1, "train_acc": tr_acc, "val_acc": va_acc})
        print(f"epoch {ep+1}/{epochs}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")

    if out_png:
        plt.figure()
        plt.plot(range(1,epochs+1), tr_hist, label="train_acc")
        plt.plot(range(1,epochs+1), va_hist, label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("INR Classifier Accuracy")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()
    if out_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    return tr_hist, va_hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--subset", type=int, default=200)
    ap.add_argument("--epochs_per_inr", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--classifier_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--force_refit", action="store_true", help="refit INRs even if inrs.pt exists")
    ap.add_argument("--log_inr_k", type=int, default=4, help="log and plot INR losses for first K samples (0=disable)")
    args = ap.parse_args()

    outdir = Path("./inr_mnist"); outdir.mkdir(parents=True, exist_ok=True)
    device = args.device
    H = W = 28
    coords = make_coord_grid(H,W,device)

    # If dataset exists and not forcing refit, load and skip fitting
    inrs_path = outdir / "inrs.pt"
    if inrs_path.exists() and not args.force_refit:
        print("Loading existing INR dataset:", inrs_path)
        blob = torch.load(inrs_path, map_location="cpu")
        X, y = blob["X"], blob["y"]
    else:
        print("Loading images and fitting INRs...")
        imgs, labels = load_mnist_subset(n=args.subset, device=device)
        thetas, recons, all_loss_traces = [], [], []
        for i in range(args.subset):
            log_losses = (i < args.log_inr_k and args.log_inr_k > 0)
            _, theta, recon, losses = train_single_inr(
                imgs[i], coords, epochs=args.epochs_per_inr, hidden=args.hidden, depth=args.depth,
                lr=args.lr, weight_decay=args.weight_decay, device=device, log_losses=log_losses
            )
            thetas.append(theta); recons.append(recon)
            if log_losses: all_loss_traces.append(losses)
            if (i+1) % 10 == 0:
                print(f"  fit INR {i+1}/{args.subset}")
        # pad to same length
        max_len = max(t.shape[0] for t in thetas)
        X = torch.zeros(len(thetas), max_len)
        for i,t in enumerate(thetas):
            X[i,:t.shape[0]] = t
        y = labels.cpu()
        torch.save({"X": X, "y": y}, inrs_path)
        print("Saved INR parameters to", inrs_path)
        # Recon grid
        import torch as T
        plot_grid(imgs.cpu(), T.stack(recons,0), outdir / "recon_samples.png")
        # INR loss plot for first K
        if all_loss_traces:
            plt.figure()
            for j, losses in enumerate(all_loss_traces):
                plt.plot(range(1, len(losses)+1), losses, label=f"img{j}")
            plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("INR Fit Loss (first K samples)")
            plt.yscale("log"); plt.legend(); plt.tight_layout(); plt.savefig(outdir / "inr_losses.png", dpi=160); plt.close()

    # Train classifier
    ds = INRParamDataset(X, y)
    clf_png = outdir / "clf_curve.png"
    clf_csv = outdir / "clf_log.csv"
    _, _ = train_classifier(ds, epochs=args.classifier_epochs, batch_size=args.batch_size, device=device, out_png=clf_png, out_csv=clf_csv)
    print("Saved:", clf_png, "and", clf_csv)
    print("Done.")

if __name__ == "__main__":
    main()
