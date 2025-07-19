#!/usr/bin/env python
import argparse, time, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------
#  Activation (optional approx SiLU)
# -----------------------------------------------------------
class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25*x - (1/12)*x**2 + (1/48)*x**3)

# -----------------------------------------------------------
#  Vector field  f(t,x)
# -----------------------------------------------------------
def make_f(ch, use_groupnorm=False, approx_act=False):
    act    = ApproxSiLU() if approx_act else nn.SiLU()
    layers = [nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    if use_groupnorm:
        layers.append(nn.GroupNorm(8, ch))
    layers += [act, nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    net = nn.Sequential(*layers)

    class VF(nn.Module):
        def __init__(self): super().__init__(); self.net = net
        def forward(self, t, x): return self.net(x)      #  t is ignored

    return VF()

# -----------------------------------------------------------
#  Blocks
# -----------------------------------------------------------
class EulerBlock(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f, self.h, self.steps = f, float(h), int(steps)
    def forward(self, x, t0=0.0):
        t, h, f, s = t0, self.h, self.f, self.steps
        for _ in range(s):
            x = x + h * f(t, x); t += h
        return x

class RK4Block(nn.Module):
    """Classic 4‑stage Runge–Kutta."""
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f, self.h, self.steps = f, float(h), int(steps)
    def forward(self, x, t0=0.0):
        f, h, s = self.f, self.h, self.steps
        t = t0
        for _ in range(s):
            k1 = f(t        , x)
            k2 = f(t + 0.5*h, x + 0.5*h*k1)
            k3 = f(t + 0.5*h, x + 0.5*h*k2)
            k4 = f(t +     h, x +     h*k3)
            x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            t += h
        return x

# -----------------------------------------------------------
#  Tiny NODE model
# -----------------------------------------------------------
class TinyNODE(nn.Module):
    def __init__(self, solver='rk4', width=32,
                 use_groupnorm=False, approx_act=False,
                 h=0.25, steps=4):
        super().__init__()
        act = ApproxSiLU() if approx_act else nn.SiLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            act,
            nn.MaxPool2d(2)      # 32×32 → 16×16
        )
        f = make_f(width, use_groupnorm, approx_act)

        if solver == 'rk4':
            self.ode = RK4Block(f, h=h, steps=steps)
        elif solver == 'euler':
            self.ode = EulerBlock(f, h=h, steps=steps)
        else:
            raise ValueError(f"unknown solver {solver!r}")

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 10, bias=False)
        )

    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# -----------------------------------------------------------
#  Data
# -----------------------------------------------------------
def get_loaders(data_dir, batch=128):
    tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    tr_ds = datasets.CIFAR10(data_dir, train=True , download=True, transform=tfm)
    te_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    tr_ld = DataLoader(tr_ds, batch, shuffle=True , num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, 256 , shuffle=False, num_workers=4, pin_memory=True)
    return tr_ld, te_ld, len(tr_ds)

# -----------------------------------------------------------
#  Train / Eval helpers
# -----------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device):
    model.eval(); corr = tot = 0
    for x,y in loader:
        p  = model(x.to(device)).argmax(1).cpu()
        corr += (p == y).sum().item(); tot += y.size(0)
    return 100.0 * corr / tot

def train(num_epochs, solver, width, h, steps,
          use_groupnorm, approx_act, data_dir, log_every=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = TinyNODE(solver, width, use_groupnorm, approx_act, h, steps).to(device)

    tr_ld, te_ld, N = get_loaders(data_dir)
    opt   = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    lossF = nn.CrossEntropyLoss()

    for ep in range(num_epochs):
        model.train(); t0 = time.time()
        for bi,(x,y) in enumerate(tr_ld):
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            lossF(model(x), y).backward(); opt.step()
            if bi % log_every == 0:
                print(f'Ep{ep:02d} [{bi*len(x)}/{N}]  loss={lossF(model(x),y).item():.4f}')
        acc = validate(model, te_ld, device)
        print(f'Ep{ep:02d}  acc={acc:.2f}%  (epoch time {time.time()-t0:.1f}s)')
        sched.step()

# -----------------------------------------------------------
#  CLI
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--solver', choices=['rk4','euler'], default='rk4')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--width' , type=int, default=32)
    p.add_argument('--h'     , type=float, default=0.25)
    p.add_argument('--steps' , type=int,   default=4)
    p.add_argument('--groupnorm', action='store_true')
    p.add_argument('--approx-act', action='store_true')
    p.add_argument('--data', type=str, default='./cifar_data')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args.epochs, args.solver, args.width, args.h, args.steps,
          args.groupnorm, args.approx_act, args.data)
