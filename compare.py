#!/usr/bin/env python
import argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd.functional import jvp

# ─────────────────────────────────────────────────────────────────────────────
#  ESRK‐15 tableau (as before)
# ─────────────────────────────────────────────────────────────────────────────
a_np = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0243586417803786,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0258303808904268,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0667956303329210,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0140960387721938,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0412105997557866,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0149469583607297,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.414086419082813,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00395908281378477,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.480561088337756,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.319660987317690,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0.00668808071535874,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.0374638233561973,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.422645975498266,0.439499983548480,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.422645975498266,0.0327614907498598,0.367805790222090,0]
]
b_np = [
    0.035898932499408134,0.035898932499408134,0.035898932499408134,0.035898932499408134,
    0.035898932499408134,0.035898932499408134,0.035898932499408134,0.035898932499408134,
    0.006612457947210495,0.21674686949693006,0.0,0.42264597549826616,
    0.03276149074985981,0.0330623263939421,0.0009799086295048407
]
a = torch.tensor(a_np, dtype=torch.float32)
b = torch.tensor(b_np, dtype=torch.float32)
c = a.tril(-1).sum(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Activation + Vector‐field
# ─────────────────────────────────────────────────────────────────────────────
class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25*x - (1/12)*x**2 + (1/48)*x**3)

def make_f(ch, use_groupnorm=False, approx_act=False):
    act = ApproxSiLU() if approx_act else nn.SiLU()
    layers = [nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    if use_groupnorm:
        layers.append(nn.GroupNorm(8, ch))
    layers += [act, nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    net = nn.Sequential(*layers)
    class VF(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
        def forward(self, t, x):
            return self.net(x)
    return VF()

# ─────────────────────────────────────────────────────────────────────────────
#  Euler, RK4 & ESRK Blocks
# ─────────────────────────────────────────────────────────────────────────────
class EulerBlock(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__()
        self.f, self.h, self.steps = f, h, steps
    def forward(self, x, t0=0.0):
        t = t0
        for _ in range(self.steps):
            x = x + self.h * self.f(t, x)
            t += self.h
        return x

class RK4Block(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__()
        self.f, self.h, self.steps = f, h, steps
    def forward(self, x, t0=0.0):
        t = t0
        for _ in range(self.steps):
            x = self._step(x, t)
            t += self.h
        return x
    def _step(self, x, t):
        h, f = self.h, self.f
        k1 = f(t, x)
        k2 = f(t + 0.5*h, x + 0.5*h*k1)
        k3 = f(t + 0.5*h, x + 0.5*h*k2)
        k4 = f(t +     h, x +     h*k3)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class ESRKBlock(nn.Module):
    def __init__(self, f, a, b, c, h=1.0, steps=1):
        super().__init__()
        self.f = f; self.h = float(h); self.steps = int(steps)
        A_l = torch.as_tensor(a, dtype=torch.float32).tril(-1)
        self.register_buffer("A_l", A_l)
        self.register_buffer("b",   torch.as_tensor(b, dtype=torch.float32))
        self.register_buffer("c",   torch.as_tensor(c, dtype=torch.float32))
        self.s = A_l.shape[0]
    def forward(self, x, t0=0.0):
        t = t0
        for _ in range(self.steps):
            x = self._step(x, t)
            t += self.h
        return x
    def _step(self, x_in, t):
        h, A_l, b, c, s = self.h, self.A_l, self.b, self.c, self.s
        k_list = []
        for i in range(s):
            Y = x_in if i==0 else x_in + sum(h*A_l[i,j]*k_list[j]
                                              for j in range(i) if A_l[i,j]!=0)
            k = self.f(t + c[i]*h, Y)
            k_list.append(k)
        return x_in + sum(h*b[i]*k_list[i] for i in range(s) if b[i]!=0)

# ─────────────────────────────────────────────────────────────────────────────
#  TinyNODE
# ─────────────────────────────────────────────────────────────────────────────
class TinyNODE(nn.Module):
    def __init__(self, solver='esrk', width=32, use_groupnorm=False, approx_act=False,
                 h=1.0, steps=1):
        super().__init__()
        act = ApproxSiLU() if approx_act else nn.SiLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            act,
            nn.MaxPool2d(2)
        )
        f = make_f(width, use_groupnorm, approx_act)
        if solver == 'euler':
            self.ode = EulerBlock(f, h=1/15.0, steps=60)
        elif solver == 'rk4':
            self.ode = RK4Block(f, h=h, steps=steps)
        elif solver == 'esrk':
            self.ode = ESRKBlock(f, a, b, c, h=h, steps=steps)
        else:
            raise ValueError(f"Unknown solver: {solver!r}")
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 10, bias=False)
        )
    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# ─────────────────────────────────────────────────────────────────────────────
#  Spectral‑norm estimate via power iteration of JVP
# ─────────────────────────────────────────────────────────────────────────────
def estimate_spectral_norm(model, x, iters=5):
    model.eval()
    with torch.no_grad():
        z = model.encoder(x)
    z = z.detach().requires_grad_(True)
    def vf(z_in): return model.ode.f(0.0, z_in)
    v = torch.randn_like(z)
    sigma = 0.0
    for _ in range(iters):
        _, jv = jvp(vf, (z,), (v,), create_graph=False)
        sigma = jv.norm().item()
        v = jv / (sigma + 1e-12)
    return sigma

# ─────────────────────────────────────────────────────────────────────────────
#  Train / Eval / Spectral hooked together
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, corr = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        corr += (p==y).sum().item()
        total += y.size(0)
    return 100.0 * corr / total

def train_loop(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # data
    tfm = transforms.Compose([
        transforms.RandomCrop(32,4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    train_ds = datasets.CIFAR10(args.data, True, transform=tfm, download=True)
    val_ds   = datasets.CIFAR10(args.data, False, transform=tfm, download=True)
    tr_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4)
    va_ld = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)

    # model
    model = TinyNODE(solver=args.solver, width=32,
                     use_groupnorm=args.groupnorm,
                     approx_act=args.approx_act,
                     h=args.h, steps=args.steps).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossF = nn.CrossEntropyLoss()

    start = time.time()
    for ep in range(args.epochs):
        ep0 = time.time()
        model.train()
        for x,y in tr_ld:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossF(model(x), y)
            loss.backward()
            opt.step()

        train_acc = validate(model, tr_ld, device)
        val_acc   = validate(model, va_ld, device)
        xb,_ = next(iter(va_ld))
        σ = estimate_spectral_norm(model, xb[:args.spec_batch].to(device),
                                   iters=args.spec_iters)

        print(f"Epoch {ep:02d}  "
              f"train={train_acc:.2f}%  val={val_acc:.2f}%  "
              f"σ_est≈{σ:.3f}  time={time.time()-ep0:.1f}s")
        sched.step()

    print(f"Total training: {(time.time()-start)/60:.1f} min")
    torch.save(model.state_dict(), f"tiny_{args.solver}.pth")
    print("Saved checkpoint.")

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--solver',    choices=['euler','rk4','esrk'], default='esrk')
    p.add_argument('--epochs',    type=int,   default=50)
    p.add_argument('--batch',     type=int,   default=128)
    p.add_argument('--h',         type=float, default=10.0)
    p.add_argument('--steps',     type=int,   default=1)
    p.add_argument('--groupnorm', action='store_true')
    p.add_argument('--approx_act',action='store_true')
    p.add_argument('--data',      type=str,   default='./cifar_data')
    p.add_argument('--spec_iters',type=int,   default=5)
    p.add_argument('--spec_batch',type=int,   default=1)
    return p.parse_args()

if __name__=='__main__':
    args = parse()
    train_loop(args)
