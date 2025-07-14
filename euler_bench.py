#Simple Euler to thorw int there


import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------
# 1. Taylor–SiLU (cheap, MCU-friendly)
# -----------------------------------------------------------
class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25 * x - (1/12)*x**2 + (1/48)*x**3)

# -----------------------------------------------------------
# 2. ODE vector field
# -----------------------------------------------------------
class ODEFunc(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        )
    def forward(self, t, x):
        return self.net(x)

# -----------------------------------------------------------
# 3. Euler solver block
# -----------------------------------------------------------
class EulerBlock(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__()
        self.f = f
        self.h = float(h)
        self.steps = int(steps)
    @torch.cuda.amp.custom_fwd
    def forward(self, x, t0=0.0):
        t = t0
        for _ in range(self.steps):
            x = x + self.h * self.f(t, x)
            t += self.h
        return x

# -----------------------------------------------------------
# 4. TinyODEEuler model (~19k params)
# -----------------------------------------------------------
class TinyODEEuler(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        self.ode = EulerBlock(ODEFunc(32), h=1.0, steps=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10, bias=False)
        )
    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# -----------------------------------------------------------
# 5. Parameter counter
# -----------------------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total

# -----------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------
def train_float(num_epochs=100, log_every=100):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyODEEuler().to(dev)
    count_parameters(model)

    tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
    ])
    tr_ds = datasets.CIFAR10("./cifar_data", True, download=True, transform=tfm)
    te_ds = datasets.CIFAR10("./cifar_data", False, download=True, transform=tfm)
    tr_ld = DataLoader(tr_ds, 128, shuffle=True, num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, 256, shuffle=False, num_workers=4, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    lossF = nn.CrossEntropyLoss()
    n_train = len(tr_ds)

    for ep in range(num_epochs):
        model.train()
        for bi, (x, y) in enumerate(tr_ld):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossF(model(x), y)
            loss.backward()
            opt.step()
            if bi % log_every == 0:
                done = bi * x.size(0)
                print(f"Epoch {ep:02d} [{done}/{n_train}] Loss: {loss.item():.4f}")

        # Validation
        model.eval(); corr = tot = 0
        with torch.no_grad():
            for x, y in te_ld:
                p = model(x.to(dev)).argmax(1).cpu()
                corr += (p == y).sum().item(); tot += y.size(0)
        print(f"Epoch {ep:02d} acc {100*corr/tot:.2f}%")

        sched.step()

    torch.save(model.state_dict(), "tiny_ode_euler.pth")
    return model.cpu()

# -----------------------------------------------------------
if __name__ == "__main__":
    m_f = train_float(num_epochs=100, log_every=50)
