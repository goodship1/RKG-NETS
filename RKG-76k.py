import os, torch, torch.nn as nn
import torch.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------
# 1.  Taylor–SiLU (cheap, MCU-friendly)
# -----------------------------------------------------------
class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25 * x - (1/12)*x**2 + (1/48)*x**3)

# -----------------------------------------------------------
# 2.  ESRK-15  (coefficients unchanged)
# -----------------------------------------------------------
# -----------------------------------------------------------
a_np= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0]]

b_np=[0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.006612457947210495, 0.21674686949693006, 0.0, 0.42264597549826616, 0.03276149074985981, 0.0330623263939421, 0.0009799086295048407]

a = torch.tensor(a_np, dtype=torch.float32)
b = torch.tensor(b_np, dtype=torch.float32)

# Optional – pre-compute c_i = Σ_j<i a_ij  (saves flops)
c = a.tril(-1).sum(1)
print("256")
# ------------------------
# -----------------------------------------------------------
# 3.  ODE vector field  (depthwise → channel-wise convs)
# -----------------------------------------------------------
class ODEFunc(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        )
    def forward(self, t, x):      # t unused but retained
        return self.net(x)

# -----------------------------------------------------------
# 4.  Two-register ESRK-15 block
# -----------------------------------------------------------
class ESRKBlock(nn.Module):
    def __init__(self, f, a, b, c, h=1.0, steps=1):
        super().__init__()
        self.f, self.h, self.steps = f, float(h), int(steps)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("c", c)
        self.s = len(b)
    # Add to ESRKBlock class
    def __getattr__(self, name):
    	if name == '_modules':
        	return {}  # Prevent AttributeError during module traversal
    	return super().__getattr__(name)

    @torch.cuda.amp.custom_fwd
    def forward(self, x, t0: float = 0.0):
        t = t0
        for _ in range(self.steps):
            x = self._step(x, t)
            t += self.h
        return x

    def _step(self, x, t):
      ### Van der houwen 2S-k schemes to keep tensors low 
        x_in   = x                # register-0
        k_prev = None             # register-1
        out    = x.clone()        # accumulate Σ b_i k_i

        for i in range(self.s):
            x_stage = x_in if i == 0 else x_in + self.h * self.a[i, i-1] * k_prev
            k_new   = self.f(t + self.c[i]*self.h, x_stage)
            out     = out + self.h * self.b[i] * k_new
            k_prev  = k_new       # shift registers

        return out

# -----------------------------------------------------------
# 5.  64-channel classifier  (≈ 76 k params)
# -----------------------------------------------------------
class TinyODESRK(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )
        self.ode  = ESRKBlock(ODEFunc(256), a, b, c, h=1, steps=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10, bias=False)
        )
    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# -----------------------------------------------------------
# 6.  Training loop with batch-level logging
# -----------------------------------------------------------
def train_float(num_epochs=100, log_every=100):
    dev   = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyODESRK().to(dev)

    tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    tr_ds = datasets.CIFAR10("./cifar_data",  True,  download=True, transform=tfm)
    te_ds = datasets.CIFAR10("./cifar_data", False, download=True, transform=tfm)
    tr_ld = DataLoader(tr_ds, 128, shuffle=True,  num_workers=4, pin_memory=True)
    te_ld = DataLoader(te_ds, 256, shuffle=False, num_workers=4, pin_memory=True)

    opt   = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    lossF = nn.CrossEntropyLoss()

    n_train = len(tr_ds)
    for ep in range(num_epochs):
        model.train()
        for bi, (x,y) in enumerate(tr_ld):
            x,y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossF(model(x), y)
            loss.backward()
            opt.step()

            if bi % log_every == 0:
                done = bi * x.size(0)
                print(f"Epoch {ep:02d} [{done}/{n_train}] Loss: {loss.item():.4f}")

        # ­­­validation
        model.eval(); corr = tot = 0
        with torch.no_grad():
            for x,y in te_ld:
                p = model(x.to(dev)).argmax(1).cpu()
                corr += (p==y).sum().item(); tot += y.size(0)
        print(f"Epoch {ep:02d}  acc {100*corr/tot:.2f}%")
        sched.step()

    torch.save(model.state_dict(), "tiny_odesrk_float2.pth")
    return model.cpu()

# -----------------------------------------------------------
# 7.  Post-training INT8 quantisation
# -----------------------------------------------------------
def quantise_ptq(model_float):
    fused = torch.quantization.fuse_modules(
        model_float, [['encoder.0','encoder.1']], inplace=False)
    fused.qconfig = tq.get_default_qconfig("xnnpack")
    prep  = tq.prepare(fused, inplace=False)

    # simple  calibration
    with torch.no_grad():
        prep(torch.randn(32,3,32,32))

    qmodel = tq.convert(prep.eval(), inplace=False)
    torch.save(qmodel.state_dict(), "tiny_odesrk_int8.pth")
    kb = os.path.getsize("tiny_odesrk_int8.pth")/1024
    print(f"INT8 checkpoint ≈ {kb:.1f} kB on disk")
    return qmodel

# -----------------------------------------------------------
if __name__ == "__main__":
    m_f = train_float(num_epochs=100, log_every=50)   # expect ≈ 84 % @100 epochs
    quantise_ptq(m_f)
