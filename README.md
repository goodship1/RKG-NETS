# 📦 RKG-NETS

**RKG-NETS** is a lightweight framework for building **small, memory-efficient Neural ODEs** based on **Explicit Stabilized Runge-Kutta (ESRK)** methods inspired by the paper:
> "Runge–Kutta–Gegenbauer explicit methods for advection-diffusion problems" (O’Sullivan, 2019)  
> [arXiv:1712.03971](https://arxiv.org/abs/1712.03971)

This project focuses on:
- Neural ODE architectures with **sub-100k parameters** (e.g., CIFAR-10 model with only ~19k params).  
- Support for **large stable step sizes (e.g., h=1 or more) without numerical blow-up**.  
- Efficient forward pass using **two-register low-storage schemes** for MCU/embedded targets.  
- Compatibility with **post-training quantization** (e.g., PyTorch INT8) for deployment.

---

## 🔬 Key features:
- ✅ **Van der Houwen style ESRK-15 scheme**
- ✅ Extremely compact models:  
  Example: CIFAR-10 classifier with `19,616` trainable parameters, achieving competitive accuracy (~50%+ after 5 epochs baseline, 80%+ achievable).

---

## 📐 Mathematical foundations

The ESRK solver used in this project derives from **Runge–Kutta–Gegenbauer (RKG) methods**. 

### Canonical form:
The ODE system is:
\[
w'(t) = f(w(t))
\]

The ESRK scheme advances \( w \) using multiple explicit stages:
\[
W^{(l)} = W^{(0)} + T \sum_{j=1}^l a_j f(W^{(j-1)})
\]

### Stability polynomial:
The stability function:
\[
R^{\nu,N}_M(z) = G^{\nu,N}_M \left( 1 + \frac{2z}{\beta^{\nu,N}_M} \right)
\]

Where:
- \( G^{\nu,N}_M(z) \) is a Gegenbauer polynomial expansion ensuring stability and accuracy.
- Parameters \( (\nu, N, M) \) control **shape of the stability region**.
- Coefficients ensure order conditions up to order \( N \).

### Extended stability region:
- Stability along negative real axis ≈ \( M^2 \), allowing **large stable step sizes**.
- **Tunability for advection-diffusion problems and neural ODEs with stiff gradients**.

This gives **robust performance at h=1 or even larger** — outperforming conventional explicit schemes in stiff regimes typical of neural ODEs.

---

## 📊 Example results:
- CIFAR-10, ESRK-15 block, 19k params:
  - Large stable step sizes (`h=1`, `steps=1`)  
  - Fast training convergence  
  - Quantizable to INT8 for low-resource inference

---

## 📖 Reference:
- O’Sullivan, S. (2019).  
  *Runge–Kutta–Gegenbauer explicit methods for advection-diffusion problems*.  
  [Journal of Computational Physics, DOI: 10.1016/j.jcp.2019.03.001](https://doi.org/10.1016/j.jcp.2019.03.001)  
  [arXiv:1712.03971](https://arxiv.org/abs/1712.03971)

---

## ⚡ Usage:
```bash
# Train baseline
python th.py

# Count parameters:
Total trainable parameters: 19,616
