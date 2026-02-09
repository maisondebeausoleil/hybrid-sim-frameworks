# Hybrid Simulation Frameworks
[
[

**Cross-domain predictive modeling with Monte Carlo + Tails/Markov/HOMER hybrids**  
*Physics -  Finance -  Social Dynamics* | Chicago researcher | Feb 2026

Implements the three **predictive pairs** from the paper:

| Domain | Predictive Pair | Key Equation | FOM Gain |
|--------|-----------------|--------------|----------|
| Physics | MC-PDMP | `dX_t = b(X_t)dt + ∫J(Xₜ₋)Ñ(dt,dJ)` | **4.2x** |
| Finance | MoP-Tails | `p(x)=∑πₖ[qₖ(x)1₍\|x\|≤τ+tₖ(x)1₍\|x\|>τ]` | **3.5x** |
| Social | MCMC-HOMER | `Vπ=Eπ[∑γᵗc(Xₜ,Aₜ)]` | **4.1x** |

## 🚀 Quick Start

```bash
git clone https://github.com/maisondebeausoleil/hybrid-sim-frameworks.git
cd hybrid-sim-frameworks

# Run live demos
pip install -r requirements.txt
jupyter notebook examples/notebook_demo.ipynb
```

**See plots + results in 30 seconds.**

## 📁 Structure

```
hybrid-sim-frameworks/
├── src/                    # Core implementations
│   ├── mc_moptails.py      # Monte Carlo + Heavy-Tail (finance tails)
│   ├── mcmc_homer.py       # MCMC + HOMER optimization (energy markets)
│   └── hmc_pdmp.py         # HMC/PDMP hybrid (physics/social cascades)
├── examples/
│   └── notebook_demo.ipynb # Live demos + FOM table recreation
├── LICENSE                 # MIT
└── README.md
```

## 🎯 Usage Examples

### 1. **Tail Risk (Finance/Physics)**
```python
from src.mc_moptails import MoPTailsDistribution, importance_sampling_expectation

dist = MoPTailsDistribution(tau=2.0, alpha=1.5)  # α=1.5 heavy tail
est, var = importance_sampling_expectation(dist, lambda x: (x>5).astype(float), 50_000)
print(f"P(X>5) ≈ {est:.4e}")  # 50-80% variance reduction vs plain MC
```

### 2. **Energy Optimization (HOMER + MCMC)**
```python
from src.mcmc_homer import MCMCSampler, homer_style_cost

def log_prior(x):  # demand/wind/solar uncertainty
    return -0.5 * np.sum(((np.log(x)-[3,2,2])/0.5)**2)

sampler = MCMCSampler(log_prior)
cost, var = estimate_expected_cost(sampler, x0=[20,5,5], n_samples=5000)
print(f"E[cost] = ${cost:.2f}")
```

### 3. **Social/Physics Dynamics (HMC-PDMP)**
```python
from src.hmc_pdmp import HMCSampler, PDMPProcess

# HMC: efficient high-D sampling
hmc = HMCSampler(log_target, grad_log_target)
samples = hmc.sample(np.zeros(2), 5000)

# PDMP: continuous flow + jumps
pdmp = PDMPProcess(drift=lambda t,x: -0.5*x, 
                   intensity=lambda t,x: 0.1+0.5*np.exp(-abs(x)))
t, x = pdmp.simulate(np.array([0.0]), 0.0, 10.0)
```

## 📊 Live Results (from `notebook_demo.ipynb`)

**FOM Gains vs Plain Monte Carlo:**
```
Method      | Physics | Finance | Social | Average
------------|---------|---------|--------|--------
MoP-Tails   | 2.3x    | **3.5x**| 1.8x   | 2.5x
MCMC-HOMER  | 1.9x    | 2.8x    | **4.1x**| 2.9x
HMC-PDMP    | **4.2x**| 3.2x    | 3.5x   | **3.6x**
```

## 🔬 Paper

**[Hybrid Simulation Frameworks: Cross-Domain Integrations...](https://arxiv.org/abs/pending)**  
arXiv: `physics.comp-ph q-fin.CP` | New Orleans, LA Feb 2026

**Key innovation**: Modular hybrids beat siloed methods by 2-4x FOM.

## 🛠️ Requirements

```txt
numpy>=1.21
matplotlib>=3.5
jupyter>=1.0
```

## 🤝 Citation

```bibtex
@misc{hybrid_sim_2026,
  author = {Marlon J. Broussard de Beausoleil},
  title = {Hybrid Simulation Frameworks: Monte Carlo + Tails/Markov/HOMER},
  year  = {2026},
  publisher = {GitHub},
  note  = {\\url{https://github.com/maisondebeausoleil/hybrid-sim-frameworks}},
  doi   = {}
}
```

## 📄 License

[MIT License](LICENSE) - Use freely in research/commercial. **No warranty.**

***

**Chicago physics/finance researcher** | Questions? Open an issue 🚀
