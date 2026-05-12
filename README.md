# anti-poaching-patrol-framework

A transparent, white-box mathematical framework for anti-poaching resource allocation, applied to Etosha National Park, Namibia.

> **Associated preprint:** Chan Ka Hin, Ao Long Nam, Loi Weng Kin, Tse Long Tin, Cheng Sok Kin. *A Game-Theoretic and Dynamical-Systems Framework for Anti-Poaching Resource Allocation: A Case Study of Etosha National Park.* EcoEvoRxiv, 2026. [DOI: to be added after upload]

---

## Overview

This repository implements a three-layer pipeline that addresses two interlinked anti-poaching management questions jointly:

1. **Where** should a limited ranger workforce be deployed to maximise conservation return?
2. **How many** rangers are dynamically sufficient to prevent prey population decline?

All modelling choices are explicit and auditable. No black-box ML/DL is used.

---

## Framework Architecture

```
Layer 1 — WPP Field (wpp_maut_engine.py)
    Ecological priority map using nonlinear MAUT:
    • Endangered utility:  U_R = 1 − exp(−α · max(N − N_crit, 0) / N_init,R)
    • Abundant utility:    U_E = ln(1 + κN) / ln(1 + κN_init,E)
    • Waterhole attraction: Σ_h 10 · exp(−0.5 · d_h)
    • NHPP threat intensity: λ(s,t) = λ₀(t) · exp(β⊤ X(s,t))
    • Weighted combination: WPP = (0.65·U_R + 0.35·U_E) · attraction · (1 + λ)
         ↓
Layer 2 — Spatial Allocation (ssg_dro_optimizer.py)
    SLSQP optimisation of WPP-weighted interception objective:
    • max Σᵢ (1 − exp(−λ_det · τᵢ / μᵢ)) · WPPᵢ
    • subject to: Σ τᵢ = 295, 0 ≤ τᵢ ≤ 50
    • Technology: λ_full = 0.25 (full tech), λ_low = 0.08 (degraded)
    • Seasonal friction: μ_savanna = 1.0, μ_pan,dry = 3.0, μ_pan,wet = 15.0
         ↓
Layer 3 — Stability Analysis (jacobian_stability.py)
    3D prey–predator–poacher ODE system:
    • Ṅ = rN(1 − N/K) − α_pred·NP − α_poach·NZ
    • Ṗ = η·NP − m·P
    • Ż = ρ·NZ − σ·τ·Z − m_z·Z
    Jacobian eigenvalue crossing identifies the critical staffing threshold T*
```

---

## Parameter Reference

### Ecology (Layer 1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.08 | Endangered species utility scale |
| `kappa` | 0.02 | Abundant species utility scale |
| `n_crit` | 120 | Critical population floor (endangered) |
| `n_initial_endangered` | 300 | Initial endangered population (black rhino) |
| `n_initial_abundant` | 2500 | Initial abundant population (elephant) |
| `nhpp_beta` | [−0.45, −0.25, −0.15, −0.18] | NHPP covariates: distance to water, edge, road, friction |
| `nhpp_lambda0_dry` | 0.55 | Baseline threat intensity, dry season |
| `nhpp_lambda0_wet` | 0.35 | Baseline threat intensity, wet season |

### Optimisation (Layer 2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda_full_tech` | 0.25 | Detection rate with full technology |
| `lambda_low_tech` | 0.08 | Detection rate with degraded technology |
| `total_personnel` | 295 | Total ranger workforce |
| `tau_max` | 50 | Per-cell ranger cap |
| `mu_savanna` | 1.0 | Baseline terrain friction |
| `mu_pan_dry` | 3.0 | Pan friction, dry season |
| `mu_pan_wet` | 15.0 | Pan friction, wet season |

### Population Dynamics (Layer 3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `r_prey` | 0.18 | Prey intrinsic growth rate (yr⁻¹) |
| `k_prey` | 1800 | Prey carrying capacity |
| `alpha_predation` | 0.0055 | Natural predation rate |
| `alpha_poaching` | 0.0060 | Poaching extraction rate |
| `eta_conversion` | 0.00015 | Predator conversion efficiency |
| `m_predator` | 0.09 | Predator natural mortality |
| `rho_poacher_growth` | 0.010 | Poacher activity growth rate |
| `sigma_suppression` | 450.0 | Ranger suppression coefficient |
| `m_poacher` | 0.06 | Poacher natural exit rate |

All parameters are stored in `data/raw_pdf_specs.json`.

---

## Installation

```bash
git clone https://github.com/skckenneth/anti-poaching-patrol-framework.git
cd anti-poaching-patrol-framework
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, numpy, scipy, matplotlib, seaborn

---

## Usage

Run the full pipeline (all experiments, all figures):

```bash
python main.py --mode all
```

Run individual modules:

```bash
python main.py --mode baseline   # Layer 1-3 baseline + stability threshold
python main.py --mode seasonal   # Dry vs wet season deployment comparison
python main.py --mode tech       # Technology degradation sensitivity
python main.py --mode global     # Congo + Himalaya adaptation probes
```

All outputs are written to `outputs/`.

---

## Output Files

| File | Description |
|------|-------------|
| `protection_strategy.png` | Layer-2 patrol density heatmap |
| `stability_bifurcation.png` | Jacobian eigenvalue vs. staffing (Layer 3) |
| `phase_space_portrait.png` | Phase portrait of ODE trajectories |
| `lorenz_curve.png` | Patrol allocation Lorenz curve (Gini ≈ 0.77) |
| `marginal_utility_curve.png` | Marginal capture gain per additional ranger |
| `seasonal_comparison.png` | Dry/wet season deployment shift map |
| `tech_degradation.png` | Interception loss under λ degradation |
| `topology_wpp_3d_surface.png` | 3D WPP field surface |
| `voronoi_patrol_zones.png` | Voronoi patrol sector partition |
| `pareto_front.png` | Capture–equity Pareto frontier |
| `scenario_radar.png` | Multi-metric scenario comparison |
| `evaluation_summary.json` | All numeric metrics |
| `run_metrics.json` | Pipeline run output summary |

---

## Repository Structure

```
anti-poaching-patrol-framework/
├── main.py                    # Unified pipeline entrypoint
├── scenario_runner.py         # Seasonal and tech degradation scenarios
├── global_adaption.py         # Cross-region adaptation (Congo, Himalayas)
├── requirements.txt
├── data/
│   └── raw_pdf_specs.json     # All model parameters and park specs
├── src/
│   ├── wpp_maut_engine.py     # Layer 1: WPP field generation
│   ├── ssg_dro_optimizer.py   # Layer 2: SLSQP patrol allocation
│   ├── jacobian_stability.py  # Layer 3: 3D ODE stability analysis
│   ├── rhc_temporal_shift.py  # Receding-horizon temporal weighting
│   └── advanced_analytics.py  # Gini, Lorenz, Pareto, Monte Carlo
├── utils/
│   ├── topology_constructor.py  # Etosha grid + waterhole placement
│   └── spatial_friction.py      # Seasonal terrain friction maps
├── visualizations/
│   ├── bifurcation_plots.py   # Stability boundary figures
│   ├── paper_figures.py       # Phase portrait, 3D surface, Lorenz, Pareto
│   ├── spatial_heatmaps.py    # Patrol density heatmaps
│   └── style_guide.py         # Academic figure style (300 dpi)
└── outputs/                   # Generated figures and metrics (gitignored)
```

---

## Reproducing Paper Results

The figures in the associated preprint are generated by:

```bash
python main.py --mode all
```

Expected runtime: approximately 2–5 minutes on a standard laptop.

The critical staffing threshold T* is printed to stdout and saved in `outputs/run_metrics.json` under the key `critical_staff_threshold`.

---

## Notes on Modelling Choices

**Why SLSQP instead of Wasserstein DRO?**
The spatial allocation (Layer 2) uses `scipy.optimize.minimize` with SLSQP, which optimises the deterministic WPP-weighted interception objective directly. This is computationally tractable and transparent. A full Wasserstein DRO dual reformulation (Esfahani & Kuhn 2018) would add robustness against distributional ambiguity in the poaching incident distribution; this is noted as a future extension.

**Why 3D ODE in Layer 3?**
The stability analysis uses a three-population system (prey × natural predator × poacher) rather than a simplified 2D prey–poacher model. Including the predator compartment captures the ecological reality that lion and leopard populations in Etosha interact with prey dynamics on the same timescale as poaching pressure.

**Grid resolution**
The default grid uses 5 km × 5 km cells (resulting in a 30 × 30 grid for Etosha's ~22,935 km²). This is deliberately coarser than a 1 km × 1 km grid to keep the SLSQP problem tractable; results are qualitatively robust to resolution changes.

---

## Citation

If you use this code, please cite the associated preprint:

```bibtex
@misc{chan2026antipoaching,
  author    = {Chan, Ka Hin and Ao, Long Nam and Loi, Weng Kin and
               Tse, Long Tin and Cheng, Sok Kin},
  title     = {A Game-Theoretic and Dynamical-Systems Framework for
               Anti-Poaching Resource Allocation: A Case Study of
               Etosha National Park},
  year      = {2026},
  publisher = {EcoEvoRxiv},
  doi       = {[to be added after upload]}
}
```

---

## Licence

This code is released under the [Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE) licence, consistent with the associated preprint.
