# anti-poaching-patrol-framework

A transparent, white-box mathematical framework for anti-poaching resource allocation, applied to Etosha National Park, Namibia.

> **Associated preprint:** Chan Ka Hin, Ao Long Nam, Loi Weng Kin, Tse Long Tin, Cheng Sok Kin. *A Game-Theoretic and Dynamical-Systems Framework for Anti-Poaching Resource Allocation: A Case Study of Etosha National Park.* EcoEvoRxiv, 2026. [DOI: to be added after upload]

---

## Overview

This repository implements a three-layer pipeline that addresses two interlinked anti-poaching management questions jointly:

1. **Where** should a limited ranger workforce be deployed to maximise conservation return?
2. **How many** rangers are dynamically sufficient to prevent prey population decline?

Three analytical enhancements go beyond the base three-layer pipeline:

| Enhancement | Description | Key file |
|-------------|-------------|----------|
| **Self-consistent loop** | Iterates Layers 1–2–3 until deployment converges to a fixed-point equilibrium | `src/feedback_loop.py` |
| **Multi-objective Pareto** | Explicitly trades capture efficiency vs. patrol equity; identifies the knee point | `visualizations/enhanced_figures.py` |
| **GBIF data anchoring** | Integrates real *Diceros bicornis* occurrence records (103 within Etosha bbox) from GBIF as a spatial prior | `data/fetch_real_data.py` |

---

## Framework Architecture

```
Layer 1 — WPP Field (src/wpp_maut_engine.py)
    Ecological priority map using nonlinear MAUT:
    • Endangered utility:  U_R = 1 − exp(−α · max(N − N_crit, 0) / N_init,R)
    • Abundant utility:    U_E = ln(1 + κN) / ln(1 + κN_init,E)
    • Waterhole attraction: Σ_h 10 · exp(−0.5 · d_h)
    • NHPP threat intensity: λ(s,t) = λ₀(t) · exp(β⊤ X(s,t))
    • Weighted combination: WPP = (0.65·U_R + 0.35·U_E) · attraction · (1 + λ)
    • Optional GBIF prior: WPP ← WPP · (1 + occurrence_density)
         ↓  ↑  [feedback loop updates λ₀ based on Z*]
Layer 2 — Spatial Allocation (src/ssg_dro_optimizer.py)
    SLSQP optimisation of WPP-weighted interception objective:
    • max Σᵢ (1 − exp(−λ_det · τᵢ / μᵢ)) · WPPᵢ
    • subject to: Σ τᵢ = 295, 0 ≤ τᵢ ≤ 50
         ↓
Layer 3 — Stability Analysis (src/jacobian_stability.py)
    3D prey–predator–poacher ODE system:
    • Ṅ = rN(1 − N/K) − α_pred·NP − α_poach·NZ
    • Ṗ = η·NP − m·P
    • Ż = ρ·NZ − σ·τ·Z − m_z·Z
    Jacobian eigenvalue crossing identifies critical staffing threshold T*
    Equilibrium Z* fed back to update Layer 1 threat intensity
```

---

## Quick Start

```bash
git clone https://github.com/skckenneth/anti-poaching-patrol-framework.git
cd anti-poaching-patrol-framework
pip install -r requirements.txt

# Run base pipeline
python main.py --mode all

# Run three analytical enhancements
python run_enhancements.py
```

All outputs are written to `outputs/`.

---

## Run Modes

```bash
python main.py --mode baseline    # Layer 1-3 baseline + stability threshold
python main.py --mode seasonal    # Dry vs wet season deployment comparison
python main.py --mode tech        # Technology degradation sensitivity
python main.py --mode global      # Congo + Himalaya adaptation probes
python main.py --mode enhanced    # All three enhancements (feedback loop, Pareto, GBIF)
python main.py --mode all         # Everything
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

---

## Output Files

### Base pipeline (`python main.py --mode all`)

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
| `run_metrics.json` | All pipeline metrics |

### Enhancements (`python run_enhancements.py`)

| File | Description |
|------|-------------|
| `feedback_convergence.png` | 4-panel convergence diagnostics for the feedback loop |
| `feedback_deployment_comparison.png` | Open-loop vs self-consistent deployment maps |
| `gbif_wpp_overlay.png` | GBIF occurrence density vs WPP (with and without prior) |
| `pareto_frontier_enhanced.png` | Pareto frontier with knee point, baseline, and feedback solutions |
| `enhancement_metrics.json` | All scalar metrics from the three enhancements |

---

## GBIF Data

Real *Diceros bicornis* occurrence records are fetched automatically from the GBIF API on first run and cached at `data/gbif_rhino_cache.json`. The dataset contains:

- **162 total** southern Africa records
- **155 Namibia** records
- **103 records** within the Etosha bounding box (lat −19.25 to −18.20, lon 15.60 to 16.55)
- Year range: 2023–2026

To refresh the cache: `python -c "from data.fetch_real_data import fetch_gbif_occurrences; fetch_gbif_occurrences(use_cache=False)"`

**Data citation:** GBIF.org (2026). *Diceros bicornis* occurrences. [https://doi.org/10.15468/dl.XXXXXXX](https://doi.org/10.15468/dl.XXXXXXX) *(replace with actual download DOI from GBIF)*

---

## Repository Structure

```
anti-poaching-patrol-framework/
├── main.py                       # Unified pipeline entrypoint
├── run_enhancements.py           # Three analytical enhancements runner
├── scenario_runner.py            # Seasonal and tech degradation scenarios
├── global_adaption.py            # Cross-region adaptation (Congo, Himalayas)
├── requirements.txt
├── data/
│   ├── raw_pdf_specs.json        # All model parameters and park specs
│   ├── fetch_real_data.py        # GBIF occurrence fetching + grid mapping
│   └── gbif_rhino_cache.json     # Cached GBIF records (auto-generated)
├── src/
│   ├── wpp_maut_engine.py        # Layer 1: WPP field generation
│   ├── ssg_dro_optimizer.py      # Layer 2: SLSQP patrol allocation
│   ├── jacobian_stability.py     # Layer 3: 3D ODE stability analysis
│   ├── feedback_loop.py          # Self-consistent iterative loop (NEW)
│   ├── rhc_temporal_shift.py     # Receding-horizon temporal weighting
│   └── advanced_analytics.py    # Gini, Lorenz, Pareto, Monte Carlo
├── utils/
│   ├── topology_constructor.py   # Etosha grid + waterhole placement
│   └── spatial_friction.py      # Seasonal terrain friction maps
├── visualizations/
│   ├── enhanced_figures.py       # Convergence, Pareto, GBIF figures (NEW)
│   ├── bifurcation_plots.py      # Stability boundary figures
│   ├── paper_figures.py          # Phase portrait, 3D surface, Lorenz, Pareto
│   ├── spatial_heatmaps.py       # Patrol density heatmaps
│   └── style_guide.py            # Academic figure style (300 dpi)
└── outputs/                      # Generated figures and metrics (gitignored)
```

---

## Citation

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

[Creative Commons Attribution 4.0 International (CC-BY 4.0)](LICENSE)
