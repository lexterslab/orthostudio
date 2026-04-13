# OrthoMerge Studio

Expert-level model merge preparation and execution for diffusion models. Analyzes, recommends, and executes OrthoMergeV2 recipes with multi-metric analysis, lineage detection, and automatic compatibility fixing.

Built for Chroma, FLUX.2 Klein, Z-Image, and any safetensors-based diffusion architecture.

## What it does

OrthoMerge Studio takes the guesswork out of model merging. Instead of blindly combining checkpoints and hoping for the best, it runs a 6-phase analysis pipeline that tells you exactly which models to merge, in what order, against which base, and with what parameters.

**Phase 1 — Validation & Auto-Fix**
Detects and repairs incompatible models automatically: strips CLIP/VAE keys from full checkpoints, removes `model.diffusion_model.` prefixes, remaps `.weight`↔`.scale` norm keys, converts FP8→BF16.

**Phase 2 — Streaming Metrics**
Computes Cosine Similarity, L2 Distance, Conflict Ratio, Spearman Rank Correlation, and Top-K Overlap — all key-by-key streaming, using ~5GB RAM regardless of model count or size.

**Phase 3 — Signal Quality**
Measures Effective Rank (how focused is each finetune?) and detects weight outliers that could destabilize SVD operations during merge.

**Phase 4 — Derived Analysis**
Calculates parallel/perpendicular decomposition (how much new information does each delta bring?), detects model lineage (which model was finetuned from which?), and predicts cumulative interference (how much of each delta survives the merge?).

**Phase 5 — Dual Strategy Recommendation**
Outputs two complete merge strategies:
- **Strategy A (Lineage)** — Multi-base deltas computed against each model's optimal parent, `mean` aggregation, no conflict detection. Mirrors the approach used in hand-crafted expert recipes.
- **Strategy B (Automatic)** — Single-base deltas, data-driven strategy selection (direct/all-at-once/pairwise), `median` for large merges, automatic conflict handling.

**Phase 6 — Script Generation**
Produces ready-to-run Python scripts with post-merge validation (missing key restoration, NaN/Inf repair).

## Features

- **Streaming analysis** — never loads a full model into RAM. Handles 10+ models on 16GB+ systems.
- **Lineage detection** — automatically finds the cleanest base for each delta by measuring inter-model distances.
- **Redundancy detection** — identifies models that are too similar (using 3 independent metrics) and recommends exclusion or reinforcement.
- **Duplication recommendation** — suggests doubling highly orthogonal deltas instead of excluding them.
- **Per-block conflict localization** — pinpoints which transformer blocks have the most sign conflicts.
- **Survival prediction** — estimates what percentage of each delta's contribution remains after merging.
- **Auto-fix pipeline** — handles CLIP/VAE stripping, prefix normalization, key remapping, and FP8 conversion transparently.
- **Post-merge validation** — generated scripts automatically restore missing keys from base and replace NaN/Inf values.

## Installation

```bash
cd ~/ComfyUI
git clone https://github.com/YOUR_USERNAME/ortho-merge-studio.git
cd ortho-merge-studio

# Copy files to ComfyUI root
cp analyze_deltas.py ortho_studio.py OrthoMerge.py ~/ComfyUI/

# Install dependencies (in your ComfyUI venv)
source ~/ComfyUI/venv/bin/activate
pip install gradio safetensors torch
```

### Requirements

- Python 3.10+
- PyTorch
- safetensors
- gradio (for GUI only)
- sd-mecha + a model config for your architecture (for merge execution)

## Usage

### GUI

```bash
cd ~/ComfyUI
python ortho_studio.py           # opens http://localhost:7860
python ortho_studio.py --verbose # with terminal logging
```

Three tabs:
- **📊 Analyse** — Select architecture, base model, run analysis, pick strategy A or B
- **🔀 Merge** — Edit and execute the generated script
- **🖼️ Preview** — Generate test images via ComfyUI API

### CLI

```bash
# Scan entire folder
python analyze_deltas.py models/base.safetensors models/folder/ --arch flux2-klein

# Specific models
python analyze_deltas.py models/base.safetensors models/a.safetensors models/b.safetensors --arch chroma-flow

# With auto-fix for incompatible models
python analyze_deltas.py models/base.safetensors models/folder/ --arch flux2-klein --fix
```

### Creating sd-mecha configs for new architectures

If your architecture isn't supported by sd-mecha yet, dump the key structure and create a YAML config:

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for key in sorted(f.keys()):
        t = f.get_tensor(key)
        print(f"{key}: {{shape: {list(t.shape)}, dtype: {str(t.dtype).replace('torch.', '')}}}")
```

Save the output as a YAML file with `identifier:` and `components: diffuser:` headers, then place it in:
```
venv/lib/python3.*/site-packages/sd_mecha/extensions/builtin/model_configs/
```

## Supported Architectures

| Architecture | Config Name | Status |
|---|---|---|
| Chroma | `chroma-flow` | Built-in (sd-mecha) |
| FLUX.1 | `flux-flux` | Built-in (sd-mecha) |
| FLUX.2 Klein 9B | `flux2-klein` | Custom config needed |
| Z-Image | `zimage-base` | Custom config needed |
| SDXL | `sdxl-sgm` | Built-in (sd-mecha) |
| SD 1.5 | `sd1-ldm` | Built-in (sd-mecha) |

## How OrthoMergeV2 works

OrthoMerge decomposes each model delta into a **rotation** (how the weight matrix's orientation changed) and a **residual** (everything else). Rotations are averaged in Cayley space — mathematically correct for matrices on the orthogonal manifold — while residuals are averaged linearly. This preserves structural properties that linear merging (weighted sum, SLERP) destroys.

Key parameters:
- `alpha` — merge strength (1.0 = full merge)
- `theta_agg` — how to aggregate rotation magnitudes (`mean` for curated sets, `median` for large/noisy sets)
- `conflict_aware` — whether to detect and handle sign conflicts between deltas (disable with `mean` for curated sets)
- `direction_weight` — how to weight rotation directions (`theta` = magnitude-weighted)

## File Structure

```
analyze_deltas.py   — CLI analysis tool (standalone, no GUI dependencies)
ortho_studio.py     — Gradio GUI (imports from analyze_deltas.py)
OrthoMerge.py       — OrthoMergeV2 implementation (sd-mecha merge method)
```

## Credits

- OrthoMergeV2 algorithm implementation based on Orthogonal-Residual Decoupling with Cayley parameterization
- [sd-mecha](https://github.com/ljleb/sd-mecha) by ljleb — memory-efficient state dict recipe merger
- [Chroma](https://huggingface.co/lodestones/Chroma) by lodestone-rock — the model family that inspired this tooling

## License

MIT
