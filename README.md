# OrthoMerge Studio — README

Here's a comprehensive English README for your GitHub repository:

```markdown
# 🔬 OrthoMerge Studio v5

**Intelligent Model Merging for Diffusion Models**

Analyze, optimize, and merge diffusion models (Flux, Chroma, Z-Image, etc.) using
orthogonal delta decomposition. OrthoMerge Studio automatically finds the optimal
delta-base assignments, detects redundancy, and generates merge scripts — while giving
you full manual control through an intuitive mixer interface.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange)

---

## 🎯 What Does This Do?

When you have multiple fine-tuned variants of a base model (e.g., Chroma v48 with
variants for detail, speed, realism, etc.), you want to **combine the best qualities
of each** into a single model. Naive linear merging (`0.5 × A + 0.5 × B`) causes
**destructive interference** — unique features cancel each other out.

**OrthoMerge Studio solves this by:**

1. **Analyzing** how each model differs from the base (and from each other)
2. **Finding the optimal delta-base** for each model (not always the global base!)
3. **Detecting redundancy** (models that are too similar to both include)
4. **Generating merge scripts** that use orthogonal projection to preserve unique features
5. **Letting you override everything** through a visual mixer

### Linear Merge vs. OrthoMerge

```
Linear Merge:    result = 0.5 × model_A + 0.5 × model_B
                 → Shared features reinforced, unique features diluted

OrthoMerge:      delta_A = model_A - base
                 delta_B = model_B - base
                 delta_B_ortho = delta_B - projection(delta_B onto delta_A)
                 result = base + delta_A + delta_B_ortho
                 → Both unique contributions fully preserved
```

---

## ✨ Key Features

### Exhaustive Delta-Base Optimization
Instead of always computing deltas against the global base, the system tests
**every possible parent** for each model:

```
Traditional:     delta = ModelA - Base           (magnitude: 1200)
Optimized:       delta = ModelA - ModelC         (magnitude: 340)
                 → 72% more focused, less noise
```

This works because some fine-tunes share a common ancestor. Computing the delta
against a closer relative produces a **cleaner, more focused signal**.

### Full-Pairwise Streaming Analysis
All pairwise dot products are computed in a single streaming pass:

```
dot(δ_A←B, δ_C←D) = dot(A,C) - dot(A,D) - dot(B,C) + dot(B,D)
```

This means any delta-base combination can be evaluated **without reloading models**.

### Visual Delta Mixer
Every parameter is pre-filled from the analysis but fully editable:

| ✓ | Model | Delta From | Alpha | Repeat | Info |
|---|-------|-----------|-------|--------|------|
| ☑ | detail-calibrated | base | 1.0 | 2× | High orthogonality → 2× recommended |
| ☑ | Chroma1-HD | chromav47 | 1.0 | 1× | Via chromav47: 45% more focused |
| ☑ | flash-heun | base | 0.8 | 1× | Standard delta (mag=890) |
| ☐ | RL47 | base | 0.0 | 1× | Redundant to Chroma1-HD (cos=0.84) |

- **Add/remove rows** with a single click
- **Change delta-base** per model via dropdown
- **Adjust alpha** (0–2) and repeat count (1–3)
- **Re-enable excluded models** by checking the box

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (for preview generation, optional)
- [sd-mecha](https://github.com/ljleb/sd-mecha) (for merge execution)
- [OrthoMerge](https://github.com/e-c-k-e-r/OrthoMerge) module

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/orthomerge-studio.git
cd orthomerge-studio

# Install dependencies
pip install gradio safetensors torch

# Install sd-mecha (required for merge execution)
pip install sd-mecha

# Copy OrthoMerge module to your working directory
# (or ensure it's importable)
cp /path/to/OrthoMerge.py .
```

### File Structure

```
orthomerge-studio/
├── ortho_studio_v5.py       # GUI (Gradio web interface)
├── analyze_deltas_v5.py     # Backend (analysis + optimization engine)
├── OrthoMerge.py            # OrthoMerge algorithm (external dependency)
├── README.md
└── examples/
    └── chroma_merge.json    # Example mixer configuration
```

---

## 🚀 Quick Start

### Launch the GUI

```bash
python ortho_studio_v5.py

# With verbose logging:
python ortho_studio_v5.py --verbose
```

Opens at `http://localhost:7860`

### Step-by-Step Workflow

1. **Select architecture** (Chroma, Flux, Z-Image, or Custom)
2. **Choose base model** (e.g., `chromav48.safetensors`)
3. **Select models to analyze** (or leave empty for all)
4. **Click "🔬 Analyse + Optimierung starten"**
5. **Review the Mixer** (Tab 2) — adjust as needed
6. **Click "📝 Script generieren"**
7. **Click "🚀 Merge"** (Tab 3) to execute
8. **Preview the result** (Tab 4) via ComfyUI

### CLI Usage (without GUI)

```bash
# Analyze all models in a directory
python analyze_deltas_v5.py /path/to/base.safetensors /path/to/models/ \
    --arch chroma-flow --fix

# Analyze specific models
python analyze_deltas_v5.py base.safetensors modelA.safetensors modelB.safetensors \
    --arch chroma-flow --rank-samples 10
```

---

## 📊 Analysis Pipeline

### Phase 1: Validation & Auto-Fix

Detects and optionally repairs common issues:

| Issue | Auto-Fix |
|-------|----------|
| CLIP/text encoder keys embedded in model | Strip non-diffusion keys |
| VAE keys embedded in model | Strip VAE keys |
| `model.diffusion_model.` prefix | Remove prefix |
| FP8 quantized weights | Convert to BF16 |
| `.norm.weight` vs `.norm.scale` naming | Remap keys |

### Phase 2: Full-Pairwise Streaming

Streams through all model weights **once**, computing:

- **Dot products** between all pairs (Base + all models)
- **Conflict counts** (weights where deltas point in opposite directions)
- **Per-block norms** (energy distribution across transformer blocks)
- **Key magnitudes** (for top-K overlap analysis)
- **Outlier tracking** (maximum absolute delta values)

Memory-efficient: only one key is loaded at a time.

### Phase 3: Signal Quality (SVD-Based)

For the largest tensors, computes via Singular Value Decomposition:

- **Effective Rank**: fraction of singular values needed for 90% energy
  - Low (< 0.3) = focused, targeted fine-tune ★★★
  - High (> 0.6) = diffuse, noisy changes ★
- **SNR (Signal-to-Noise Ratio)**: top-10% singular values vs. rest
  - High SNR = clean training signal
  - Low SNR = noisy or overfitted

### Phase 4: Metrics & Optimization

**Three key matrices** (reduced from six in v4):

| Metric | What It Tells You |
|--------|-------------------|
| **Cosine Similarity** | Direction similarity of deltas. High = redundant, low = complementary |
| **New Information %** | `1 - cos²` — perpendicular component. Higher = more unique contribution |
| **Conflict Ratio** | Fraction of weights where deltas disagree on direction. High = destructive |

**Exhaustive delta-base optimization:**

```
For each model M_i:
    For each possible parent P ∈ {Base, M_1, ..., M_n} \ {M_i}:
        Compute ||M_i - P|| using stored dot products
        Score = orthogonality × focus × conflict_avoidance

Find assignment that maximizes total orthogonality
Subject to: no circular dependencies (A←B←A)
```

Algorithm: Greedy initialization → iterative local search → cycle elimination.

---

## 🧮 How the Math Works

### Delta Computation from Stored Dot Products

Given stored values `dot(X, Y)` and `||X||²` for all models X, Y:

```
dot(A-B, C-D) = dot(A,C) - dot(A,D) - dot(B,C) + dot(B,D)

||A-B|| = sqrt(||A||² + ||B||² - 2·dot(A,B))

cos(δ_A←B, δ_C←D) = dot(A-B, C-D) / (||A-B|| · ||C-D||)
```

This allows evaluating **any** delta-base assignment without touching the model files.

### Orthogonality Score

For a set of assignments `{(M_i, P_i)}`, the total orthogonality is:

```
score = mean over all pairs (i,j):  1 - cos²(M_i - P_i, M_j - P_j)
```

Higher = deltas are more independent = less information loss during merge.

### Quality Score

Per model, combining three signals:

```
Quality = 0.40 × Focus + 0.40 × SNR_norm + Survival_bonus - Outlier_penalty

where:
  Focus     = 1 - effective_rank  (lower rank = more focused)
  SNR_norm  = min(1, log10(SNR) / 2)  (SNR 100 → 1.0)
  Survival  = bonus if >30% survives averaging
  Outlier   = penalty for extreme weight values
```

---

## 🎛️ The Mixer in Detail

### Row Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Enabled** (✓) | on/off | Include this delta in the merge |
| **Model** | dropdown | The fine-tuned model |
| **Delta From** | dropdown | Parent for delta computation (`base` or any other model) |
| **Alpha** (α) | 0.0 – 2.0 | Scaling factor for this delta. 1.0 = full strength |
| **Repeat** (×) | 1 – 3 | How many times to include this delta (amplification) |

### Global Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| **theta_agg** | mean, median, sum | How OrthoMerge aggregates projection angles |
| **conflict_aware** | true/false | Enable conflict detection during merge |
| **direction_weight** | theta, magnitude, equal | How to weight delta directions |
| **Texture Boost** | model + alpha | Post-merge add_difference for texture preservation |

### Auto-Fill Logic

After analysis, the mixer is pre-filled with:

- **Optimal delta-base** per model (from exhaustive search)
- **Repeat = 2** for models with >85% new information vs. all others
- **Enabled = false** for redundant models (cos > 0.8 to a higher-quality model)
- **Alpha = 1.0** for all enabled models (conservative default)
- **Info field** explains the reasoning for each suggestion

---

## 📁 Supported Architectures

| Architecture | Config String | Model Directory |
|-------------|--------------|-----------------|
| Flux.2 Klein | `flux2-klein` | `ComfyUI/models/diffusion_models/Flux/` |
| Chroma | `chroma-flow` | `ComfyUI/models/diffusion_models/Chroma/` |
| Z-Image | `zimage-base` | `ComfyUI/models/diffusion_models/Z-Image/` |
| Custom | (user-defined) | (user-defined) |

---

## 💡 Example: Chroma Merge Recipe

Starting with this manual recipe:

```
Base (for OrthoMerge):  chromav48

Deltas from v47:              Deltas from v48:
  chroma2k    - v47             Chroma1-HD        - v48
  RL47        - v47             detail-calibrated  - v48  (2×!)
  flash-heun  - v47             UnCanny            - v48
```

The analyzer might discover that:

- `Chroma1-HD` is closer to `chromav47` than to `chromav48` → **delta via v47 is 45% cleaner**
- `detail-calibrated` has 92% new information vs. all others → **2× repeat confirmed**
- `RL47` and `flash-heun` have cos=0.83 → **one is redundant**, keep the higher-quality one
- `UnCanny` has the highest texture score → **use as post-merge texture boost (α=0.2)**

Resulting optimized mixer:

| ✓ | Model | Delta From | α | × | Reason |
|---|-------|-----------|---|---|--------|
| ☑ | detail-calibrated | base (v48) | 1.0 | 2 | High orthogonality |
| ☑ | Chroma1-HD | chromav47 | 1.0 | 1 | 45% more focused via v47 |
| ☑ | chroma2k | chromav47 | 1.0 | 1 | Standard delta |
| ☑ | flash-heun | chromav47 | 1.0 | 1 | Quality: A (0.72) |
| ☐ | RL47 | chromav47 | 0.0 | 1 | Redundant to flash-heun |
| ☑ | UnCanny | base (v48) | 1.0 | 1 | Texture boost candidate |

With texture boost: `UnCanny` at α=0.2 post-merge.

---

## 🔧 Configuration

### Environment Variables

```bash
# Custom ComfyUI directory (default: ~/ComfyUI)
export COMFYUI_DIR="/path/to/ComfyUI"

# Custom API endpoint (default: http://127.0.0.1:8188)
export COMFYUI_API="http://localhost:8188"
```

### Command-Line Options

```bash
python ortho_studio_v5.py [OPTIONS]

Options:
  --verbose, -v    Enable detailed logging
  --port PORT      Web UI port (default: 7860)
```

```bash
python analyze_deltas_v5.py BASE MODELS [OPTIONS]

Options:
  --arch ARCH          sd_mecha architecture config
  --fix                Auto-fix incompatible models
  --rank-samples N     Number of keys for SVD analysis (default: 5)
```

---

## ⚙️ Generated Merge Script Structure

The generated Python scripts follow this structure:

```python
import sd_mecha, torch
from OrthoMerge import orthomergev2
from safetensors.torch import load_file, save_file

# Load base and models
base = sd_mecha.model("base.safetensors", "chroma-flow")
model_a = sd_mecha.model("modelA.safetensors", "chroma-flow")
model_b = sd_mecha.model("modelB.safetensors", "chroma-flow")

# Compute deltas (with optimized delta-bases)
delta_a = sd_mecha.subtract(model_a, base)
delta_b = sd_mecha.subtract(model_b, other_model)  # Lineage-optimized!

# OrthoMerge
merged = orthomergev2(base, delta_a, delta_b, delta_b,  # delta_b 2×
    alpha=1.0, theta_agg="mean", conflict_aware=False,
    direction_weight="theta")

# Optional: Texture boost
tex_delta = sd_mecha.subtract(texture_model, base)
final = sd_mecha.add_difference(merged, tex_delta, alpha=0.2)

# Execute merge
sd_mecha.merge(final, output="result.safetensors",
    merge_device="cpu", merge_dtype=torch.float32,
    output_dtype=torch.bfloat16)

# Post-merge fix: restore missing keys + repair NaN
base_sd = load_file("base.safetensors")
merged_sd = load_file("result.safetensors")
for k, v in base_sd.items():
    if k not in merged_sd:
        merged_sd[k] = v  # Restore missing keys
for k, v in merged_sd.items():
    if not torch.isfinite(v.float()).all():
        merged_sd[k] = base_sd.get(k, torch.zeros_like(v))  # Fix NaN
save_file(merged_sd, "result.safetensors")
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **New architectures** (SD3, AuraFlow, etc.)
- **Better clustering heuristics** for Realism vs. Utility separation
- **Perceptual quality metrics** (FID/CLIP-score integration)
- **Block-level alpha control** (different alpha per transformer block)
- **Memory optimization** for very large models (>20GB)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [sd-mecha](https://github.com/ljleb/sd-mecha) — Model merging framework
- [OrthoMerge](https://github.com/e-c-k-e-r/OrthoMerge) — Orthogonal merge algorithm
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Image generation backend
- [Gradio](https://gradio.app/) — Web UI framework

---

*Built for the model merging community. If this tool helps you create better models,
consider sharing your merge recipes!*
```
