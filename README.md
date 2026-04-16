```markdown
# 🔬 OrthoMerge Studio v5

**Intelligent Model Merging for Diffusion Models**





## What Does This Do?

When you have multiple fine-tuned variants of a base model (e.g., Chroma v48 with variants for detail, speed, realism, etc.), you want to **combine the best qualities of each** into a single model. Naive linear merging causes **destructive interference** — unique features cancel each other out.

**OrthoMerge Studio solves this by:**

1. **Analyzing** how each model differs from the base (and from each other)
2. **Finding the optimal delta-base** for each model (not always the global base!)
3. **Detecting redundancy** (models that are too similar to both include)
4. **Generating merge scripts** that use orthogonal projection to preserve unique features
5. **Letting you override everything** through a visual mixer

### Linear Merge vs. OrthoMerge

**Linear Merge (naive):**

```text
result = 0.5 × model_A + 0.5 × model_B
→ Shared features reinforced, unique features diluted by 50%
```

**OrthoMerge (orthogonal projection):**

```text
delta_A       = model_A - base
delta_B       = model_B - base
delta_B_ortho = delta_B - projection(delta_B onto delta_A)
result        = base + delta_A + delta_B_ortho
→ Both unique contributions fully preserved
```

The key insight: by projecting each delta onto the orthogonal complement of the others, **no information is lost**. Each model's unique contribution survives the merge intact.

---

## Key Features

### 1. Exhaustive Delta-Base Optimization

Instead of always computing deltas against the global base, the system tests **every possible parent** for each model:

```text
Traditional:     delta = ModelA - Base           (magnitude: 1200)
Optimized:       delta = ModelA - ModelC         (magnitude:  340)
                 → 72% more focused, less noise
```

This works because some fine-tunes share a common ancestor. Computing the delta against a closer relative produces a **cleaner, more focused signal**.

The optimizer uses greedy initialization followed by iterative local search, with cycle detection to prevent circular dependencies (A←B←A).

### 2. Full-Pairwise Streaming Analysis

All pairwise dot products between **all** models (including Base) are computed in a single streaming pass through the weights:

```text
dot(δ_A←B, δ_C←D) = dot(A,C) - dot(A,D) - dot(B,C) + dot(B,D)
```

Any delta-base combination can be evaluated **without reloading model files**. For 6 models this gives complete information about all 7⁶ = 117,649 possible delta-base assignments from a single analysis pass.

### 3. Visual Delta Mixer

Every parameter is pre-filled from the analysis but **fully editable**:

| ✓ | Model | Delta From | Alpha | Repeat | Info |
|---|-------|-----------|-------|--------|------|
| ☑ | detail-calibrated | base | 1.0 | 2× | High orthogonality → 2× recommended |
| ☑ | Chroma1-HD | chromav47 | 1.0 | 1× | Via chromav47: 45% more focused |
| ☑ | flash-heun | base | 0.8 | 1× | Standard delta (mag=890) |
| ☐ | RL47 | base | 0.0 | 1× | Redundant to Chroma1-HD (cos=0.84) |

**What you can do:**

- ➕ **Add rows** and ➖ **Remove rows** with a single click
- **Change delta-base** per model via dropdown (any model or base)
- **Adjust alpha** per delta (0.0–2.0)
- **Set repeat count** (1–3× for amplification)
- **Re-enable excluded models** by checking the enable box
- **Override any automatic decision** — the analysis is a suggestion, not a mandate

### 4. Streamlined Analysis

Reduced from 6 matrices to **3 essential metrics**:

| Metric | What It Tells You | Why It Matters |
|--------|-------------------|----------------|
| **Cosine Similarity** | Direction similarity of deltas | High → redundant, low → complementary |
| **New Information %** | Perpendicular component | Higher → more unique contribution |
| **Conflict Ratio** | Weights where deltas disagree | High → destructive interference risk |

Removed in v5 (compared to v4): Spearman Rank (redundant to Cosine), Block Coherence (too abstract), Sparsity Score (questionable assumptions).

### 5. ComfyUI Integration

Preview any model (including your freshly merged result) directly through the ComfyUI API — no need to switch applications. The preview tab builds and submits ComfyUI workflows programmatically.

---

## Installation

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | 2.0+ | Tensor operations |
| safetensors | latest | Model file I/O |
| Gradio | 4.0+ | Web UI |
| sd-mecha | latest | Merge execution |
| OrthoMerge | latest | Orthogonal merge algorithm |
| ComfyUI | latest | Preview generation (optional) |

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/orthomerge-studio.git
cd orthomerge-studio

# Install Python dependencies
pip install gradio safetensors torch

# Install sd-mecha (required for merge execution)
pip install sd-mecha

# Copy OrthoMerge module to your working directory
# (or ensure it's importable from your Python path)
cp /path/to/OrthoMerge.py .
```

### Verify Installation

```bash
python -c "
import gradio, safetensors, torch, sd_mecha
from OrthoMerge import orthomergev2
print('All dependencies OK')
"
```

### File Structure

```text
orthomerge-studio/
├── ortho_studio_v5.py         # GUI (Gradio web interface)
├── analyze_deltas_v5.py       # Backend (analysis + optimization engine)
├── OrthoMerge.py              # OrthoMerge algorithm (external dependency)
├── README.md                  # This file
├── LICENSE                    # MIT License
└── examples/
    └── chroma_merge.json      # Example mixer configuration
```

---

## Quick Start

### Launch the GUI

```bash
python ortho_studio_v5.py
```

Opens automatically at [http://localhost:7860](http://localhost:7860).

For detailed logging:

```bash
python ortho_studio_v5.py --verbose
```

### Step-by-Step Workflow

| Step | Action | Where |
|------|--------|-------|
| 1 | Select architecture (Chroma, Flux, Z-Image, Custom) | Top bar |
| 2 | Choose base model (e.g., `chromav48.safetensors`) | Top bar |
| 3 | Select models to analyze (or leave empty for all) | Top bar |
| 4 | Click **🔬 Analyse + Optimierung starten** | Tab 1: Analyse |
| 5 | Review the Mixer — adjust delta-bases, alphas, repeats | Tab 2: Mixer |
| 6 | Click **📝 Script generieren** | Tab 2: Mixer |
| 7 | Click **🚀 Merge** to execute | Tab 3: Merge |
| 8 | Generate test images via ComfyUI | Tab 4: Preview |

### CLI Usage (Without GUI)

```bash
# Analyze all models in a directory
python analyze_deltas_v5.py /path/to/base.safetensors /path/to/models/ \
    --arch chroma-flow --fix

# Analyze specific models
python analyze_deltas_v5.py base.safetensors modelA.safetensors modelB.safetensors \
    --arch chroma-flow --rank-samples 10
```

**CLI Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--arch ARCH` | sd-mecha architecture config string | *(required)* |
| `--fix` | Auto-fix incompatible models | off |
| `--rank-samples N` | Number of keys for SVD analysis | 5 |

---

## Analysis Pipeline

### Phase 1: Validation & Auto-Fix

Detects and optionally repairs common compatibility issues:

| Issue | Detection | Auto-Fix |
|-------|-----------|----------|
| CLIP/text encoder keys in model | Key prefix matching | Strip non-diffusion keys |
| VAE keys in model | Key prefix matching | Strip VAE keys |
| `model.diffusion_model.` prefix | Prefix check | Remove prefix |
| FP8 quantized weights | Dtype inspection | Convert to BF16 |
| `.norm.weight` vs `.norm.scale` | Key name mismatch | Remap keys |
| Less than 50% key overlap | Set intersection | Attempt remap, skip if failed |

### Phase 2: Full-Pairwise Streaming

Streams through all model weights **once**, loading only one key at a time.

**Computed per key:**

- **Dot products** between all pairs (Base + all models)
- **Delta conflict counts** (weights where model deltas point in opposite directions)
- **Per-block norms** (energy distribution across transformer blocks)
- **Key magnitudes** (for redundancy analysis)
- **Outlier tracking** (maximum absolute delta values)

**Memory efficiency:** For N models with K keys, memory usage is O(N) per key, not O(N × K). Total time complexity is O(K × N²).

### Phase 3: Signal Quality (SVD-Based)

For the largest tensors (configurable, default top 5 by parameter count), performs Singular Value Decomposition on each delta.

**Effective Rank:**

```text
For delta matrix D, compute singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ
Find smallest r such that Σᵢ₌₁ʳ σᵢ² ≥ 0.9 × Σᵢ₌₁ⁿ σᵢ²
Effective rank = r / n
```

| Effective Rank | Interpretation | Rating |
|---------------|----------------|--------|
| < 0.3 | Focused, targeted fine-tune | ★★★ |
| 0.3 – 0.6 | Moderate changes | ★★ |
| > 0.6 | Diffuse, noisy changes | ★ |

**Signal-to-Noise Ratio (SNR):**

```text
Signal = energy in top 10% of singular values
Noise  = energy in remaining 90%
SNR    = Signal / Noise
```

High SNR indicates a clean training signal. Low SNR suggests noisy or overfitted weights.

### Phase 4: Metrics & Exhaustive Optimization

**Three key matrices computed:**

1. **Cosine Similarity** — direction overlap between deltas
2. **New Information %** — perpendicular (unique) component: `1 - cos²`
3. **Conflict Ratio** — fraction of weights where deltas disagree

**Exhaustive optimization algorithm:**

```text
1. GREEDY START
   For each model Mᵢ:
     Test all parents P ∈ {Base, M₁, ..., Mₙ} \ {Mᵢ}
     Pick P that minimizes ‖Mᵢ - P‖

2. LOCAL SEARCH (up to 50 iterations)
   For each model Mᵢ:
     For each alternative parent P:
       If swapping improves total orthogonality score → accept swap

3. CYCLE ELIMINATION
   If A←B←A detected:
     Break cycle by resetting the weaker link to Base

4. REDUNDANCY DETECTION
   If cos(δᵢ, δⱼ) > 0.8 after optimization:
     Disable the lower-quality model

5. REPEAT RECOMMENDATION
   If avg new_info(δᵢ, δⱼ) > 0.85 for all j ≠ i:
     Recommend 2× repeat for model i
```

---

## How the Math Works

### Delta Computation from Stored Dot Products

The core insight: if we store `dot(Xᵢ, Xⱼ)` for all model pairs, we can compute the dot product of **any** delta pair without touching the model files:

```text
Given:  dot(X, Y) stored for all X, Y ∈ {Base, M₁, ..., Mₙ}
        ‖X‖² = dot(X, X) also stored

Then:   dot(A-B, C-D) = dot(A,C) - dot(A,D) - dot(B,C) + dot(B,D)
        ‖A-B‖         = sqrt(‖A‖² + ‖B‖² - 2·dot(A,B))
        cos(A-B, C-D)  = dot(A-B, C-D) / (‖A-B‖ · ‖C-D‖)
```

This allows evaluating all (N+1)ᴺ possible delta-base assignments from a single O(K × N²) streaming pass.

### Orthogonality Score

For a set of delta assignments {δᵢ = Mᵢ - Pᵢ}:

```text
Total Orthogonality = mean over all pairs (i,j):  1 - cos²(δᵢ, δⱼ)

Range: 0 (all parallel) to 1 (all perpendicular)
Higher = deltas are more independent = less information loss during merge
```

### Quality Score per Model

```text
Quality = 0.40 × Focus
        + 0.40 × SNR_normalized
        + Survival_bonus
        - Outlier_penalty

where:
  Focus          = 1 - effective_rank         (range 0–1, higher = more focused)
  SNR_normalized = min(1, log₁₀(SNR) / 2)    (SNR of 100 → score 1.0)
  Survival_bonus = min(0.2, max(0, (survival% - 30) / 200))
  Outlier_penalty = min(0.15, max(0, (max_outlier - 5) / 50))
```

**Quality Grades:**

| Grade | Score Range | Interpretation |
|-------|------------|----------------|
| A+ | > 0.80 | Excellent — focused, clean, high signal |
| A | 0.65 – 0.80 | Very good — reliable merge candidate |
| B | 0.50 – 0.65 | Good — usable with some noise |
| C | 0.35 – 0.50 | Fair — consider with caution |
| D | < 0.35 | Poor — likely noisy or overfitted |

### Survival Prediction

Estimates how much of each delta survives in an averaged merge:

```text
For unit vectors uᵢ = δᵢ / ‖δᵢ‖:
  avg_direction = Σ uᵢ
  survival(i)   = dot(uᵢ, avg_direction / ‖avg_direction‖) × 100%
```

Models aligned with the average direction survive well. Orthogonal models lose signal in naive averaging — which is exactly why OrthoMerge exists.

---

## The Mixer in Detail

### Row Parameters

| Parameter | UI Element | Range | Description |
|-----------|-----------|-------|-------------|
| **Enabled** | Checkbox | on / off | Include this delta in the merge |
| **Model** | Dropdown | all models | The fine-tuned model to use |
| **Delta From** | Dropdown | `base` + all models | Parent for delta computation |
| **Alpha (α)** | Slider | 0.0 – 2.0 | Scaling factor for this delta. 1.0 = full strength |
| **Repeat (×)** | Slider | 1 – 3 | How many times to include this delta (amplification) |
| **Info** | Text (read-only) | — | Explanation of the automatic suggestion |

### Global Merge Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| **theta_agg** | `mean`, `median`, `sum` | `mean` | Aggregation method for projection angles |
| **conflict_aware** | `true`, `false` | `false` | Detect and handle weight conflicts during merge |
| **direction_weight** | `theta`, `magnitude`, `equal` | `theta` | How to weight delta directions in orthogonal projection |
| **Texture Boost Model** | any model or `(none)` | `(none)` | Post-merge add_difference source for texture preservation |
| **Texture Boost Alpha** | 0.0 – 0.5 | 0.0 | Strength of the texture boost |

### Auto-Fill Logic

After analysis, the mixer is automatically populated based on the optimization results:

| Condition | Action | Reasoning |
|-----------|--------|-----------|
| Model is closer to another model than to base | Set **Delta From** to that model | Produces a more focused, cleaner delta |
| New info > 85% vs. all others | Set **Repeat = 2** | Highly orthogonal — amplify its unique contribution |
| Cosine > 0.8 to a higher-quality model | Set **Enabled = false**, α = 0.0 | Redundant — would dilute without adding value |
| Null delta (magnitude ≈ 0) | Set **Enabled = false** | No meaningful changes to merge |
| All other cases | **Enabled = true**, α = 1.0, × 1 | Conservative defaults |

### Manual Override Examples

**Re-enable an excluded model:**

The analyzer excluded `RL47` as redundant to `flash-heun`. But you know RL47 has specific lighting improvements you want.
→ Check the enable box, set alpha to 0.5 for a lighter contribution.

**Change delta-base:**

The analyzer suggests `Chroma1-HD ← chromav47`. But you know Chroma1-HD was actually fine-tuned from `chromav46`.
→ Change the "Delta From" dropdown to `chromav46` (if available in the model list).

**Amplify a specific model:**

You want `detail-calibrated` to have extra influence because photorealistic skin detail is your priority.
→ Set repeat to 3× or increase alpha to 1.5.

**Add a model not in the original selection:**

You realize you forgot to include a model in the initial analysis.
→ Click ➕ to add a row, select the model from the dropdown, choose its delta-base manually.

---

## Supported Architectures

| Architecture | Config String | Default Model Directory |
|-------------|--------------|------------------------|
| Flux.2 Klein | `flux2-klein` | `~/ComfyUI/models/diffusion_models/Flux/` |
| Chroma | `chroma-flow` | `~/ComfyUI/models/diffusion_models/Chroma/` |
| Z-Image | `zimage-base` | `~/ComfyUI/models/diffusion_models/Z-Image/` |
| Custom | *(user-defined)* | *(user-defined)* |

To add a new architecture, edit the `DEFAULT_DIRS` and `ARCH_CONFIGS` dictionaries in `ortho_studio_v5.py`, or simply select "Custom" in the UI and enter the path and config string manually.

---

## Generated Merge Script Structure

The generated Python scripts follow this pattern:

```python
import sd_mecha, torch
from OrthoMerge import orthomergev2
from safetensors.torch import load_file, save_file

# === Configuration ===
MODEL_DIR = "/path/to/models"
ARCH = "chroma-flow"
BASE_FILE = "chromav48.safetensors"
OUTPUT_NAME = "OrthoMerge_result.safetensors"

# === Load base and models ===
base = sd_mecha.model(f"{MODEL_DIR}/{BASE_FILE}", ARCH)
model_a = sd_mecha.model(f"{MODEL_DIR}/modelA.safetensors", ARCH)
model_b = sd_mecha.model(f"{MODEL_DIR}/modelB.safetensors", ARCH)
parent_c = sd_mecha.model(f"{MODEL_DIR}/chromav47.safetensors", ARCH)

# === Compute deltas (with optimized delta-bases) ===
subtract = sd_mecha.subtract
delta_a = subtract(model_a, base)             # Standard: vs base
delta_b = subtract(model_b, parent_c)         # Lineage-optimized: vs v47

# === OrthoMerge ===
merged = orthomergev2(base, delta_a, delta_a, delta_b,  # delta_a repeated 2×
    alpha=1.0, theta_agg="mean", conflict_aware=False,
    direction_weight="theta")

# === Optional: Texture boost ===
tex_delta = subtract(texture_model, base)
new = sd_mecha.add_difference(merged, tex_delta, alpha=0.2)

# === Execute merge ===
sd_mecha.merge(new, output=f"{MODEL_DIR}/{OUTPUT_NAME}",
    merge_device="cpu", merge_dtype=torch.float32,
    output_dtype=torch.bfloat16, threads=1)

# === Post-merge fix ===
print("Post-Merge: Checking missing keys + NaN...")
base_sd = load_file(f"{MODEL_DIR}/{BASE_FILE}")
merged_sd = load_file(f"{MODEL_DIR}/{OUTPUT_NAME}")

# Restore any keys missing from the merge
added = 0
for k, v in base_sd.items():
    if k not in merged_sd:
        merged_sd[k] = v
        added += 1

# Repair any NaN or Inf values
bad_keys = []
for k, v in merged_sd.items():
    if not torch.isfinite(v.float()).all():
        bad_keys.append(k)
        merged_sd[k] = base_sd.get(k, torch.zeros_like(v))

if added:
    print(f"  {added} missing keys restored")
if bad_keys:
    print(f"  {len(bad_keys)} NaN/Inf keys repaired")

save_file(merged_sd, f"{MODEL_DIR}/{OUTPUT_NAME}")
print(f"Done: {OUTPUT_NAME} ({len(merged_sd)} keys)")
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_DIR` | `~/ComfyUI` | Path to ComfyUI installation |
| `COMFYUI_API` | `http://127.0.0.1:8188` | ComfyUI API endpoint |

```bash
# Example: custom ComfyUI location
export COMFYUI_DIR="/opt/ComfyUI"
export COMFYUI_API="http://192.168.1.100:8188"
```

### GUI Options

```bash
python ortho_studio_v5.py [OPTIONS]

Options:
  --verbose, -v    Enable detailed console logging
```

The GUI launches on `0.0.0.0:7860` by default (accessible from other machines on your network).

### CLI Options

```bash
python analyze_deltas_v5.py BASE MODELS [OPTIONS]

Positional:
  BASE               Path to base model (.safetensors)
  MODELS             Path(s) to model files, or a directory

Options:
  --arch ARCH        sd-mecha architecture config (required)
  --fix              Auto-fix incompatible models in-place
  --rank-samples N   Number of keys for SVD analysis (default: 5)
```

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: sd_mecha` | sd-mecha not installed | `pip install sd-mecha` |
| `ModuleNotFoundError: OrthoMerge` | OrthoMerge.py not found | Copy to working directory or add to PYTHONPATH |
| `❌ Base nicht gefunden` | Wrong directory or filename | Check path in top bar, click 🔄 to refresh |
| `❌ Zu wenige Modelle` | Less than 2 compatible models | Ensure models match the base architecture |
| Out of memory during analysis | Too many large models | Reduce `rank-samples` or analyze fewer models |
| Merge timeout (> 4h) | Very large models or slow disk | Use SSD, reduce number of deltas |
| ComfyUI preview not working | ComfyUI not running | Start ComfyUI first, check API endpoint |
| NaN in merged model | Numerical instability | Post-merge fix handles this automatically |

### Performance Tips

- **SSD storage** dramatically speeds up streaming analysis (many random reads)
- **rank-samples = 3** is usually sufficient; increase to 10 only for detailed analysis
- **Fewer models** = faster analysis (O(N²) pairwise comparisons)
- The streaming pass is the bottleneck — subsequent optimization is near-instant

---

## Contributing

Contributions are welcome! Areas of particular interest:

- **New architectures** — SD3, AuraFlow, Stable Cascade, etc.
- **Better clustering heuristics** — improved Realism vs. Utility separation
- **Perceptual quality metrics** — FID/CLIP-score integration for result evaluation
- **Block-level alpha control** — different alpha per transformer block
- **Memory optimization** — support for very large models (> 20 GB)
- **Batch processing** — analyze and merge multiple recipes in sequence
- **Preset system** — save and load mixer configurations

### Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/orthomerge-studio.git
cd orthomerge-studio
pip install -e ".[dev]"  # If setup.py exists
# or simply:
pip install gradio safetensors torch
```

### Code Structure

| File | Lines | Responsibility |
|------|-------|---------------|
| `analyze_deltas_v5.py` | ~800 | All analysis, optimization, and script generation |
| `ortho_studio_v5.py` | ~500 | Gradio UI, ComfyUI integration, merge execution |

The backend (`analyze_deltas_v5.py`) is fully independent of the GUI and can be used as a library or CLI tool.

---

## Acknowledgments

- **[sd-mecha](https://github.com/ljleb/sd-mecha)** — Model merging framework that handles the actual weight manipulation
- **[OrthoMerge](https://github.com/e-c-k-e-r/OrthoMerge)** — Orthogonal merge algorithm preserving unique delta contributions
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — Node-based image generation backend used for previews
- **[Gradio](https://gradio.app/)** — Web UI framework powering the interactive interface
- **[safetensors](https://github.com/huggingface/safetensors)** — Fast and safe model file format

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2025 OrthoMerge Studio Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*Built for the model merging community. If this tool helps you create better models, consider sharing your merge recipes!*
```
