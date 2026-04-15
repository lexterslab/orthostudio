#!/usr/bin/env python3
"""
OrthoMerge Analyzer v5 — Optimized with Exhaustive Delta-Base Search

Analysiert Modelle in 4 Phasen:
  1. Validierung & Auto-Fix
  2. Streaming-Metriken (alle Paare, inkl. Model↔Model)
  3. Signalqualität (Effective Rank, SNR)
  4. Exhaustive Optimierung (beste Delta-Base pro Modell, Alpha, Repeat)

Kern-Innovation: Streaming berechnet ALLE paarweisen Deltas (nicht nur vs. Base),
sodass die optimale Delta-Base-Zuweisung ohne nochmaliges Laden gefunden wird.
"""

import sys, os, glob, math, time, itertools, json, copy
from collections import defaultdict
from safetensors import safe_open

# ═════════════════════════════════════════════════════════════════
# PHASE 1: VALIDIERUNG
# ═════════════════════════════════════════════════════════════════

CLIP_PREFIXES = [
    "text_encoders.", "te.", "te1.", "te2.", "cond_stage_model.",
    "clip_l.", "clip_g.", "text_model.", "clip.",
]
VAE_PREFIXES = [
    "first_stage_model.", "vae.", "vae_model.",
]
KNOWN_DIFFUSER_PREFIXES = [
    "double_blocks.", "single_blocks.", "img_in.", "txt_in.",
    "final_layer.", "time_in.", "double_stream_", "single_stream_",
    "distilled_guidance_", "context_refiner.", "cap_embedder.",
    "joint_blocks.", "x_embedder.", "t_embedder.", "y_embedder.",
]


def classify_keys(keys):
    result = {"diffuser": [], "clip": [], "vae": [], "unknown": []}
    for k in keys:
        k_check = k
        if k.startswith("model.diffusion_model."):
            k_check = k[len("model.diffusion_model."):]
        if any(k_check.startswith(p) for p in CLIP_PREFIXES):
            result["clip"].append(k)
        elif any(k_check.startswith(p) for p in VAE_PREFIXES):
            result["vae"].append(k)
        elif any(k_check.startswith(p) for p in KNOWN_DIFFUSER_PREFIXES):
            result["diffuser"].append(k)
        else:
            result["unknown"].append(k)
    return result


def validate_model(path, base_keys):
    name = os.path.basename(path)
    issues, fixes = [], []
    with safe_open(path, framework="pt", device="cpu") as f:
        model_keys = list(f.keys())
        sample_dtype = str(f.get_tensor(model_keys[0]).dtype)

    if "float8" in sample_dtype.lower():
        issues.append(f"FP8 ({sample_dtype})")
        fixes.append("convert_bf16")

    classified = classify_keys(model_keys)
    if classified["clip"]:
        issues.append(f"CLIP-Keys ({len(classified['clip'])})")
        fixes.append("strip_clip")
    if classified["vae"]:
        issues.append(f"VAE-Keys ({len(classified['vae'])})")
        fixes.append("strip_vae")

    has_prefix = all(k.startswith("model.diffusion_model.") for k in model_keys)
    if has_prefix:
        issues.append("Prefix 'model.diffusion_model.'")
        fixes.append("strip_prefix")

    effective_keys = set(model_keys)
    if has_prefix:
        effective_keys = {k.removeprefix("model.diffusion_model.") for k in model_keys}
    base_set = set(base_keys)
    common = effective_keys & base_set

    if len(common) < len(base_set) * 0.5:
        remapped = set()
        rc = 0
        for k in effective_keys:
            if ".norm." in k and k.endswith(".weight"):
                alt = k.rsplit(".weight", 1)[0] + ".scale"
                remapped.add(alt)
                rc += 1
            else:
                remapped.add(k)
        if len(remapped & base_set) > len(common):
            issues.append(f"Norm .weight→.scale ({rc})")
            fixes.append("remap_norm")
            common = remapped & base_set

    if not issues:
        issues.append("✅ OK")

    return {"name": name, "path": path, "issues": issues, "fixes": fixes,
            "common": len(common), "total": len(base_keys)}


def fix_model(path, fixes, base_keys):
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(path)
    base_set = set(base_keys)

    if "strip_clip" in fixes or "strip_vae" in fixes:
        cl = classify_keys(sd.keys())
        remove = set()
        if "strip_clip" in fixes:
            remove.update(cl["clip"])
        if "strip_vae" in fixes:
            remove.update(cl["vae"])
        sd = {k: v for k, v in sd.items() if k not in remove}
    if "strip_prefix" in fixes:
        sd = {k.removeprefix("model.diffusion_model."): v for k, v in sd.items()}
    if "remap_norm" in fixes:
        new_sd = {}
        for k, v in sd.items():
            if ".norm." in k and k.endswith(".weight"):
                alt = k.rsplit(".weight", 1)[0] + ".scale"
                if alt in base_set and k not in base_set:
                    new_sd[alt] = v
                    continue
            new_sd[k] = v
        sd = new_sd
    if "convert_bf16" in fixes:
        for k in sd:
            if "float8" in str(sd[k].dtype).lower():
                sd[k] = sd[k].to(torch.bfloat16)
    save_file(sd, path)
    return path


# ═════════════════════════════════════════════════════════════════
# PHASE 2: FULL-PAIRWISE STREAMING
# ═════════════════════════════════════════════════════════════════

def build_key_mapping(model_path, base_keys):
    base_set = set(base_keys)
    with safe_open(model_path, framework="pt", device="cpu") as f:
        mkeys = list(f.keys())
    fwd = {}
    for k in mkeys:
        ek = k
        if k.startswith("model.diffusion_model."):
            ek = k[len("model.diffusion_model."):]
        if ".norm." in ek and ek.endswith(".weight"):
            alt = ek.rsplit(".weight", 1)[0] + ".scale"
            if alt in base_set and ek not in base_set:
                ek = alt
        if ek in base_set:
            fwd[k] = ek
    return {v: k for k, v in fwd.items()}


def get_block(key):
    parts = key.split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0]
    return "_misc"


def streaming_core(base_path, model_paths, log_fn=print):
    """Full-pairwise Streaming: Base als Index 0, Modelle als 1..n.

    Berechnet ALLE paarweisen Metriken zwischen Base UND allen Modellen.
    Das ermöglicht: optimale Delta-Base-Zuweisung OHNE nochmaliges Laden.

    Index-Schema:
        0 = Base
        1..n = Modelle

    Returns dict with (n+1)×(n+1) Matrizen für dot_prod, norm_sq, etc.
    """
    n = len(model_paths)
    total_n = n + 1  # Base + n Modelle
    names = ["__base__"] + [
        os.path.basename(p).replace(".safetensors", "") for p in model_paths
    ]

    base_f = safe_open(base_path, framework="pt", device="cpu")
    base_keys = sorted(base_f.keys())
    log_fn(f"Base: {len(base_keys)} Keys")

    model_files = []
    rev_maps = []
    for mi, p in enumerate(model_paths):
        mf = safe_open(p, framework="pt", device="cpu")
        model_files.append(mf)
        rev = build_key_mapping(p, base_keys)
        rev_maps.append(rev)
        log_fn(f"  {names[mi+1]}: {len(rev)}/{len(base_keys)} Keys mapped")

    # Akkumulatoren für ALLE Paare (inkl. Base als Index 0)
    # dot_prod[i][j] = sum(tensor_i * tensor_j) über alle Keys
    # norm_sq[i] = sum(tensor_i^2)
    # Für Base (idx 0): tensor = base_tensor selbst
    # Für Modelle (idx 1..n): tensor = model_tensor

    # Wir brauchen: dot(model_i - parent_j, model_k - parent_l) für beliebige Parents
    # Das lässt sich aus den Rohdaten berechnen:
    # dot(A-B, C-D) = dot(A,C) - dot(A,D) - dot(B,C) + dot(B,D)

    # Also speichern wir dot_prod und norm_sq für ALLE Tensoren (Base + Modelle)
    dot_prod = [[0.0] * total_n for _ in range(total_n)]
    norm_sq = [0.0] * total_n
    conflict_cnt = [[0] * total_n for _ in range(total_n)]
    active_cnt = [[0] * total_n for _ in range(total_n)]
    block_norms_all = defaultdict(lambda: [0.0] * total_n)

    # Key-Magnitudes (für Redundanz-Erkennung)
    key_mags = {i: {} for i in range(total_n)}

    # Outlier-Tracking
    outlier_max = [0.0] * total_n

    # Für Effective Rank
    key_sizes = {}
    for k in base_keys:
        t = base_f.get_tensor(k)
        key_sizes[k] = t.numel()
        del t

    total = len(base_keys)
    t0 = time.time()
    last_t = 0

    for ki, bk in enumerate(base_keys):
        now = time.time()
        if ki % 10 == 0 or now - last_t > 5:
            pct = ki / total * 100
            eta = ((now - t0) / max(ki, 1)) * (total - ki)
            log_fn(f"  [{pct:5.1f}%] {ki}/{total} — {bk[:45]}  ETA:{eta:.0f}s")
            last_t = now

        bt = base_f.get_tensor(bk).float().flatten()
        block = get_block(bk)

        # Index 0 = Base
        tensors = [bt]  # Index 0

        for mi in range(n):
            mk = rev_maps[mi].get(bk)
            if mk is None:
                tensors.append(None)
                continue
            mt = model_files[mi].get_tensor(mk)
            if mt.shape != bt.reshape(-1).shape and mt.numel() != bt.numel():
                tensors.append(None)
                continue
            tensors.append(mt.float().flatten())

        # Berechne alle paarweisen Dot-Products
        for i in range(total_n):
            if tensors[i] is None:
                continue
            ti = tensors[i]
            ti_sq = (ti * ti).sum().item()
            norm_sq[i] += ti_sq
            block_norms_all[block][i] += ti_sq

            # Key magnitude (für Deltas vs Base, also model - base)
            if i > 0:  # Modelle
                delta = ti - tensors[0]  # Delta vs Base
                mag = delta.norm().item()
                key_mags[i][bk] = mag
                abs_max = delta.abs().max().item()
                if abs_max > outlier_max[i]:
                    outlier_max[i] = abs_max
                del delta

            for j in range(i + 1, total_n):
                if tensors[j] is None:
                    continue
                tj = tensors[j]
                dot = (ti * tj).sum().item()
                dot_prod[i][j] += dot
                dot_prod[j][i] += dot

                # Conflict: Richtungskonflikte der DELTAS vs Base
                # (nur sinnvoll zwischen Modellen, nicht Base vs Modell)
                if i > 0 and j > 0:
                    di = ti - tensors[0]
                    dj = tj - tensors[0]
                    both = (di != 0) & (dj != 0)
                    na = both.sum().item()
                    active_cnt[i][j] += na
                    active_cnt[j][i] += na
                    if na > 0:
                        nc = ((di[both] * dj[both]) < 0).sum().item()
                        conflict_cnt[i][j] += nc
                        conflict_cnt[j][i] += nc
                    del di, dj

        del tensors, bt

    elapsed = time.time() - t0
    log_fn(f"  [100%] Streaming fertig in {elapsed:.1f}s")

    del base_f
    for mf in model_files:
        del mf

    return {
        "n_models": n, "total_n": total_n,
        "names": names,  # [0] = __base__, [1..n] = Modelle
        "base_keys": base_keys, "key_sizes": key_sizes,
        "dot_prod": dot_prod, "norm_sq": norm_sq,
        "conflict_cnt": conflict_cnt, "active_cnt": active_cnt,
        "block_norms": dict(block_norms_all),
        "key_mags": key_mags, "outlier_max": outlier_max,
        "model_paths": model_paths, "base_path": base_path,
        "elapsed": elapsed,
    }


# ═════════════════════════════════════════════════════════════════
# PHASE 3: SIGNALQUALITÄT
# ═════════════════════════════════════════════════════════════════

def compute_effective_rank(base_path, model_paths, rev_maps, base_keys,
                           key_sizes, n_samples=5, log_fn=print):
    """SVD-basiert: Effective Rank + SNR pro Modell (Delta vs Base)."""
    import torch
    candidates = [(sz, k) for k, sz in key_sizes.items() if sz > 1024]
    candidates.sort(reverse=True)
    sample_keys = [k for _, k in candidates[:n_samples]]

    n = len(model_paths)
    if not sample_keys:
        return [0.5] * n, [1.0] * n

    log_fn(f"  Effective Rank + SNR ({len(sample_keys)} Keys)...")
    eff_ranks = [[] for _ in range(n)]
    snr_vals = [[] for _ in range(n)]

    base_f = safe_open(base_path, framework="pt", device="cpu")
    model_files = [safe_open(p, framework="pt", device="cpu") for p in model_paths]

    for bk in sample_keys:
        bt = base_f.get_tensor(bk).float()
        if bt.ndim < 2:
            continue
        for mi in range(n):
            mk = rev_maps[mi].get(bk)
            if mk is None:
                continue
            mt = model_files[mi].get_tensor(mk).float()
            if mt.shape != bt.shape:
                continue
            delta = (mt - bt).reshape(bt.shape[0], -1)
            try:
                S = torch.linalg.svdvals(delta)
                S_sq = S * S
                total_energy = S_sq.sum().item()
                if total_energy < 1e-10:
                    eff_ranks[mi].append(0)
                    snr_vals[mi].append(0)
                    continue
                cumulative = 0
                for rank, sv in enumerate(S_sq, 1):
                    cumulative += sv.item()
                    if cumulative >= 0.9 * total_energy:
                        eff_ranks[mi].append(rank / len(S))
                        break
                n_sv = len(S)
                top_k = max(1, n_sv // 10)
                signal_energy = S_sq[:top_k].sum().item()
                noise_energy = S_sq[top_k:].sum().item()
                snr = signal_energy / max(noise_energy, 1e-10)
                snr_vals[mi].append(snr)
            except Exception:
                pass
            del delta, mt
        del bt

    del base_f
    for mf in model_files:
        del mf

    avg_ranks = [sum(r) / len(r) if r else 0.5 for r in eff_ranks]
    avg_snr = [sum(s) / len(s) if s else 1.0 for s in snr_vals]
    return avg_ranks, avg_snr


# ═════════════════════════════════════════════════════════════════
# PHASE 4: ABGELEITETE METRIKEN + EXHAUSTIVE OPTIMIERUNG
# ═════════════════════════════════════════════════════════════════

def _delta_dot(raw, a, b, c, d):
    """Berechne dot(tensor_a - tensor_b, tensor_c - tensor_d) aus gespeicherten Dot-Products.

    dot(A-B, C-D) = dot(A,C) - dot(A,D) - dot(B,C) + dot(B,D)

    wobei dot(X,X) = norm_sq[X] und dot(X,Y) = dot_prod[X][Y].
    """
    def _dot(i, j):
        if i == j:
            return raw["norm_sq"][i]
        return raw["dot_prod"][i][j]

    return _dot(a, c) - _dot(a, d) - _dot(b, c) + _dot(b, d)


def _delta_norm(raw, a, b):
    """||tensor_a - tensor_b|| = sqrt(dot(A-B, A-B))."""
    val = _delta_dot(raw, a, b, a, b)
    return math.sqrt(max(val, 0))


def _delta_cosine(raw, a, b, c, d):
    """Cosine similarity zwischen Delta(a-b) und Delta(c-d)."""
    dot = _delta_dot(raw, a, b, c, d)
    n1 = _delta_norm(raw, a, b)
    n2 = _delta_norm(raw, c, d)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return dot / (n1 * n2)


def compute_derived(raw):
    """Berechnet Metriken für Modelle (Index 1..n) relativ zu Base (Index 0)."""
    n = raw["n_models"]
    model_names = raw["names"][1:]  # Ohne __base__

    # Magnitudes (Delta vs Base)
    mags = [_delta_norm(raw, i + 1, 0) for i in range(n)]

    # Cosine zwischen Deltas (alle vs Base)
    cos_m = [[0.0] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        c = _delta_cosine(raw, i + 1, 0, j + 1, 0)
        cos_m[i][j] = cos_m[j][i] = round(c, 4)

    # Conflict Ratio (aus dem Streaming, zwischen Model-Deltas vs Base)
    conf_m = [[0.0] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        ac = raw["active_cnt"][i + 1][j + 1]
        conf_m[i][j] = conf_m[j][i] = round(
            raw["conflict_cnt"][i + 1][j + 1] / ac if ac > 0 else 0, 4)

    # Neue Info
    new_info = [[0.0] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        perp = 1.0 - cos_m[i][j] ** 2
        new_info[i][j] = new_info[j][i] = round(perp, 4)

    # L2 zwischen Modell-Deltas
    l2_m = [[0.0] * n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        # ||delta_i - delta_j|| = ||(model_i - base) - (model_j - base)||
        # = ||model_i - model_j||
        d = _delta_norm(raw, i + 1, j + 1)
        l2_m[i][j] = l2_m[j][i] = round(d, 2)

    # Top-K Overlap (für Redundanz)
    common_keys = set(raw["base_keys"])
    for mi in range(n):
        common_keys &= set(raw["key_mags"][mi + 1].keys())
    common_list = sorted(common_keys)

    topk_m = [[0.0] * n for _ in range(n)]
    if common_list:
        k_count = max(1, int(len(common_list) * 0.05))
        topk_sets = {}
        for mi in range(n):
            sk = sorted(common_list,
                       key=lambda k: raw["key_mags"][mi + 1].get(k, 0),
                       reverse=True)
            topk_sets[mi] = set(sk[:k_count])
        for i, j in itertools.combinations(range(n), 2):
            union = topk_sets[i] | topk_sets[j]
            jaccard = len(topk_sets[i] & topk_sets[j]) / len(union) if union else 0
            topk_m[i][j] = topk_m[j][i] = round(jaccard, 4)

    # Lineage: für jedes Modell die OPTIMALE Delta-Base
    # (Kern-Innovation: wir können das jetzt exakt berechnen!)
    lineage = {}
    for i in range(n):
        best_parent = "base"
        best_parent_idx = 0
        # Magnitude des Deltas vs Base
        best_mag = mags[i]

        # Teste jedes andere Modell als Alternative Delta-Base
        for j in range(n):
            if i == j:
                continue
            # Delta von Modell i relativ zu Modell j
            alt_mag = _delta_norm(raw, i + 1, j + 1)
            if alt_mag < best_mag:
                best_mag = alt_mag
                best_parent = model_names[j]
                best_parent_idx = j + 1

        improvement = (mags[i] - best_mag) / max(mags[i], 1e-8) * 100

        lineage[model_names[i]] = {
            "best_parent": best_parent,
            "best_parent_idx": best_parent_idx,
            "dist_to_parent": round(best_mag, 2),
            "dist_to_base": round(mags[i], 2),
            "improvement": round(improvement, 1),
        }

    # Survival (Interferenz-Vorhersage)
    survival = [0.0] * n
    if n >= 2:
        for i in range(n):
            if mags[i] < 1e-8:
                continue
            dot_with_avg = 1.0
            for j in range(n):
                if i != j:
                    dot_with_avg += cos_m[i][j]
            avg_mag_sq = n
            for ii, jj in itertools.combinations(range(n), 2):
                avg_mag_sq += 2 * cos_m[ii][jj]
            avg_mag = math.sqrt(max(avg_mag_sq, 1e-8))
            survival[i] = round(dot_with_avg / avg_mag * 100, 1)

    # Texture Score (Block-Profil)
    texture_scores = []
    for mi in range(n):
        # Delta-Norm pro Block
        total_delta_norm_sq = max(mags[mi] ** 2, 1e-10)
        tex_energy = 0.0
        for block, norms in raw["block_norms"].items():
            # Block-Norm für Delta = ||model_block - base_block||^2
            # ≈ norm_sq_model_block + norm_sq_base_block - 2*dot_block
            # Vereinfachung: block_norms[block][mi+1] ≈ Energie im Block
            block_frac = norms[mi + 1] / max(raw["norm_sq"][mi + 1], 1e-10)
            if block == "single_blocks":
                tex_energy += block_frac * 1.0
            elif block == "double_blocks":
                tex_energy += block_frac * 0.4
            else:
                tex_energy += block_frac * 0.3
        texture_scores.append(round(tex_energy, 3))

    return {
        "names": model_names, "mags": [round(m, 2) for m in mags], "n": n,
        "cos": cos_m, "l2": l2_m, "conf": conf_m, "new_info": new_info,
        "topk": topk_m, "lineage": lineage, "survival": survival,
        "texture_scores": texture_scores,
        "base_path": raw["base_path"], "model_paths": raw["model_paths"],
        "outlier_max": [raw["outlier_max"][i + 1] for i in range(n)],
    }


# ═════════════════════════════════════════════════════════════════
# QUALITY SCORING
# ═════════════════════════════════════════════════════════════════

def compute_quality_scores(derived, eff_ranks, snr_scores):
    """Quality: Focus (40%) + SNR (40%) + Survival-Bonus (20%)."""
    n = derived["n"]
    quality, details = [], []
    for mi in range(n):
        focus = 1.0 - min(eff_ranks[mi], 1.0)
        raw_snr = snr_scores[mi]
        snr_norm = min(1.0, math.log10(max(raw_snr, 1.0)) / 2.0)
        surv = derived["survival"][mi]
        surv_bonus = min(0.2, max(0, (surv - 30) / 200))
        outlier_penalty = min(0.15, max(0, (derived["outlier_max"][mi] - 5.0) / 50))
        score = 0.40 * focus + 0.40 * snr_norm + surv_bonus - outlier_penalty
        score = round(max(0.0, min(1.0, score)), 3)
        quality.append(score)
        details.append({
            "focus": round(focus, 3),
            "snr_raw": round(raw_snr, 1),
            "snr_norm": round(snr_norm, 3),
            "survival": round(surv, 1),
            "total": score,
        })
    return quality, details


# ═════════════════════════════════════════════════════════════════
# EXHAUSTIVE DELTA-BASE OPTIMIERUNG
# ═════════════════════════════════════════════════════════════════

def _orthogonality_score(raw, assignments):
    """Berechne den Gesamt-Orthogonalitäts-Score einer Delta-Base-Zuweisung.

    assignments: list of (model_idx_1based, parent_idx) tuples
        parent_idx: 0 = base, 1..n = Modell

    Score = Summe der paarweisen "neuen Information" (1 - cos²).
    Höher = orthogonaler = besser für OrthoMerge.
    """
    n = len(assignments)
    if n < 2:
        return 0.0

    total_ortho = 0.0
    total_conflict = 0.0
    count = 0

    for idx_a in range(n):
        model_a, parent_a = assignments[idx_a]
        mag_a = _delta_norm(raw, model_a, parent_a)
        if mag_a < 1e-8:
            continue

        for idx_b in range(idx_a + 1, n):
            model_b, parent_b = assignments[idx_b]
            mag_b = _delta_norm(raw, model_b, parent_b)
            if mag_b < 1e-8:
                continue

            cos = _delta_cosine(raw, model_a, parent_a, model_b, parent_b)
            ortho = 1.0 - cos * cos  # Neue Information
            total_ortho += ortho
            count += 1

    return total_ortho / max(count, 1)


def _delta_quality_score(raw, model_idx, parent_idx, base_mag):
    """Score für ein einzelnes Delta: kleiner = fokussierter."""
    mag = _delta_norm(raw, model_idx, parent_idx)
    # Normalisiert: wie viel kleiner ist das Delta als vs. Base?
    if base_mag < 1e-8:
        return 0.0
    return 1.0 - min(mag / base_mag, 1.0)


def exhaustive_optimize(raw, eff_ranks, snr_scores, quality_scores,
                        max_models=8, log_fn=print):
    """Findet die optimale Delta-Base-Zuweisung für alle Modelle.

    Strategie:
    1. Für jedes Modell: teste alle möglichen Parents (Base + andere Modelle)
    2. Bewerte jede Gesamt-Konfiguration nach:
       - Orthogonalität der resultierenden Deltas (60%)
       - Fokus/Sauberkeit der einzelnen Deltas (25%)
       - Conflict-Minimierung (15%)
    3. Berücksichtige: zirkuläre Abhängigkeiten verboten (A←B←A)

    Für n Modelle mit je (n+1) Eltern-Optionen = (n+1)^n Kombinationen.
    Bei n≤8 ist das ≤9^8 = 43M → zu viel für brute force.

    Lösung: Greedy + lokale Suche.
    """
    n = raw["n_models"]
    model_names = raw["names"][1:]
    log_fn(f"\n  Exhaustive Optimierung ({n} Modelle)...")

    # Basis-Magnitudes (jedes Modell vs. Base)
    base_mags = [_delta_norm(raw, i + 1, 0) for i in range(n)]

    # ── Schritt 1: Greedy-Start ──
    # Für jedes Modell: finde lokal besten Parent
    def valid_assignment(assignments):
        """Prüfe auf zirkuläre Abhängigkeiten."""
        # assignments: [(model_1based, parent)] list
        parent_map = {}
        for model_idx, parent_idx in assignments:
            parent_map[model_idx] = parent_idx

        for model_idx, parent_idx in assignments:
            if parent_idx == 0:
                continue  # Base als Parent: immer OK
            # Folge der Kette: parent → parent.parent → ...
            visited = {model_idx}
            current = parent_idx
            while current != 0:
                if current in visited:
                    return False  # Zirkel!
                visited.add(current)
                current = parent_map.get(current, 0)
        return True

    # Für jedes Modell: berechne Score für JEDEN möglichen Parent
    parent_scores = {}  # (model_idx_1based, parent_idx) → score
    for mi in range(n):
        model_idx = mi + 1
        for pi in range(n + 1):  # 0=Base, 1..n=Modelle
            if pi == model_idx:
                continue  # Kann nicht sein eigener Parent sein
            mag = _delta_norm(raw, model_idx, pi)
            focus = 1.0 - min(mag / max(base_mags[mi], 1e-8), 1.5)
            parent_scores[(model_idx, pi)] = {
                "magnitude": round(mag, 2),
                "focus_score": round(max(0, focus), 3),
                "parent_name": raw["names"][pi] if pi > 0 else "base",
            }

    # Greedy: starte mit jedem Modell → bester lokaler Parent
    best_assignments = []
    for mi in range(n):
        model_idx = mi + 1
        best_parent = 0
        best_mag = base_mags[mi]
        for pi in range(n + 1):
            if pi == model_idx:
                continue
            mag = parent_scores[(model_idx, pi)]["magnitude"]
            if mag < best_mag:
                best_mag = mag
                best_parent = pi
        best_assignments.append((model_idx, best_parent))

    # Prüfe Zirkel und fixe sie
    if not valid_assignment(best_assignments):
        log_fn("  ⚠ Greedy hat Zirkel → fixe auf Base")
        # Einfacher Fix: wenn Zirkel, setze den schwächeren auf Base
        for attempt in range(n):
            if valid_assignment(best_assignments):
                break
            # Finde den Zirkel-Teilnehmer mit geringstem Improvement
            worst_idx = -1
            worst_improvement = float('inf')
            for bi, (midx, pidx) in enumerate(best_assignments):
                if pidx != 0:
                    impr = base_mags[midx - 1] - parent_scores[(midx, pidx)]["magnitude"]
                    if impr < worst_improvement:
                        worst_improvement = impr
                        worst_idx = bi
            if worst_idx >= 0:
                midx = best_assignments[worst_idx][0]
                best_assignments[worst_idx] = (midx, 0)

    best_score = _orthogonality_score(raw, best_assignments)
    log_fn(f"  Greedy-Score: {best_score:.4f}")

    # ── Schritt 2: Lokale Suche ──
    improved = True
    iterations = 0
    while improved and iterations < 50:
        improved = False
        iterations += 1
        for mi in range(n):
            model_idx = mi + 1
            current_parent = best_assignments[mi][1]

            for pi in range(n + 1):
                if pi == model_idx or pi == current_parent:
                    continue

                # Teste Alternative
                trial = list(best_assignments)
                trial[mi] = (model_idx, pi)

                if not valid_assignment(trial):
                    continue

                trial_score = _orthogonality_score(raw, trial)
                if trial_score > best_score + 0.001:
                    best_score = trial_score
                    best_assignments = trial
                    improved = True

    log_fn(f"  Optimiert nach {iterations} Iterationen: Score={best_score:.4f}")

    # ── Schritt 3: Redundanz-Erkennung + Repeat-Empfehlung ──
    # Berechne paarweise Cosine der optimierten Deltas
    optimized_cosines = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mi, pi = best_assignments[i]
            mj, pj = best_assignments[j]
            c = _delta_cosine(raw, mi, pi, mj, pj)
            optimized_cosines[i][j] = optimized_cosines[j][i] = round(c, 4)

    # Redundanz: hohe Cosine + hoher TopK-Overlap der Basis-Deltas
    redundant = set()
    redundant_reason = {}
    for i, j in itertools.combinations(range(n), 2):
        if abs(optimized_cosines[i][j]) > 0.8:
            # Behalte das mit höherem Quality-Score
            loser = i if quality_scores[i] < quality_scores[j] else j
            winner = j if loser == i else i
            if loser not in redundant:
                redundant.add(loser)
                redundant_reason[loser] = (
                    f"Redundant zu {model_names[winner]} "
                    f"(cos={optimized_cosines[i][j]:.3f})")

    # Repeat: Modelle mit hoher Orthogonalität zu ALLEN anderen → 2x
    repeat_candidates = set()
    for i in range(n):
        if i in redundant:
            continue
        active = [j for j in range(n) if j != i and j not in redundant]
        if not active:
            continue
        avg_new_info = sum(1.0 - optimized_cosines[i][j] ** 2
                          for j in active) / len(active)
        if avg_new_info > 0.85:
            repeat_candidates.add(i)

    # ── Ergebnis zusammenbauen ──
    results = []
    for mi in range(n):
        model_idx, parent_idx = best_assignments[mi]
        parent_name = raw["names"][parent_idx] if parent_idx > 0 else "base"
        mag_vs_base = base_mags[mi]
        mag_vs_parent = _delta_norm(raw, model_idx, parent_idx)
        improvement = (mag_vs_base - mag_vs_parent) / max(mag_vs_base, 1e-8) * 100

        is_redundant = mi in redundant
        repeat = 2 if mi in repeat_candidates else 1
        alpha = 1.0 if not is_redundant else 0.0

        # Reason
        if is_redundant:
            reason = redundant_reason.get(mi, "Redundant")
        elif parent_idx != 0 and improvement > 5:
            reason = (f"Delta via {parent_name}: {improvement:.0f}% "
                     f"fokussierter (mag {mag_vs_base:.0f}→{mag_vs_parent:.0f})")
        elif mi in repeat_candidates:
            reason = f"Hohe Orthogonalität zu allen → ×2 empfohlen"
        else:
            reason = f"Standard-Delta vs Base (mag={mag_vs_base:.0f})"

        # Alternative Parents für das UI (sortiert nach Qualität)
        alternatives = []
        for pi in range(n + 1):
            if pi == model_idx:
                continue
            pn = raw["names"][pi] if pi > 0 else "base"
            pm = _delta_norm(raw, model_idx, pi)
            alternatives.append({
                "parent": pn,
                "parent_idx": pi,
                "magnitude": round(pm, 2),
                "is_optimal": pi == parent_idx,
            })
        alternatives.sort(key=lambda x: x["magnitude"])

        results.append({
            "model": model_names[mi],
            "model_idx": model_idx,
            "delta_base": parent_name,
            "delta_base_idx": parent_idx,
            "alpha": alpha,
            "repeat": repeat,
            "enabled": not is_redundant,
            "reason": reason,
            "quality_grade": ("A+" if quality_scores[mi] > 0.8 else
                            "A" if quality_scores[mi] > 0.65 else
                            "B" if quality_scores[mi] > 0.5 else
                            "C" if quality_scores[mi] > 0.35 else "D"),
            "quality_score": quality_scores[mi],
            "texture_score": 0.0,  # Wird von derived befüllt
            "magnitude_vs_base": round(mag_vs_base, 2),
            "magnitude_vs_parent": round(mag_vs_parent, 2),
            "improvement": round(improvement, 1),
            "survival": 0.0,  # Wird befüllt
            "alternatives": alternatives,
        })

    # Sortiere: enabled zuerst, dann nach Magnitude
    results.sort(key=lambda r: (not r["enabled"], -r["magnitude_vs_base"]))

    return {
        "rows": results,
        "total_orthogonality": round(best_score, 4),
        "optimized_cosines": optimized_cosines,
        "redundant": list(redundant),
        "repeat_candidates": list(repeat_candidates),
        "parent_scores": {f"{k[0]}_{k[1]}": v for k, v in parent_scores.items()},
    }


# ═════════════════════════════════════════════════════════════════
# SCRIPT-GENERIERUNG AUS MIXER
# ═════════════════════════════════════════════════════════════════

def _vn(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()[:30]


def _dedup_varnames(names_list):
    vns = [_vn(s) for s in names_list]
    seen = {}
    for i, v in enumerate(vns):
        if v in seen:
            seen[v] += 1
            vns[i] = f"{v}_{seen[v]}"
        else:
            seen[v] = 0
    return vns


def generate_script_from_mixer(base_path, model_dir, arch, mixer_rows,
                               theta_agg="mean", conflict_aware=False,
                               direction_weight="theta",
                               texture_boost_model=None,
                               texture_boost_alpha=0.0):
    """Generiert ein Merge-Script direkt aus den Mixer-Zeilen."""
    bf = os.path.basename(base_path)
    a_str = f'"{arch}"' if arch else '"FIXME"'

    active = [r for r in mixer_rows
              if r.get("enabled", True) and r.get("alpha", 0) > 0]
    if not active:
        return "# Fehler: Keine aktiven Modelle im Mixer"

    # Sammle alle benötigten Modelle und Parents
    all_models = set()
    all_parents = {"base"}
    for row in active:
        all_models.add(row["model"])
        if row.get("delta_base", "base") != "base":
            all_parents.add(row["delta_base"])
            all_models.add(row["delta_base"])
    if texture_boost_model and texture_boost_alpha > 0:
        all_models.add(texture_boost_model)

    model_list = sorted(all_models)
    vns = _dedup_varnames(model_list)
    name_to_var = dict(zip(model_list, vns))

    L = [
        "import sd_mecha", "import torch", "import os",
        "from OrthoMerge import orthomergev2",
        "from safetensors.torch import load_file, save_file", "",
        "sd_mecha.set_log_level()", "",
        f'MODEL_DIR = "{model_dir}"',
        f'ARCH = {a_str}',
        f'BASE_FILE = "{bf}"',
        f'OUTPUT_NAME = "OrthoMerge_result.safetensors"', "",
        "# ── Base ──",
        f'base = sd_mecha.model(f"{{MODEL_DIR}}/{{BASE_FILE}}", ARCH)', "",
        "# ── Modelle ──",
    ]
    for mname in model_list:
        var = name_to_var[mname]
        L.append(
            f'{var} = sd_mecha.model(f"{{MODEL_DIR}}/{mname}.safetensors", ARCH)')

    L += ["", "subtract = sd_mecha.subtract", "",
          "# ── Deltas (Mixer-Konfiguration) ──"]

    delta_args = []
    for row in active:
        var = name_to_var[row["model"]]
        db = row.get("delta_base", "base")
        parent_var = "base" if db == "base" else name_to_var.get(db, "base")
        comment = f"  # via {db}" if db != "base" else ""
        L.append(f"delta_{var} = subtract({var}, {parent_var}){comment}")
        repeat = row.get("repeat", 1)
        for _ in range(repeat):
            delta_args.append(f"delta_{var}")
        if repeat > 1:
            L.append(f"# ↑ {row['model']} ×{repeat}")

    L.append("")

    # Alpha-Handling
    alphas = []
    for r in active:
        for _ in range(r.get("repeat", 1)):
            alphas.append(r.get("alpha", 1.0))
    all_same = len(set(alphas)) <= 1
    main_alpha = alphas[0] if alphas else 1.0

    if all_same:
        args = ", ".join(delta_args)
        L += [
            "# ── OrthoMerge ──",
            f"merged = orthomergev2(base, {args},",
            f'    alpha={main_alpha}, theta_agg="{theta_agg}", '
            f'conflict_aware={conflict_aware}, '
            f'direction_weight="{direction_weight}")',
        ]
    else:
        L.append("# ── Alpha-gewichtete Deltas ──")
        scaled_args = []
        arg_idx = 0
        for row in active:
            var = name_to_var[row["model"]]
            alpha = row.get("alpha", 1.0)
            repeat = row.get("repeat", 1)
            if abs(alpha - 1.0) > 1e-6:
                sname = f"delta_{var}_a{str(alpha).replace('.','')}"
                L.append(f"{sname} = sd_mecha.multiply(delta_{var}, {alpha})")
                for _ in range(repeat):
                    scaled_args.append(sname)
            else:
                for _ in range(repeat):
                    scaled_args.append(f"delta_{var}")
            arg_idx += repeat

        args = ", ".join(scaled_args)
        L += [
            "", "# ── OrthoMerge ──",
            f"merged = orthomergev2(base, {args},",
            f'    alpha=1.0, theta_agg="{theta_agg}", '
            f'conflict_aware={conflict_aware}, '
            f'direction_weight="{direction_weight}")',
        ]

    # Texture Boost
    if texture_boost_model and texture_boost_alpha > 0:
        boost_var = name_to_var.get(texture_boost_model)
        if boost_var:
            boost_parent = "base"
            for row in mixer_rows:
                if row.get("model") == texture_boost_model:
                    boost_parent = row.get("delta_base", "base")
                    break
            bp_var = ("base" if boost_parent == "base"
                     else name_to_var.get(boost_parent, "base"))
            L += [
                "",
                f"# ── Texture-Boost ({texture_boost_model}) ──",
                "add_difference = sd_mecha.add_difference",
                f"tex_delta = subtract({boost_var}, {bp_var})",
                f"new = add_difference(merged, tex_delta, "
                f"alpha={texture_boost_alpha})",
            ]
        else:
            L += ["", "new = merged"]
    else:
        L += ["", "new = merged"]

    # Post-Merge Fix
    L += [
        "", "# ── Merge ausführen ──",
        'output_path = f"{MODEL_DIR}/{OUTPUT_NAME}"',
        "sd_mecha.merge(new,",
        "    output=output_path,",
        '    merge_device="cpu", merge_dtype=torch.float32,',
        "    output_dtype=torch.bfloat16, threads=1)",
        "",
        "# ── Post-Merge Fix ──",
        'print("Post-Merge: Prüfe fehlende Keys + NaN...")',
        'base_sd = load_file(f"{MODEL_DIR}/{BASE_FILE}")',
        "merged_sd = load_file(output_path)",
        "added = 0",
        "for k, v in base_sd.items():",
        "    if k not in merged_sd:",
        "        merged_sd[k] = v",
        "        added += 1",
        "bad_keys = []",
        "for k, v in merged_sd.items():",
        "    if not torch.isfinite(v.float()).all():",
        "        bad_keys.append(k)",
        "        merged_sd[k] = base_sd.get(k, torch.zeros_like(v))",
        'if added: print(f"  {added} fehlende Keys ergänzt")',
        'if bad_keys: print(f"  {len(bad_keys)} NaN/Inf Keys repariert")',
        "save_file(merged_sd, output_path)",
        'print(f"Fertig: {output_path} ({len(merged_sd)} Keys)")',
    ]
    return "\n".join(L)
