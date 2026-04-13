#!/usr/bin/env python3
"""
OrthoMerge Analyzer v4 — Expert-Level Merge Preparation

Analysiert Modelle in 6 Phasen:
  1. Validierung & Auto-Fix (CLIP/VAE/Prefix/FP8/Shape)
  2. Streaming-Metriken (Cosine, L2, Conflict, Spearman, Top-K, Per-Block)
  3. Signalqualität (Effective Rank, Outlier-Erkennung)
  4. Lineage-Erkennung (optimale Base pro Delta)
  5. Interferenz-Vorhersage (Parallel/Perpendicular, kumulative Verwässerung)
  6. Empfehlung (Multi-Base Rezept, Alpha-Balance, fertiges Script mit Post-Merge-Fix)

Verwendung:
    python analyze_deltas.py BASE ORDNER/ [--arch ARCH] [--fix] [--topk 0.05]
    python analyze_deltas.py BASE MODEL_A MODEL_B [...] [--arch ARCH] [--fix]

Optionen:
    --arch ARCH     sd_mecha Config
    --fix           Inkompatible Modelle automatisch reparieren
    --topk FLOAT    Top-K Schwelle (Default: 0.05 = 5%)
    --rank-samples  Anzahl Keys für Effective Rank (Default: 5)
"""

import sys, os, glob, math, time, itertools, json
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
# PHASE 2: STREAMING-METRIKEN
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


def streaming_core(base_path, model_paths, topk_pct=0.05, log_fn=print):
    n = len(model_paths)
    names = [os.path.basename(p).replace(".safetensors", "") for p in model_paths]

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
        log_fn(f"  {names[mi]}: {len(rev)}/{len(base_keys)} Keys mapped")

    # Akkumulatoren
    dot_prod = [[0.0]*n for _ in range(n)]
    norm_sq = [0.0]*n
    conflict_cnt = [[0]*n for _ in range(n)]
    active_cnt = [[0]*n for _ in range(n)]
    block_dots = defaultdict(lambda: [[0.0]*n for _ in range(n)])
    block_norms = defaultdict(lambda: [0.0]*n)
    block_conflicts = defaultdict(lambda: [[0]*n for _ in range(n)])
    block_active = defaultdict(lambda: [[0]*n for _ in range(n)])
    key_mags = {mi: {} for mi in range(n)}

    # Outlier-Tracking
    outlier_max = [0.0]*n
    outlier_mean_sum = [0.0]*n
    outlier_count = [0]*n

    # Für Effective Rank: Identifiziere größte Keys
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

        bt = base_f.get_tensor(bk)
        block = get_block(bk)

        deltas = []
        for mi in range(n):
            mk = rev_maps[mi].get(bk)
            if mk is None:
                deltas.append(None); continue
            mt = model_files[mi].get_tensor(mk)
            if mt.shape != bt.shape:
                deltas.append(None); continue
            d = (mt.float() - bt.float()).flatten()
            deltas.append(d)

            # Per-key magnitude
            mag = d.norm().item()
            key_mags[mi][bk] = mag

            # Outlier tracking
            abs_max = d.abs().max().item()
            abs_mean = d.abs().mean().item()
            if abs_max > outlier_max[mi]:
                outlier_max[mi] = abs_max
            outlier_mean_sum[mi] += abs_mean
            outlier_count[mi] += 1

        for i in range(n):
            if deltas[i] is None: continue
            di = deltas[i]
            di_sq = (di * di).sum().item()
            norm_sq[i] += di_sq
            block_norms[block][i] += di_sq

            for j in range(i+1, n):
                if deltas[j] is None: continue
                dj = deltas[j]
                dot = (di * dj).sum().item()

                dot_prod[i][j] += dot; dot_prod[j][i] += dot
                block_dots[block][i][j] += dot; block_dots[block][j][i] += dot

                both = (di != 0) & (dj != 0)
                na = both.sum().item()
                active_cnt[i][j] += na; active_cnt[j][i] += na
                block_active[block][i][j] += na; block_active[block][j][i] += na
                if na > 0:
                    nc = ((di[both] * dj[both]) < 0).sum().item()
                    conflict_cnt[i][j] += nc; conflict_cnt[j][i] += nc
                    block_conflicts[block][i][j] += nc; block_conflicts[block][j][i] += nc

        del deltas, bt

    elapsed = time.time() - t0
    log_fn(f"  [100%] Streaming fertig in {elapsed:.1f}s")

    # Cleanup file handles
    del base_f
    for mf in model_files:
        del mf

    return {
        "n": n, "names": names, "base_keys": base_keys, "key_sizes": key_sizes,
        "dot_prod": dot_prod, "norm_sq": norm_sq,
        "conflict_cnt": conflict_cnt, "active_cnt": active_cnt,
        "block_dots": dict(block_dots), "block_norms": dict(block_norms),
        "block_conflicts": dict(block_conflicts), "block_active": dict(block_active),
        "key_mags": key_mags,
        "outlier_max": outlier_max, "outlier_mean_sum": outlier_mean_sum,
        "outlier_count": outlier_count,
        "model_paths": model_paths, "base_path": base_path, "elapsed": elapsed,
    }


# ═════════════════════════════════════════════════════════════════
# PHASE 3: SIGNALQUALITÄT
# ═════════════════════════════════════════════════════════════════

def compute_effective_rank(base_path, model_paths, rev_maps, base_keys, key_sizes,
                           n_samples=5, log_fn=print):
    """Effective Rank der größten Tensoren pro Delta (SVD-basiert)."""
    import torch

    # Top-N größte 2D Keys
    candidates = [(sz, k) for k, sz in key_sizes.items() if sz > 1024]
    candidates.sort(reverse=True)
    sample_keys = [k for _, k in candidates[:n_samples]]

    if not sample_keys:
        return [0.0] * len(model_paths)

    log_fn(f"\n  Effective Rank ({len(sample_keys)} Keys)...")
    n = len(model_paths)
    eff_ranks = [[] for _ in range(n)]

    base_f = safe_open(base_path, framework="pt", device="cpu")
    model_files = [safe_open(p, framework="pt", device="cpu") for p in model_paths]

    for bk in sample_keys:
        bt = base_f.get_tensor(bk).float()
        if bt.ndim < 2:
            continue

        for mi in range(n):
            mk = rev_maps[mi].get(bk)
            if mk is None: continue
            mt = model_files[mi].get_tensor(mk).float()
            if mt.shape != bt.shape: continue

            delta = (mt - bt).reshape(bt.shape[0], -1)
            try:
                S = torch.linalg.svdvals(delta)
                S_sq = S * S
                total_energy = S_sq.sum().item()
                if total_energy < 1e-10:
                    eff_ranks[mi].append(0)
                    continue
                cumulative = 0
                for rank, sv in enumerate(S_sq, 1):
                    cumulative += sv.item()
                    if cumulative >= 0.9 * total_energy:
                        eff_ranks[mi].append(rank / len(S))  # Normalized 0-1
                        break
            except Exception:
                pass
            del delta, mt
        del bt

    del base_f
    for mf in model_files:
        del mf

    # Average effective rank per model (0 = focused, 1 = diffuse)
    return [sum(r)/len(r) if r else 0.5 for r in eff_ranks]


# ═════════════════════════════════════════════════════════════════
# PHASE 4: ABGELEITETE METRIKEN
# ═════════════════════════════════════════════════════════════════

def compute_derived(raw):
    n = raw["n"]
    names = raw["names"]
    mags = [math.sqrt(s) for s in raw["norm_sq"]]

    # ── Basis-Metriken ──
    cos_m = [[0.0]*n for _ in range(n)]
    l2_m = [[0.0]*n for _ in range(n)]
    conf_m = [[0.0]*n for _ in range(n)]

    for i, j in itertools.combinations(range(n), 2):
        ni = max(math.sqrt(raw["norm_sq"][i]), 1e-8)
        nj = max(math.sqrt(raw["norm_sq"][j]), 1e-8)
        cos_m[i][j] = cos_m[j][i] = round(raw["dot_prod"][i][j] / (ni * nj), 4)

        # L2: ||di - dj||^2 = ||di||^2 + ||dj||^2 - 2*dot
        d2 = raw["norm_sq"][i] + raw["norm_sq"][j] - 2 * raw["dot_prod"][i][j]
        l2_m[i][j] = l2_m[j][i] = round(math.sqrt(max(d2, 0)), 2)

        ac = raw["active_cnt"][i][j]
        conf_m[i][j] = conf_m[j][i] = round(raw["conflict_cnt"][i][j] / ac if ac > 0 else 0, 4)

    # ── Parallel/Perpendicular (neue Info %) ──
    new_info = [[0.0]*n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        cos2 = cos_m[i][j] ** 2
        perp_pct = (1.0 - cos2)  # Anteil neue Information
        new_info[i][j] = new_info[j][i] = round(perp_pct, 4)

    # ── Spearman Rank ──
    common_keys = set(raw["base_keys"])
    for mi in range(n):
        common_keys &= set(raw["key_mags"][mi].keys())
    common_list = sorted(common_keys)

    spearman_m = [[0.0]*n for _ in range(n)]
    if len(common_list) >= 10:
        ranks = {}
        for mi in range(n):
            vals = sorted([(raw["key_mags"][mi].get(k, 0), k) for k in common_list])
            ranks[mi] = {k: r for r, (_, k) in enumerate(vals)}
        nk = len(common_list)
        for i, j in itertools.combinations(range(n), 2):
            d2 = sum((ranks[i][k] - ranks[j][k])**2 for k in common_list)
            sp = 1 - (6 * d2) / (nk * (nk*nk - 1))
            spearman_m[i][j] = spearman_m[j][i] = round(sp, 4)

    # ── Top-K Overlap ──
    topk_m = [[0.0]*n for _ in range(n)]
    k_count = max(1, int(len(common_list) * 0.05))
    topk_sets = {}
    for mi in range(n):
        sk = sorted(common_list, key=lambda k: raw["key_mags"][mi].get(k, 0), reverse=True)
        topk_sets[mi] = set(sk[:k_count])
    for i, j in itertools.combinations(range(n), 2):
        union = topk_sets[i] | topk_sets[j]
        jaccard = len(topk_sets[i] & topk_sets[j]) / len(union) if union else 0
        topk_m[i][j] = topk_m[j][i] = round(jaccard, 4)

    # ── Per-Block Cosine + Conflicts ──
    block_cos = {}
    block_conf = {}
    for block in raw["block_dots"]:
        bc = [[0.0]*n for _ in range(n)]
        bf = [[0.0]*n for _ in range(n)]
        for i, j in itertools.combinations(range(n), 2):
            ni = max(math.sqrt(raw["block_norms"][block][i]), 1e-8)
            nj = max(math.sqrt(raw["block_norms"][block][j]), 1e-8)
            bc[i][j] = bc[j][i] = round(raw["block_dots"][block][i][j] / (ni * nj), 4)
            ba = raw["block_active"][block][i][j]
            bf[i][j] = bf[j][i] = round(raw["block_conflicts"][block][i][j] / ba if ba > 0 else 0, 4)
        block_cos[block] = bc
        block_conf[block] = bf

    # ── Lineage (optimale Base pro Delta) ──
    # L2 zwischen Modellen: ||model_i - model_j|| = ||delta_i - delta_j||
    lineage = {}
    for i in range(n):
        # Distanz zur Base = magnitude
        best_parent = "base"
        best_dist = mags[i]

        for j in range(n):
            if i == j: continue
            dist = l2_m[i][j]
            if dist < best_dist:
                best_dist = dist
                best_parent = names[j]

        lineage[names[i]] = {
            "best_parent": best_parent,
            "dist_to_parent": round(best_dist, 2),
            "dist_to_base": round(mags[i], 2),
            "improvement": round((mags[i] - best_dist) / max(mags[i], 1e-8) * 100, 1)
        }

    # ── Outlier-Score ──
    outlier_scores = []
    for mi in range(n):
        mean_val = raw["outlier_mean_sum"][mi] / max(raw["outlier_count"][mi], 1)
        ratio = raw["outlier_max"][mi] / max(mean_val, 1e-10)
        outlier_scores.append(round(ratio, 1))

    # ── Kumulative Interferenz-Vorhersage ──
    # Für jedes Delta: wie viel % überlebt im gemittelten Merge?
    # Projektion auf die durchschnittliche Delta-Richtung
    survival = [0.0] * n
    if n >= 2:
        # avg_direction proportional zu sum of unit vectors
        # Projection of delta_i onto avg = dot(d_i, avg) / ||avg||
        # With unit vectors: u_i = d_i / ||d_i||
        # avg = sum(u_i), survival_i = dot(u_i, avg/||avg||)
        # = (1/||avg||) * (1 + sum_{j!=i} cos(i,j))
        for i in range(n):
            if mags[i] < 1e-8:
                survival[i] = 0
                continue
            # Component of u_i along average direction
            dot_with_avg = 1.0  # dot with self = 1
            for j in range(n):
                if i != j:
                    dot_with_avg += cos_m[i][j]
            # Normalize by avg magnitude
            avg_mag_sq = n  # sum of ||u_i||^2 = n
            for ii, jj in itertools.combinations(range(n), 2):
                avg_mag_sq += 2 * cos_m[ii][jj]
            avg_mag = math.sqrt(max(avg_mag_sq, 1e-8))
            survival[i] = round(dot_with_avg / avg_mag * 100, 1)

    return {
        "names": names, "mags": [round(m, 2) for m in mags], "n": n,
        "cos": cos_m, "l2": l2_m, "conf": conf_m, "new_info": new_info,
        "spearman": spearman_m, "topk": topk_m,
        "block_cos": block_cos, "block_conf": block_conf,
        "lineage": lineage, "outlier_scores": outlier_scores,
        "survival": survival,
        "base_path": raw["base_path"], "model_paths": raw["model_paths"],
    }


# ═════════════════════════════════════════════════════════════════
# PHASE 5: EMPFEHLUNG
# ═════════════════════════════════════════════════════════════════

def merge_score(i, j, d):
    """Kombinierter Score: niedriger = besser für Merge."""
    return round(
        0.30 * d["cos"][i][j] +
        0.20 * d["conf"][i][j] +
        0.20 * d["topk"][i][j] +
        0.15 * d["spearman"][i][j] +
        0.15 * (1.0 - d["new_info"][i][j])
    , 4)


def _common_analysis(d, eff_ranks):
    """Gemeinsame Analyse für beide Strategien."""
    n, names, mags = d["n"], d["names"], d["mags"]

    # Redundanz
    redundant = []
    has_conflicts = False
    for i, j in itertools.combinations(range(n), 2):
        if d["cos"][i][j] > 0.8 and d["spearman"][i][j] > 0.7 and d["topk"][i][j] > 0.5:
            redundant.append((i, j, d["cos"][i][j]))
        if d["conf"][i][j] > 0.4:
            has_conflicts = True

    exclude = set()
    for i, j, cos in sorted(redundant, key=lambda x: -x[2]):
        if i not in exclude and j not in exclude:
            weaker = i if mags[i] < mags[j] else j
            exclude.add(weaker)
    for i in range(n):
        if mags[i] == 0:
            exclude.add(i)

    selected = [i for i in range(n) if i not in exclude]

    # Duplikations-Kandidaten
    ns = len(selected)
    dup_candidates = []
    for idx in selected:
        avg_ni = sum(d["new_info"][idx][j] for j in selected if j != idx) / max(ns-1, 1)
        if avg_ni > 0.85:
            dup_candidates.append((idx, avg_ni))

    # Lineage-Hints
    lineage_hints = {}
    for idx in selected:
        li = d["lineage"][names[idx]]
        if li["best_parent"] != "base" and li["improvement"] > 20:
            lineage_hints[idx] = li

    # Konfliktherde
    conflict_blocks = []
    for block in d["block_conf"]:
        avg_conf = 0
        cnt = 0
        for i, j in itertools.combinations(range(n), 2):
            if i in exclude or j in exclude: continue
            avg_conf += d["block_conf"][block][i][j]
            cnt += 1
        if cnt > 0: avg_conf /= cnt
        if avg_conf > 0.35:
            conflict_blocks.append((block, avg_conf))

    return {
        "redundant": redundant, "has_conflicts": has_conflicts,
        "exclude": exclude, "selected": selected,
        "dup_candidates": dup_candidates, "lineage_hints": lineage_hints,
        "conflict_blocks": conflict_blocks,
    }


def _format_header(d, exclude, redundant, selected, eff_ranks, dup_candidates):
    """Gemeinsamer Header für beide Strategien."""
    names, mags = d["names"], d["mags"]
    L = []

    if exclude:
        L.append("AUSGESCHLOSSEN:")
        for idx in exclude:
            if mags[idx] == 0:
                L.append(f"  ✗ {names[idx]} — Null-Delta")
            else:
                for i, j, c in redundant:
                    p = j if i == idx else (i if j == idx else None)
                    if p is not None:
                        L.append(f"  ✗ {names[idx]} — redundant zu {names[p]} (cos={c:+.4f})")
                        break
        L.append("")

    L.append("SIGNALQUALITÄT:")
    for idx in selected:
        er = eff_ranks[idx]
        surv = d["survival"][idx]
        quality = "★★★" if er < 0.3 else "★★" if er < 0.6 else "★"
        ow = " ⚠outlier" if d["outlier_scores"][idx] > 500 else ""
        dup = " ↑2x" if any(c[0] == idx for c in dup_candidates) else ""
        L.append(f"  {names[idx]:30s}  {quality} rank={er:.2f}  surv:{surv:.0f}%{ow}{dup}")

    return L


def build_recommendation(derived, eff_ranks, arch=""):
    d = derived
    n, names, mags = d["n"], d["names"], d["mags"]
    base_name = os.path.basename(d["base_path"]).replace(".safetensors", "")

    if n < 2:
        return "Zu wenige Modelle.", "", {}

    ca = _common_analysis(d, eff_ranks)
    selected = ca["selected"]
    ns = len(selected)
    if ns < 2:
        return "Nach Ausschluss zu wenige Modelle.", "", {}

    # Name-to-Path Mapping
    name_to_path = {names[i]: d["model_paths"][i] for i in range(n)}
    name_to_path["base"] = d["base_path"]

    # ═══ STRATEGIE A: Lineage-basiert (wie manuelles Rezept) ═══
    L_a = []
    L_a.extend(_format_header(d, ca["exclude"], ca["redundant"], selected, eff_ranks, ca["dup_candidates"]))
    L_a.append("")
    L_a.append("╔══════════════════════════════════════════════════════════╗")
    L_a.append("║  STRATEGIE A — Lineage-basiert (manueller Stil)        ║")
    L_a.append("╚══════════════════════════════════════════════════════════╝")
    L_a.append(f"\n  OrthoMerge Base: {base_name}")
    L_a.append(f"  theta_agg:       mean")
    L_a.append(f"  conflict_aware:  False")
    L_a.append(f"  direction_weight: theta")

    if ca["lineage_hints"]:
        L_a.append(f"\n  MULTI-BASE DELTAS (Lineage-optimiert):")
        for idx, li in ca["lineage_hints"].items():
            L_a.append(f"    {names[idx]:30s} ← Delta via '{li['best_parent']}'")
            L_a.append(f"      ({li['improvement']:.0f}% sauberer als via Base)")

    L_a.append(f"\n  Modelle (nach Magnitude):")
    sel_a = sorted(selected, key=lambda i: mags[i], reverse=True)
    dup_set = {c[0] for c in ca["dup_candidates"]}
    for rank, idx in enumerate(sel_a, 1):
        parent = ca["lineage_hints"].get(idx, {}).get("best_parent", "base")
        dup = " (2x)" if idx in dup_set else ""
        L_a.append(f"    {rank}. {names[idx]:28s}  mag:{mags[idx]:.0f}  via:{parent}{dup}")

    if ca["conflict_blocks"]:
        L_a.append(f"\n  KONFLIKTHERDE:")
        for block, conf in sorted(ca["conflict_blocks"], key=lambda x: -x[1]):
            L_a.append(f"    {block:35s}  {conf:.1%}")

    # ═══ STRATEGIE B: Automatisch (datengetrieben) ═══
    L_b = []
    L_b.append("")
    L_b.append("╔══════════════════════════════════════════════════════════╗")
    L_b.append("║  STRATEGIE B — Automatisch (datengetrieben)            ║")
    L_b.append("╚══════════════════════════════════════════════════════════╝")

    if ns <= 2:
        strat_b, theta_b, ca_b = "direct", "mean", ca["has_conflicts"]
    elif not ca["has_conflicts"]:
        strat_b, theta_b, ca_b = "all_at_once", "median", False
    else:
        strat_b, theta_b, ca_b = "pairwise", "mean", True

    strat_labels = {"direct": "Direkt", "all_at_once": "Alle auf einmal (median)", "pairwise": "Paarweise"}
    L_b.append(f"\n  OrthoMerge Base: {base_name}")
    L_b.append(f"  Strategie:       {strat_labels[strat_b]}")
    L_b.append(f"  theta_agg:       {theta_b}")
    L_b.append(f"  conflict_aware:  {ca_b}")

    # Auto-Pairing
    pairs_b, leftover_b = None, None
    sel_b = list(selected)
    if strat_b == "pairwise":
        ps = []
        for a, b in itertools.combinations(range(ns), 2):
            oi, oj = sel_b[a], sel_b[b]
            ps.append((merge_score(oi, oj, d), a, b))
        ps.sort()
        used = set()
        pairs_b = []
        for _, a, b in ps:
            if a not in used and b not in used:
                pairs_b.append((a, b)); used.add(a); used.add(b)
        leftover_b = [a for a in range(ns) if a not in used]

        L_b.append(f"\n  Schritt-für-Schritt:")
        ml = []
        for step, (a, b) in enumerate(pairs_b, 1):
            oi, oj = sel_b[a], sel_b[b]
            sc = merge_score(oi, oj, d)
            ml.append(f"Merge_{step}")
            L_b.append(f"    Schritt {step}: {names[oi]} + {names[oj]}  score={sc:.4f}")
        if leftover_b:
            for lo in leftover_b:
                ml.append(names[sel_b[lo]])
        L_b.append(f"    Final: Base + {' + '.join(ml)}")
    else:
        sel_b.sort(key=lambda i: mags[i], reverse=True)
        L_b.append(f"\n  Modelle (Einzel-Base, nach Magnitude):")
        for rank, idx in enumerate(sel_b, 1):
            L_b.append(f"    {rank}. {names[idx]:28s}  mag:{mags[idx]:.0f}")

    # Magnitude-Warnung
    sel_mags = [mags[i] for i in selected]
    if sel_mags and min(sel_mags) > 0 and max(sel_mags) > 3 * min(sel_mags):
        L_b.append(f"\n  ⚠ Magnitude-Spread: {max(sel_mags)/min(sel_mags):.1f}x")

    rec_text = "\n".join(L_a + L_b)

    # ═══ SCRIPTS ═══
    # Script A: Lineage-basiert
    script_a = _gen_script_lineage(
        d["base_path"], d["model_paths"], sel_a, names, arch,
        ca["dup_candidates"], ca["lineage_hints"], name_to_path)

    # Script B: Auto
    script_b = _gen_script_auto(
        strat_b, theta_b, ca_b, d["base_path"], d["model_paths"],
        sel_b, names, arch, pairs_b, leftover_b, ca["dup_candidates"])

    combined_script = (
        "# ╔══════════════════════════════════════════════════╗\n"
        "# ║  STRATEGIE A — Lineage-basiert                  ║\n"
        "# ╚══════════════════════════════════════════════════╝\n\n"
        + script_a
        + "\n\n"
        + "# ╔══════════════════════════════════════════════════╗\n"
        + "# ║  STRATEGIE B — Automatisch (auskommentiert)     ║\n"
        + "# ║  Zum Nutzen: Strategie A auskommentieren,       ║\n"
        + "# ║  Strategie B einkommentieren                    ║\n"
        + "# ╚══════════════════════════════════════════════════╝\n\n"
        + "\n".join("# " + line for line in script_b.split("\n"))
    )

    return rec_text, script_a, script_b, {"selected": selected}


# ═════════════════════════════════════════════════════════════════
# PHASE 6: SCRIPT-GENERIERUNG
# ═════════════════════════════════════════════════════════════════

def _vn(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()[:30]


def _dedup_varnames(names_list):
    vns = [_vn(s) for s in names_list]
    seen = {}
    for i, v in enumerate(vns):
        if v in seen: seen[v] += 1; vns[i] = f"{v}_{seen[v]}"
        else: seen[v] = 0
    return vns


def _post_merge_block(base_file_var="BASE_FILE"):
    """Gemeinsamer Post-Merge Fix Code."""
    return [
        "", "# ── Merge ausführen ──",
        'output_path = f"{MODEL_DIR}/{OUTPUT_NAME}"',
        "sd_mecha.merge(new,",
        "    output=output_path,",
        '    merge_device="cpu", merge_dtype=torch.float32,',
        "    output_dtype=torch.bfloat16, threads=1)",
        "",
        "# ── Post-Merge Fix ──",
        'print("Post-Merge: Prüfe fehlende Keys + NaN...")',
        f'base_sd = load_file(f"{{MODEL_DIR}}/{{{base_file_var}}}")',
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


def _gen_script_lineage(base_path, model_paths, selected, names, arch,
                        dup_candidates, lineage_hints, name_to_path):
    """Script mit Multi-Base Deltas und Duplikation."""
    bd = os.path.dirname(base_path)
    bf = os.path.basename(base_path)
    a_str = f'"{arch}"' if arch else '"FIXME"'

    sp = [model_paths[i] for i in selected]
    sel_names = [names[i] for i in selected]
    vns = _dedup_varnames(sel_names)
    dup_set = {c[0] for c in dup_candidates}

    # Finde alle unique Parents (Base + Lineage-Parents)
    parent_paths = {"base": base_path}
    delta_parent = {}  # var_index -> parent_var_name
    for vi, idx in enumerate(selected):
        if idx in lineage_hints:
            pname = lineage_hints[idx]["best_parent"]
            if pname in name_to_path:
                parent_paths[pname] = name_to_path[pname]
                delta_parent[vi] = pname
            else:
                delta_parent[vi] = "base"
        else:
            delta_parent[vi] = "base"

    # Parent Variablennamen
    parent_vars = {}
    for pname, ppath in parent_paths.items():
        if pname == "base":
            parent_vars[pname] = "base"
        else:
            parent_vars[pname] = "parent_" + _vn(pname)

    L = [
        "import sd_mecha", "import torch", "import os",
        "from OrthoMerge import orthomergev2",
        "from safetensors.torch import load_file, save_file", "",
        "sd_mecha.set_log_level()", "",
        f'MODEL_DIR = "{bd}"', f'ARCH = {a_str}',
        f'BASE_FILE = "{bf}"',
        f'OUTPUT_NAME = "OrthoMerge_result.safetensors"', "",
        "# ── Base + Parents ──",
        f'base = sd_mecha.model(f"{{MODEL_DIR}}/{{BASE_FILE}}", ARCH)',
    ]

    # Lade zusätzliche Parents
    for pname, pvar in parent_vars.items():
        if pname != "base":
            pfile = os.path.basename(parent_paths[pname])
            L.append(f'{pvar} = sd_mecha.model(f"{{MODEL_DIR}}/{pfile}", ARCH)  # Lineage-Parent')

    L.append("")
    L.append("# ── Modelle ──")
    for path, var in zip(sp, vns):
        L.append(f'{var} = sd_mecha.model(f"{{MODEL_DIR}}/{os.path.basename(path)}", ARCH)')

    L += ["", "subtract = sd_mecha.subtract", "", "# ── Deltas (Lineage-optimiert) ──"]
    for vi, var in enumerate(vns):
        pname = delta_parent.get(vi, "base")
        pvar = parent_vars[pname]
        comment = f"  # via {pname}" if pname != "base" else ""
        L.append(f"delta_{var} = subtract({var}, {pvar}){comment}")

    L.append("")
    L.append("# ── OrthoMerge (Lineage-Strategie: mean, keine Konflikterkennung) ──")

    delta_args = []
    for vi, var in enumerate(vns):
        delta_args.append(f"delta_{var}")
        if selected[vi] in dup_set:
            delta_args.append(f"delta_{var}")
            L.append(f"# ↑ {var} doppelt genommen (hohe Orthogonalität)")

    args = ", ".join(delta_args)
    L.append(f'new = orthomergev2(base, {args},')
    L.append(f'    alpha=1.0, theta_agg="mean", conflict_aware=False, direction_weight="theta")')

    L.extend(_post_merge_block())
    return "\n".join(L)


def _gen_script_auto(strategy, theta, ca, base_path, model_paths,
                     selected, names, arch, pairs, leftover, dup_candidates):
    """Script mit Einzel-Base, automatischer Strategie."""
    bd = os.path.dirname(base_path)
    bf = os.path.basename(base_path)
    a_str = f'"{arch}"' if arch else '"FIXME"'

    sp = [model_paths[i] for i in selected]
    sel_names = [names[i] for i in selected]
    vns = _dedup_varnames(sel_names)
    dup_set = {c[0] for c in dup_candidates}

    L = [
        "import sd_mecha", "import torch", "import os",
        "from OrthoMerge import orthomergev2",
        "from safetensors.torch import load_file, save_file", "",
        "sd_mecha.set_log_level()", "",
        f'MODEL_DIR = "{bd}"', f'ARCH = {a_str}',
        f'BASE_FILE = "{bf}"',
        f'OUTPUT_NAME = "OrthoMerge_result.safetensors"', "",
        "# ── Modelle ──",
        f'base = sd_mecha.model(f"{{MODEL_DIR}}/{{BASE_FILE}}", ARCH)',
    ]
    for path, var in zip(sp, vns):
        L.append(f'{var} = sd_mecha.model(f"{{MODEL_DIR}}/{os.path.basename(path)}", ARCH)')

    L += ["", "subtract = sd_mecha.subtract", "", "# ── Deltas (alle gegen Base) ──"]
    for var in vns:
        L.append(f"delta_{var} = subtract({var}, base)")

    L.append("")

    if strategy in ("direct", "all_at_once"):
        strat_name = "Direkt" if strategy == "direct" else f"Alle auf einmal ({theta})"
        L.append(f"# ── OrthoMerge (Auto-Strategie: {strat_name}) ──")
        delta_args = []
        for vi, var in enumerate(vns):
            delta_args.append(f"delta_{var}")
            if selected[vi] in dup_set:
                delta_args.append(f"delta_{var}")
        args = ", ".join(delta_args)
        L.append(f'new = orthomergev2(base, {args},')
        L.append(f'    alpha=1.0, theta_agg="{theta}", conflict_aware={ca})')

    elif strategy == "pairwise" and pairs:
        L.append("# ── OrthoMerge (Auto-Strategie: Paarweise) ──")
        for step, (a, b) in enumerate(pairs, 1):
            L.append(f'merge{step} = orthomergev2(base, delta_{vns[a]}, delta_{vns[b]},')
            L.append(f'    alpha=1.0, theta_agg="mean", conflict_aware=True)')
        L.append("")
        md = []
        for step in range(1, len(pairs)+1):
            L.append(f"delta_merge{step} = subtract(merge{step}, base)")
            md.append(f"delta_merge{step}")
        if leftover:
            for idx in leftover:
                md.append(f"delta_{vns[idx]}")
        L.append("")
        L.append(f'new = orthomergev2(base, {", ".join(md)},')
        L.append(f'    alpha=1.0, theta_agg="mean", conflict_aware=True)')

    L.extend(_post_merge_block())
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════
# AUSGABE
# ═════════════════════════════════════════════════════════════════

def print_matrix(m, names, title, fmt=".4f"):
    short = [n[:20] for n in names]
    w = max(len(s) for s in short) + 2
    print(f"\n{'═'*70}\n{title}\n{'═'*70}")
    print(" "*w + "".join(f"{s:>{w}}" for s in short))
    for i in range(len(names)):
        row = f"{short[i]:<{w}}"
        for j in range(len(names)):
            if i == j:
                row += f"{'---':>{w}}"
            elif fmt == ".1%":
                row += f"{m[i][j]:>{w}.1%}"
            elif fmt == ".2f":
                row += f"{m[i][j]:>{w}.2f}"
            elif fmt == ".0%":
                row += f"{m[i][j]:>{w}.0%}"
            else:
                row += f"{m[i][j]:>{w}{fmt}}"
        print(row)


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    arch, do_fix, topk, rank_samples = None, False, 0.05, 5
    positional = []

    i = 0
    while i < len(args):
        if args[i] == "--arch" and i+1 < len(args): arch = args[i+1]; i += 2
        elif args[i] == "--fix": do_fix = True; i += 1
        elif args[i] == "--topk" and i+1 < len(args): topk = float(args[i+1]); i += 2
        elif args[i] == "--rank-samples" and i+1 < len(args): rank_samples = int(args[i+1]); i += 2
        else: positional.append(args[i]); i += 1

    if len(positional) < 2:
        print(__doc__); sys.exit(1)

    base_path = os.path.abspath(positional[0])
    rest = positional[1:]

    if len(rest) == 1 and os.path.isdir(rest[0]):
        scan_dir = os.path.abspath(rest[0])
        model_paths = sorted(glob.glob(os.path.join(scan_dir, "*.safetensors")))
        model_paths = [p for p in model_paths if os.path.abspath(p) != base_path]
    else:
        model_paths = [os.path.abspath(p) for p in rest]

    if len(model_paths) < 2:
        print("Mindestens 2 Modelle nötig."); sys.exit(1)

    # ── Phase 1: Validierung ──
    base_f = safe_open(base_path, framework="pt", device="cpu")
    base_keys = list(base_f.keys())
    del base_f

    print(f"{'═'*70}\nPHASE 1: VALIDIERUNG\n{'═'*70}")
    print(f"Base: {os.path.basename(base_path)} ({len(base_keys)} Keys)\n")

    valid_paths = []
    for p in model_paths:
        v = validate_model(p, base_keys)
        status = "✅" if not v["fixes"] else "🔧"
        print(f"{status} {v['name']}: {v['common']}/{v['total']} Keys  {' | '.join(v['issues'])}")

        if v["fixes"] and do_fix:
            fix_model(p, v["fixes"], base_keys)
            print(f"   → Repariert")
            valid_paths.append(p)
        elif not v["fixes"]:
            valid_paths.append(p)
        else:
            print(f"   → Übersprungen (--fix nötig)")

    if len(valid_paths) < 2:
        print(f"\nNur {len(valid_paths)} kompatible Modelle."); sys.exit(1)

    # ── Phase 2: Streaming ──
    print(f"\n{'═'*70}\nPHASE 2: STREAMING-ANALYSE ({len(valid_paths)} Modelle)\n{'═'*70}")
    raw = streaming_core(base_path, valid_paths, topk_pct=topk)

    # Build reverse mappings for effective rank
    rev_maps = [build_key_mapping(p, base_keys) for p in valid_paths]

    # ── Phase 3: Signalqualität ──
    print(f"\n{'═'*70}\nPHASE 3: SIGNALQUALITÄT\n{'═'*70}")
    eff_ranks = compute_effective_rank(base_path, valid_paths, rev_maps,
                                       base_keys, raw["key_sizes"],
                                       n_samples=rank_samples)
    for mi, (name, er) in enumerate(zip(raw["names"], eff_ranks)):
        ols = raw["outlier_max"][mi] / max(raw["outlier_mean_sum"][mi] / max(raw["outlier_count"][mi], 1), 1e-10)
        print(f"  {name[:35]:35s}  eff_rank={er:.3f}  outlier_ratio={ols:.0f}")

    # ── Phase 4: Abgeleitete Metriken ──
    print(f"\n{'═'*70}\nPHASE 4: METRIKEN\n{'═'*70}")
    derived = compute_derived(raw)
    derived["outlier_scores"] = [
        round(raw["outlier_max"][mi] / max(raw["outlier_mean_sum"][mi] / max(raw["outlier_count"][mi], 1), 1e-10), 1)
        for mi in range(raw["n"])
    ]

    names = derived["names"]
    print_matrix(derived["cos"], names, "COSINE SIMILARITY")
    print_matrix(derived["l2"], names, "L2-DISTANZ", fmt=".2f")
    print_matrix(derived["new_info"], names, "NEUE INFORMATION (perpendicular %)", fmt=".0%")
    print_matrix(derived["spearman"], names, "SPEARMAN RANK")
    print_matrix(derived["topk"], names, "TOP-5% KEY OVERLAP")
    print_matrix(derived["conf"], names, "CONFLICT RATIO", fmt=".1%")

    print(f"\n{'═'*70}\nMAGNITUDE + SURVIVAL\n{'═'*70}")
    mx = max(derived["mags"]) if derived["mags"] else 1
    for mi, (name, mag) in enumerate(zip(names, derived["mags"])):
        bar = "█" * int(30 * mag / mx) if mx > 0 else ""
        surv = derived["survival"][mi]
        print(f"  {name[:30]:30s}  {mag:10.0f}  surv:{surv:5.1f}%  {bar}")

    print(f"\n{'═'*70}\nLINEAGE\n{'═'*70}")
    for name, li in derived["lineage"].items():
        marker = f" ← besser via '{li['best_parent']}' ({li['improvement']:.0f}% sauberer)" if li["best_parent"] != "base" and li["improvement"] > 20 else ""
        print(f"  {name[:35]:35s}  dist_base={li['dist_to_base']:.0f}{marker}")

    # ── Phase 5: Empfehlung ──
    print(f"\n{'═'*70}\nPHASE 5: EMPFEHLUNG\n{'═'*70}")
    rec, script_a, script_b, config = build_recommendation(derived, eff_ranks, arch=arch)
    print(rec)

    print(f"\n{'═'*70}\nMERGE-SCRIPT (Strategie A — Lineage)\n{'═'*70}\n{script_a}")
    print(f"\n{'═'*70}\nMERGE-SCRIPT (Strategie B — Auto)\n{'═'*70}\n{script_b}")

    sp = os.path.join(os.path.dirname(base_path), "merge_recipe_A.py")
    with open(sp, "w") as f:
        f.write(script_a + "\n")
    sp2 = os.path.join(os.path.dirname(base_path), "merge_recipe_B.py")
    with open(sp2, "w") as f:
        f.write(script_b + "\n")
    print(f"\n→ Strategie A: {sp}")
    print(f"→ Strategie B: {sp2}")
    print(f"{'═'*70}\nFertig.")


if __name__ == "__main__":
    main()
