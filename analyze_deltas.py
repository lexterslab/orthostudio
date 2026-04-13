#!/usr/bin/env python3
"""
Delta Merge Analyzer v3 — Streaming, Multi-Metrik, Auto-Fix

Analysiert Modelle vor dem Merge:
  • Validiert Kompatibilität (CLIP/VAE erkennen, Prefix, FP8, Shapes)
  • Repariert automatisch (--fix)
  • Berechnet: Cosine Similarity, L2-Distanz, Spearman Rank, Top-K Overlap, Conflict Ratio
  • Gibt Merge-Empfehlung und fertiges Script aus

Verwendung:
    python analyze_deltas.py BASE ORDNER/ [--arch ARCH] [--fix]
    python analyze_deltas.py BASE MODEL_A MODEL_B [...] [--arch ARCH] [--fix]

Optionen:
    --arch ARCH   sd_mecha Config (flux2-klein, zimage-base, chroma-flow, ...)
    --fix         Inkompatible Modelle automatisch reparieren vor der Analyse
    --topk 0.05   Top-K Schwelle für Overlap (Default: 5%)
"""

import sys, os, glob, math, time, itertools
from collections import defaultdict
from safetensors import safe_open


# ─── Konstanten ───────────────────────────────────────────────────

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


# ─── Modell-Validierung ──────────────────────────────────────────

def classify_keys(keys):
    """Klassifiziert Keys in diffuser, clip, vae, unknown."""
    result = {"diffuser": [], "clip": [], "vae": [], "unknown": []}
    for k in keys:
        # Nach Prefix-Stripping prüfen
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
    """Prüft ein Modell auf Kompatibilität und gibt Diagnose + Fixes zurück."""
    name = os.path.basename(path)
    issues = []
    fixes = []

    with safe_open(path, framework="pt", device="cpu") as f:
        model_keys = list(f.keys())
        sample_dtype = str(f.get_tensor(model_keys[0]).dtype)

    # FP8?
    is_fp8 = "float8" in sample_dtype.lower()
    if is_fp8:
        issues.append(f"FP8-Modell ({sample_dtype})")
        fixes.append("convert_bf16")

    # Klassifiziere Keys
    classified = classify_keys(model_keys)
    if classified["clip"]:
        issues.append(f"Enthält CLIP-Keys ({len(classified['clip'])})")
        fixes.append("strip_clip")
    if classified["vae"]:
        issues.append(f"Enthält VAE-Keys ({len(classified['vae'])})")
        fixes.append("strip_vae")

    # Prefix?
    has_prefix = all(k.startswith("model.diffusion_model.") for k in model_keys)
    if has_prefix:
        issues.append("Hat 'model.diffusion_model.' Prefix")
        fixes.append("strip_prefix")

    # .weight/.scale?
    effective_keys = set(model_keys)
    if has_prefix:
        effective_keys = {k.removeprefix("model.diffusion_model.") for k in model_keys}

    base_set = set(base_keys)
    common = effective_keys & base_set
    if len(common) < len(base_set) * 0.5:
        remapped = set()
        remap_count = 0
        for k in effective_keys:
            if ".norm." in k and k.endswith(".weight"):
                alt = k.rsplit(".weight", 1)[0] + ".scale"
                remapped.add(alt)
                remap_count += 1
            else:
                remapped.add(k)
        new_common = remapped & base_set
        if len(new_common) > len(common):
            issues.append(f"{remap_count} Norm-Keys: .weight statt .scale")
            fixes.append("remap_norm")
            common = new_common

    # Shape-Check (stichprobenartig)
    if common:
        with safe_open(path, framework="pt", device="cpu") as f:
            mismatches = 0
            # Nur erste 5 Keys prüfen (nicht alles laden)
            checked = 0
            for mk in model_keys[:20]:
                ek = mk
                if has_prefix:
                    ek = mk.removeprefix("model.diffusion_model.")
                if ek in base_set:
                    # Shape aus Header (ohne Laden)
                    checked += 1
            # Vollständiger Check zu teuer — machen wir beim Streaming

    if not issues:
        issues.append("✅ Kompatibel")

    return {"name": name, "path": path, "issues": issues, "fixes": fixes,
            "is_fp8": is_fp8, "has_clip": bool(classified["clip"]),
            "has_vae": bool(classified["vae"]), "has_prefix": has_prefix,
            "common_keys": len(common), "total_keys": len(model_keys)}


def fix_model(path, fixes, base_keys, outdir=None):
    """Repariert ein Modell und speichert es."""
    import torch
    from safetensors.torch import load_file, save_file

    name = os.path.basename(path)
    sd = load_file(path)
    changed = False

    # Strip CLIP + VAE
    if "strip_clip" in fixes or "strip_vae" in fixes:
        classified = classify_keys(sd.keys())
        to_remove = set()
        if "strip_clip" in fixes:
            to_remove.update(classified["clip"])
        if "strip_vae" in fixes:
            to_remove.update(classified["vae"])
        if to_remove:
            sd = {k: v for k, v in sd.items() if k not in to_remove}
            print(f"    ✓ {len(to_remove)} CLIP/VAE-Keys entfernt")
            changed = True

    # Strip Prefix
    if "strip_prefix" in fixes:
        prefix = "model.diffusion_model."
        new_sd = {}
        for k, v in sd.items():
            new_sd[k.removeprefix(prefix)] = v
        sd = new_sd
        print(f"    ✓ Prefix entfernt")
        changed = True

    # Remap .weight → .scale
    if "remap_norm" in fixes:
        base_set = set(base_keys)
        new_sd = {}
        count = 0
        for k, v in sd.items():
            if ".norm." in k and k.endswith(".weight"):
                alt = k.rsplit(".weight", 1)[0] + ".scale"
                if alt in base_set and k not in base_set:
                    new_sd[alt] = v
                    count += 1
                    continue
            new_sd[k] = v
        sd = new_sd
        if count:
            print(f"    ✓ {count} Keys .weight → .scale")
            changed = True

    # FP8 → BF16
    if "convert_bf16" in fixes:
        count = 0
        for k in sd:
            if "float8" in str(sd[k].dtype).lower():
                sd[k] = sd[k].to(torch.bfloat16)
                count += 1
        if count:
            print(f"    ✓ {count} Keys FP8 → BF16")
            changed = True

    if changed:
        out_path = os.path.join(outdir, name) if outdir else path
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        save_file(sd, out_path)
        print(f"    → {out_path} ({len(sd)} Keys)")
        return out_path
    return path


# ─── Streaming-Analyse ───────────────────────────────────────────

def streaming_analysis(base_path, model_paths, topk_pct=0.05, log_fn=None):
    """Multi-Metrik Streaming-Analyse. RAM: O(n_models * n_keys) für Spearman/TopK."""

    def log(msg):
        if log_fn:
            log_fn(msg)
        print(msg, flush=True)

    n = len(model_paths)
    names = [os.path.basename(p).replace(".safetensors", "") for p in model_paths]

    log(f"═══ Analyse: {n} Modelle ═══")

    # Öffne lazy
    base_f = safe_open(base_path, framework="pt", device="cpu")
    base_keys = sorted(base_f.keys())
    log(f"Base: {os.path.basename(base_path)} ({len(base_keys)} Keys)")

    model_files = []
    reverse_mappings = []

    for mi, p in enumerate(model_paths):
        mf = safe_open(p, framework="pt", device="cpu")
        model_files.append(mf)
        mkeys = list(mf.keys())

        # Build mapping
        base_set = set(base_keys)
        fwd = {}
        for k in mkeys:
            ek = k
            if k.startswith("model.diffusion_model."):
                ek = k[len("model.diffusion_model."):]
            if ".norm." in ek and ek.endswith(".weight"):
                alt = ek.rsplit(".weight", 1)[0] + ".scale"
                if alt in base_set:
                    ek = alt
            if ek in base_set:
                fwd[k] = ek

        rev = {v: k for k, v in fwd.items()}
        reverse_mappings.append(rev)
        log(f"  {names[mi]}: {len(rev)}/{len(base_keys)} Keys")

    # Akkumulatoren — Matrizen
    dot_products = [[0.0] * n for _ in range(n)]
    norm_sq = [0.0] * n
    diff_sq = [[0.0] * n for _ in range(n)]  # ||delta_i - delta_j||^2
    conflict_count = [[0] * n for _ in range(n)]
    active_count = [[0] * n for _ in range(n)]

    block_dots = defaultdict(lambda: [[0.0] * n for _ in range(n)])
    block_norms = defaultdict(lambda: [0.0] * n)

    # Per-Key Magnitude für Spearman + Top-K
    key_magnitudes = {mi: {} for mi in range(n)}  # mi -> {key: magnitude}

    total_keys = len(base_keys)
    t_start = time.time()
    last_report = 0

    for ki, base_key in enumerate(base_keys):
        now = time.time()
        if ki % 10 == 0 or (now - last_report) > 5:
            pct = ki / total_keys * 100
            elapsed = now - t_start
            eta = (elapsed / max(ki, 1)) * (total_keys - ki)
            log(f"  [{pct:5.1f}%] {ki}/{total_keys} — {base_key[:45]}  ETA:{eta:.0f}s")
            last_report = now

        base_tensor = base_f.get_tensor(base_key)
        block = base_key.split(".")[0] if base_key.split(".")[1:2] and base_key.split(".")[1].isdigit() else "_misc"

        deltas = []
        for mi in range(n):
            mk = reverse_mappings[mi].get(base_key)
            if mk is None:
                deltas.append(None)
                continue
            mt = model_files[mi].get_tensor(mk)
            if mt.shape != base_tensor.shape:
                deltas.append(None)
                continue
            d = (mt.float() - base_tensor.float()).flatten()
            deltas.append(d)

            # Per-Key Magnitude
            key_magnitudes[mi][base_key] = d.norm().item()

        for i in range(n):
            if deltas[i] is None:
                continue
            di = deltas[i]
            di_sq = (di * di).sum().item()
            norm_sq[i] += di_sq
            block_norms[block][i] += di_sq

            for j in range(i + 1, n):
                if deltas[j] is None:
                    continue
                dj = deltas[j]

                dot = (di * dj).sum().item()
                dot_products[i][j] += dot
                dot_products[j][i] += dot
                block_dots[block][i][j] += dot
                block_dots[block][j][i] += dot

                # L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b)
                dj_sq = (dj * dj).sum().item()
                diff_sq[i][j] += di_sq + dj_sq - 2 * dot
                diff_sq[j][i] = diff_sq[i][j]

                both = (di != 0) & (dj != 0)
                na = both.sum().item()
                active_count[i][j] += na
                active_count[j][i] += na
                if na > 0:
                    nc = ((di[both] * dj[both]) < 0).sum().item()
                    conflict_count[i][j] += nc
                    conflict_count[j][i] += nc

        del deltas, base_tensor

    elapsed = time.time() - t_start
    log(f"  [100%] Fertig in {elapsed:.1f}s")

    # ── Metriken berechnen ──
    mags = [math.sqrt(s) for s in norm_sq]

    cos_m = [[0.0] * n for _ in range(n)]
    l2_m = [[0.0] * n for _ in range(n)]
    conf_m = [[0.0] * n for _ in range(n)]

    for i, j in itertools.combinations(range(n), 2):
        ni = math.sqrt(norm_sq[i]) if norm_sq[i] > 0 else 1e-8
        nj = math.sqrt(norm_sq[j]) if norm_sq[j] > 0 else 1e-8
        cos_m[i][j] = cos_m[j][i] = round(dot_products[i][j] / (ni * nj), 4)
        l2_m[i][j] = l2_m[j][i] = round(math.sqrt(max(diff_sq[i][j], 0)), 2)
        ac = active_count[i][j]
        conf_m[i][j] = conf_m[j][i] = round(conflict_count[i][j] / ac if ac > 0 else 0, 4)

    # Spearman Rank Correlation (auf per-Key Magnitudes)
    spearman_m = [[0.0] * n for _ in range(n)]
    common_keys = set(base_keys)
    for mi in range(n):
        common_keys &= set(key_magnitudes[mi].keys())
    common_list = sorted(common_keys)

    if len(common_list) >= 10:
        # Ranks pro Modell
        ranks = {}
        for mi in range(n):
            vals = [(key_magnitudes[mi].get(k, 0), k) for k in common_list]
            vals.sort()
            r = {}
            for rank, (_, k) in enumerate(vals):
                r[k] = rank
            ranks[mi] = r

        nk = len(common_list)
        for i, j in itertools.combinations(range(n), 2):
            d2_sum = sum((ranks[i][k] - ranks[j][k]) ** 2 for k in common_list)
            sp = 1 - (6 * d2_sum) / (nk * (nk * nk - 1))
            spearman_m[i][j] = spearman_m[j][i] = round(sp, 4)

    # Top-K Overlap
    topk_m = [[0.0] * n for _ in range(n)]
    k_count = max(1, int(len(common_list) * topk_pct))
    topk_sets = {}
    for mi in range(n):
        sorted_keys = sorted(common_list, key=lambda k: key_magnitudes[mi].get(k, 0), reverse=True)
        topk_sets[mi] = set(sorted_keys[:k_count])

    for i, j in itertools.combinations(range(n), 2):
        overlap = len(topk_sets[i] & topk_sets[j])
        jaccard = overlap / len(topk_sets[i] | topk_sets[j]) if topk_sets[i] | topk_sets[j] else 0
        topk_m[i][j] = topk_m[j][i] = round(jaccard, 4)

    # Per-Block Cosine
    block_cos = {}
    for block in sorted(block_dots.keys()):
        bc = [[0.0] * n for _ in range(n)]
        for i, j in itertools.combinations(range(n), 2):
            ni = math.sqrt(block_norms[block][i]) if block_norms[block][i] > 0 else 1e-8
            nj = math.sqrt(block_norms[block][j]) if block_norms[block][j] > 0 else 1e-8
            bc[i][j] = bc[j][i] = round(block_dots[block][i][j] / (ni * nj), 4)
        block_cos[block] = bc

    del base_f
    for mf in model_files:
        del mf

    return {
        "names": names, "cos_matrix": cos_m, "l2_matrix": l2_m,
        "conf_matrix": conf_m, "spearman_matrix": spearman_m,
        "topk_matrix": topk_m, "topk_pct": topk_pct,
        "mags": [round(m, 2) for m in mags], "block_cos": block_cos,
        "model_paths": model_paths, "base_path": base_path, "elapsed": elapsed,
    }


# ─── Empfehlung ───────────────────────────────────────────────────

def compute_merge_score(i, j, r):
    """Kombinierter Score: niedriger = besser für Merge."""
    cos = r["cos_matrix"][i][j]
    conf = r["conf_matrix"][i][j]
    topk = r["topk_matrix"][i][j]
    spear = r["spearman_matrix"][i][j]

    # Ideal: niedrige Cosine (ergänzen sich), niedriger Conflict,
    # niedriger Top-K Overlap (verschiedene Layer), niedriger Spearman
    score = (
        0.35 * cos +         # Richtungsähnlichkeit (niedrig = gut)
        0.25 * conf +        # Konflikte (niedrig = gut)
        0.20 * topk +        # Top-K Overlap (niedrig = gut)
        0.20 * spear          # Rank-Korrelation (niedrig = gut)
    )
    return round(score, 4)


def build_recommendation(results, arch=""):
    names = results["names"]
    cos_m = results["cos_matrix"]
    conf_m = results["conf_matrix"]
    l2_m = results["l2_matrix"]
    spear_m = results["spearman_matrix"]
    topk_m = results["topk_matrix"]
    mags = results["mags"]
    n = len(names)

    if n < 2:
        return "Zu wenige Modelle.", "", {}

    # Redundanz (Cosine > 0.8 UND Spearman > 0.7 UND Top-K > 0.5)
    redundant = []
    has_conflicts = False
    for i, j in itertools.combinations(range(n), 2):
        is_redundant = cos_m[i][j] > 0.8 and spear_m[i][j] > 0.7 and topk_m[i][j] > 0.5
        if is_redundant:
            redundant.append((i, j, cos_m[i][j]))
        if conf_m[i][j] > 0.4:
            has_conflicts = True

    # Ausschlüsse
    exclude = set()
    for i, j, cos in sorted(redundant, key=lambda x: -x[2]):
        if i not in exclude and j not in exclude:
            weaker = i if mags[i] < mags[j] else j
            exclude.add(weaker)

    # Zero-Magnitude ausschließen
    for i in range(n):
        if mags[i] == 0:
            exclude.add(i)

    selected_idx = [i for i in range(n) if i not in exclude]
    ns = len(selected_idx)
    if ns < 2:
        return "Nach Ausschluss bleiben zu wenige Modelle.", "", {}

    # Strategie
    if ns <= 2:
        strategy, theta, ca = "direct", "mean", has_conflicts
    elif not has_conflicts:
        strategy, theta, ca = "all_at_once", "median", False
    else:
        strategy, theta, ca = "pairwise", "mean", True

    # Pairing nach kombiniertem Score
    pairs = None
    leftover = None
    if strategy == "pairwise":
        pair_scores = []
        for a, b in itertools.combinations(range(ns), 2):
            oi, oj = selected_idx[a], selected_idx[b]
            score = compute_merge_score(oi, oj, results)
            pair_scores.append((score, a, b))
        pair_scores.sort()
        used = set()
        pairs = []
        for _, a, b in pair_scores:
            if a not in used and b not in used:
                pairs.append((a, b))
                used.add(a)
                used.add(b)
        leftover = [a for a in range(ns) if a not in used]
    else:
        selected_idx.sort(key=lambda i: mags[i], reverse=True)

    # ── Text ──
    base_name = os.path.basename(results["base_path"]).replace(".safetensors", "")
    lines = []

    if exclude:
        lines.append("AUSGESCHLOSSEN:")
        for idx in exclude:
            if mags[idx] == 0:
                lines.append(f"  ✗ {names[idx]} — Delta ist Null (identisch mit Base)")
            else:
                for i, j, cos in redundant:
                    partner = j if i == idx else (i if j == idx else None)
                    if partner is not None:
                        lines.append(f"  ✗ {names[idx]} — redundant zu {names[partner]} (cos={cos:+.4f})")
                        break
        lines.append("")

    lines.append("Aufgrund der Analyse empfiehlt sich folgender Merge:\n")
    lines.append(f"  Base:           {base_name}")
    labels = {"direct": "Direkt", "all_at_once": "Alle auf einmal", "pairwise": "Paarweise"}
    lines.append(f"  Strategie:      {labels[strategy]}")
    lines.append(f"  theta_agg:      {theta}")
    lines.append(f"  conflict_aware: {ca}")
    lines.append(f"  alpha:          1.0")

    if strategy != "pairwise":
        lines.append(f"\n  Modelle (nach Stärke sortiert):")
        for rank, i in enumerate(selected_idx, 1):
            lines.append(f"    {rank}. {names[i]}  (mag: {mags[i]:.2f})")
    else:
        lines.append(f"\n  Schritt-für-Schritt:")
        merge_labels = []
        for step, (a, b) in enumerate(pairs, 1):
            oi, oj = selected_idx[a], selected_idx[b]
            label = f"Zwischenmerge_{step}"
            merge_labels.append(label)
            score = compute_merge_score(oi, oj, results)
            lines.append(f"    Schritt {step}: {names[oi]} + {names[oj]} → {label}")
            lines.append(f"      cos={cos_m[oi][oj]:+.4f} conf={conf_m[oi][oj]:.1%} "
                        f"spear={spear_m[oi][oj]:+.4f} topk={topk_m[oi][oj]:.1%} "
                        f"score={score:.4f}")
        if leftover:
            for lo in leftover:
                merge_labels.append(names[selected_idx[lo]])
                lines.append(f"    Übrig: {names[selected_idx[lo]]}")
        lines.append(f"\n    Finaler Merge: Base + {' + '.join(merge_labels)}")

    sel_mags = [mags[i] for i in selected_idx]
    min_m = min(sel_mags) if sel_mags else 0
    if len(sel_mags) > 1 and min_m > 0 and max(sel_mags) > 3 * min_m:
        lines.append(f"\n  ⚠ Magnitude-Unterschied {max(sel_mags)/min_m:.1f}x")

    rec_text = "\n".join(lines)

    # Script
    script = _gen_script(strategy, theta, ca, results["base_path"],
                         results["model_paths"], selected_idx, names,
                         arch, pairs, leftover)

    return rec_text, script, {"strategy": strategy, "selected": selected_idx}


def _gen_script(strategy, theta, ca, base_path, model_paths,
                selected_idx, names, arch, pairs, leftover):
    bd = os.path.dirname(base_path)
    bf = os.path.basename(base_path)
    a_str = f'"{arch}"' if arch else '"FIXME"'

    def vn(s):
        return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()[:30]

    sp = [model_paths[i] for i in selected_idx]
    vns = [vn(names[i]) for i in selected_idx]
    seen = {}
    for i, v in enumerate(vns):
        if v in seen:
            seen[v] += 1
            vns[i] = f"{v}_{seen[v]}"
        else:
            seen[v] = 0

    L = ["import sd_mecha", "import torch", "from OrthoMerge import orthomergev2", "",
         "sd_mecha.set_log_level()", "",
         f'MODEL_DIR = "{bd}"', f'ARCH = {a_str}', "",
         f'base = sd_mecha.model(f"{{MODEL_DIR}}/{bf}", ARCH)']
    for p, v in zip(sp, vns):
        L.append(f'{v} = sd_mecha.model(f"{{MODEL_DIR}}/{os.path.basename(p)}", ARCH)')
    L += ["", "subtract = sd_mecha.subtract", ""]
    for v in vns:
        L.append(f"delta_{v} = subtract({v}, base)")
    L.append("")

    if strategy in ("direct", "all_at_once"):
        args = ", ".join(f"delta_{v}" for v in vns)
        L.append(f'new = orthomergev2(base, {args},')
        L.append(f'    alpha=1.0, theta_agg="{theta}", conflict_aware={ca})')
    elif strategy == "pairwise" and pairs:
        for step, (a, b) in enumerate(pairs, 1):
            L.append(f'merge{step} = orthomergev2(base, delta_{vns[a]}, delta_{vns[b]},')
            L.append(f'    alpha=1.0, theta_agg="mean", conflict_aware=True)')
        L.append("")
        md = []
        for step in range(1, len(pairs) + 1):
            L.append(f"delta_merge{step} = subtract(merge{step}, base)")
            md.append(f"delta_merge{step}")
        if leftover:
            for idx in leftover:
                md.append(f"delta_{vns[idx]}")
        L.append("")
        L.append(f'new = orthomergev2(base, {", ".join(md)},')
        L.append(f'    alpha=1.0, theta_agg="mean", conflict_aware=True)')

    L += ["", "sd_mecha.merge(new,",
          f'    output=f"{{MODEL_DIR}}/OrthoMerge_result.safetensors",',
          '    merge_device="cpu", merge_dtype=torch.float32,',
          "    output_dtype=torch.bfloat16, threads=1)"]
    return "\n".join(L)


# ─── Ausgabe ──────────────────────────────────────────────────────

def print_matrix(m, names, title, fmt=".4f"):
    short = [n[:20] for n in names]
    w = max(len(s) for s in short) + 2
    print(f"\n{'═' * 70}\n{title}\n{'═' * 70}")
    print(" " * w + "".join(f"{s:>{w}}" for s in short))
    for i in range(len(names)):
        row = f"{short[i]:<{w}}"
        for j in range(len(names)):
            if i == j:
                row += f"{'---':>{w}}"
            elif fmt == ".1%":
                row += f"{m[i][j]:>{w}.1%}"
            else:
                row += f"{m[i][j]:>{w}{fmt}}"
        print(row)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    arch = None
    do_fix = False
    topk = 0.05
    positional = []

    i = 0
    while i < len(args):
        if args[i] == "--arch" and i + 1 < len(args):
            arch = args[i + 1]; i += 2
        elif args[i] == "--fix":
            do_fix = True; i += 1
        elif args[i] == "--topk" and i + 1 < len(args):
            topk = float(args[i + 1]); i += 2
        else:
            positional.append(args[i]); i += 1

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

    # ── Validierung ──
    base_f = safe_open(base_path, framework="pt", device="cpu")
    base_keys = list(base_f.keys())
    del base_f

    print(f"{'═' * 70}\nMODELL-VALIDIERUNG\n{'═' * 70}")
    print(f"Base: {os.path.basename(base_path)} ({len(base_keys)} Keys)\n")

    valid_paths = []
    for p in model_paths:
        v = validate_model(p, base_keys)
        status = "✅" if not v["fixes"] else "🔧"
        print(f"{status} {v['name']}: {v['common_keys']}/{len(base_keys)} Keys")
        for issue in v["issues"]:
            if "✅" not in issue:
                print(f"    {issue}")

        if v["fixes"] and do_fix:
            print(f"  → Repariere...")
            fixed_path = fix_model(p, v["fixes"], base_keys)
            valid_paths.append(fixed_path)
        elif not v["fixes"]:
            valid_paths.append(p)
        elif v["fixes"]:
            print(f"    → Übersprungen (nutze --fix zum Reparieren)")

    if len(valid_paths) < 2:
        print(f"\nNur {len(valid_paths)} kompatible Modelle. Mindestens 2 nötig.")
        sys.exit(1)

    # ── Analyse ──
    results = streaming_analysis(base_path, valid_paths, topk_pct=topk)
    names = results["names"]

    print_matrix(results["cos_matrix"], names, "COSINE SIMILARITY")
    print_matrix(results["l2_matrix"], names, "L2-DISTANZ", fmt=".2f")
    print_matrix(results["spearman_matrix"], names, "SPEARMAN RANK CORRELATION")
    print_matrix(results["topk_matrix"], names, f"TOP-{int(topk*100)}% KEY OVERLAP (Jaccard)")
    print_matrix(results["conf_matrix"], names, "CONFLICT RATIO", fmt=".1%")

    print(f"\n{'═' * 70}\nMAGNITUDE\n{'═' * 70}")
    max_m = max(results["mags"]) if results["mags"] else 1
    for name, mag in zip(names, results["mags"]):
        bar = "█" * int(35 * mag / max_m) if max_m > 0 else ""
        print(f"  {name[:30]:30s}  {mag:12.2f}  {bar}")

    # ── Empfehlung ──
    rec, script, config = build_recommendation(results, arch=arch)
    print(f"\n{'═' * 70}\nMERGE-REZEPT\n{'═' * 70}\n{rec}")
    print(f"\n{'═' * 70}\nMERGE-SCRIPT\n{'═' * 70}\n{script}")

    # Speichere Script
    out_dir = os.path.dirname(base_path)
    script_path = os.path.join(out_dir, "merge_recipe.py")
    with open(script_path, "w") as f:
        f.write(script + "\n")
    print(f"\n→ Script gespeichert: {script_path}")


if __name__ == "__main__":
    main()
