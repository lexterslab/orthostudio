#!/usr/bin/env python3
"""
OrthoMerge Studio v5 — GUI mit Delta-Mixer + Exhaustive Optimierung
Starten: python ortho_studio_v5.py [--verbose]
Braucht: analyze_deltas_v5.py im selben Ordner
"""

import os, sys, glob, time, math, io, subprocess, json

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
COMFYUI_DIR = os.path.expanduser("~/ComfyUI")
COMFYUI_API = "http://127.0.0.1:8188"
MAX_MIXER_ROWS = 12

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_deltas_v5 import (
    validate_model, fix_model, streaming_core, compute_effective_rank,
    compute_derived, compute_quality_scores, exhaustive_optimize,
    generate_script_from_mixer, build_key_mapping,
)
from safetensors import safe_open

DEFAULT_DIRS = {
    "Flux.2 Klein": f"{COMFYUI_DIR}/models/diffusion_models/Flux",
    "Z-Image": f"{COMFYUI_DIR}/models/diffusion_models/Z-Image",
    "Chroma": f"{COMFYUI_DIR}/models/diffusion_models/Chroma",
    "Custom": "",
}
ARCH_CONFIGS = {
    "Flux.2 Klein": "flux2-klein",
    "Z-Image": "zimage-base",
    "Chroma": "chroma-flow",
    "Custom": "",
}


def vlog(msg):
    if VERBOSE:
        print(f"[V] {msg}", flush=True)


def fmt_matrix(m, names, title, fmt=".4f"):
    short = [n[:18] for n in names]
    w = max(len(s) for s in short) + 2
    lines = [title, "─" * 55]
    lines.append(" " * w + "".join(f"{s:>{w}}" for s in short))
    for i in range(len(names)):
        row = f"{short[i]:<{w}}"
        for j in range(len(names)):
            if i == j:
                row += f"{'---':>{w}}"
            elif fmt == ".1%":
                row += f"{m[i][j]:>{w}.1%}"
            elif fmt == ".0%":
                row += f"{m[i][j]:>{w}.0%}"
            elif fmt == ".2f":
                row += f"{m[i][j]:>{w}.2f}"
            else:
                row += f"{m[i][j]:>{w}{fmt}}"
        lines.append(row)
    return "\n".join(lines)


def run_merge(script_text, output_name):
    if output_name:
        script_text = script_text.replace(
            "OrthoMerge_result.safetensors", output_name)
    tmp = os.path.join(COMFYUI_DIR, "_tmp_merge.py")
    with open(tmp, "w") as f:
        f.write(script_text)
    try:
        py = os.path.join(COMFYUI_DIR, "venv", "bin", "python")
        if not os.path.exists(py):
            py = sys.executable
        r = subprocess.run([py, tmp], capture_output=True, text=True,
                          cwd=COMFYUI_DIR, timeout=14400)
        return r.returncode == 0, r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (>4h)"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def comfyui_ok():
    try:
        import requests
        return requests.get(
            f"{COMFYUI_API}/system_stats", timeout=2).status_code == 200
    except Exception:
        return False


# ─── Gradio App ──────────────────────────────────────────────────

def create_app():
    import gradio as gr

    with gr.Blocks(title="OrthoMerge Studio v5", css="""
        .mixer-header { background: #1a1a2e; padding: 10px; border-radius: 8px;
                        margin-bottom: 8px; }
        .row-enabled { border-left: 3px solid #4CAF50; padding-left: 8px; }
        .row-disabled { border-left: 3px solid #f44336; padding-left: 8px;
                        opacity: 0.6; }
    """) as app:

        gr.Markdown(
            "# 🔬 OrthoMerge Studio v5\n"
            "*Analyse → Exhaustive Optimierung → Mixer → Merge → Preview*"
        )

        # ── Hidden State ──
        analysis_state = gr.Textbox(visible=False, value="{}")
        mixer_json = gr.Textbox(visible=False, value="[]")

        # ── Top Bar ──
        with gr.Row():
            arch_dd = gr.Dropdown(
                list(DEFAULT_DIRS.keys()), value="Chroma",
                label="Architektur", scale=1)
            arch_cfg = gr.Textbox(
                value="chroma-flow", label="Config", scale=1)
            model_dir = gr.Textbox(
                value=DEFAULT_DIRS["Chroma"],
                label="Verzeichnis", scale=3)

        with gr.Row():
            base_dd = gr.Dropdown(
                [], label="Base-Modell (für OrthoMerge)", scale=2)
            model_cb = gr.CheckboxGroup(
                [], label="Modelle (leer = alle)", scale=3)
            refresh_btn = gr.Button("🔄", scale=1)

        def on_arch(c):
            return DEFAULT_DIRS.get(c, ""), ARCH_CONFIGS.get(c, "")
        arch_dd.change(on_arch, [arch_dd], [model_dir, arch_cfg])

        def on_refresh(arch, mdir, cur_base):
            d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
            if not d or not os.path.isdir(d):
                return gr.update(choices=[]), gr.update(choices=[])
            files = [os.path.basename(f)
                    for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))]
            return (
                gr.update(choices=files, value=[]),
                gr.update(choices=files,
                         value=cur_base if cur_base in files else None),
            )
        refresh_btn.click(on_refresh, [arch_dd, model_dir, base_dd],
                         [model_cb, base_dd])

        # ══════════════════════════════════════════════════════
        with gr.Tabs():

            # ── TAB 1: ANALYSE ──
            with gr.Tab("📊 Analyse"):
                with gr.Row():
                    auto_fix = gr.Checkbox(value=True, label="Auto-Fix")
                    rank_samples = gr.Number(
                        value=5, label="Rank-Samples", precision=0)
                    analyze_btn = gr.Button(
                        "🔬 Analyse + Optimierung starten",
                        variant="primary", size="lg")

                log_box = gr.Textbox(
                    label="📋 Log", lines=15, interactive=False)
                with gr.Row():
                    matrix_box = gr.Textbox(
                        label="Metriken (Cosine | Neue Info | Conflict)",
                        lines=18, interactive=False)
                    detail_box = gr.Textbox(
                        label="Qualität & Lineage & Optimierung",
                        lines=18, interactive=False)

            # ── TAB 2: MIXER ──
            with gr.Tab("🎛️ Mixer"):
                gr.Markdown(
                    "### Delta-Mixer\n"
                    "*Vorausgefüllt aus der Analyse. "
                    "Jede Zeile = ein Delta. "
                    "Passe Delta-Base, Alpha, Repeat frei an.*"
                )

                mixer_status = gr.Textbox(
                    label="Mixer-Status", interactive=False, lines=2)

                # Mixer-Zeilen: wir erstellen MAX_MIXER_ROWS Zeilen,
                # die initial versteckt sind
                mixer_rows_ui = []
                for row_idx in range(MAX_MIXER_ROWS):
                    visible = row_idx < 3  # Initial 3 sichtbar
                    with gr.Row(visible=visible) as row_container:
                        row_enable = gr.Checkbox(
                            value=True, label="✓",
                            scale=1, min_width=40)
                        row_model = gr.Dropdown(
                            choices=[], label="Modell",
                            scale=3, min_width=150)
                        row_delta_base = gr.Dropdown(
                            choices=["base"], label="Delta von",
                            value="base", scale=3, min_width=150)
                        row_alpha = gr.Slider(
                            0.0, 2.0, value=1.0, step=0.05,
                            label="α", scale=2, min_width=100)
                        row_repeat = gr.Slider(
                            1, 3, value=1, step=1,
                            label="×", scale=1, min_width=60)
                        row_info = gr.Textbox(
                            label="Info", interactive=False,
                            scale=3, min_width=200, lines=1)

                    mixer_rows_ui.append({
                        "container": row_container,
                        "enable": row_enable,
                        "model": row_model,
                        "delta_base": row_delta_base,
                        "alpha": row_alpha,
                        "repeat": row_repeat,
                        "info": row_info,
                    })

                with gr.Row():
                    add_row_btn = gr.Button("➕ Zeile hinzufügen", scale=1)
                    remove_row_btn = gr.Button("➖ Zeile entfernen", scale=1)
                    num_visible = gr.Number(
                        value=3, visible=False, precision=0)

                # Zeilen-Steuerung
                def add_row(n_vis, *args):
                    n_vis = int(min(n_vis + 1, MAX_MIXER_ROWS))
                    updates = []
                    for i in range(MAX_MIXER_ROWS):
                        updates.append(gr.update(visible=i < n_vis))
                    return [n_vis] + updates

                def remove_row(n_vis, *args):
                    n_vis = int(max(n_vis - 1, 1))
                    updates = []
                    for i in range(MAX_MIXER_ROWS):
                        updates.append(gr.update(visible=i < n_vis))
                    return [n_vis] + updates

                add_row_btn.click(
                    add_row,
                    [num_visible],
                    [num_visible] + [r["container"]
                                     for r in mixer_rows_ui])
                remove_row_btn.click(
                    remove_row,
                    [num_visible],
                    [num_visible] + [r["container"]
                                     for r in mixer_rows_ui])

                # ── Merge-Parameter ──
                gr.Markdown("### Merge-Parameter")
                with gr.Row():
                    theta_agg = gr.Dropdown(
                        ["mean", "median", "sum"],
                        value="mean", label="theta_agg", scale=1)
                    conflict_aware = gr.Checkbox(
                        value=False, label="conflict_aware", scale=1)
                    direction_weight = gr.Dropdown(
                        ["theta", "magnitude", "equal"],
                        value="theta",
                        label="direction_weight", scale=1)

                with gr.Row():
                    tex_boost_dd = gr.Dropdown(
                        choices=["(keiner)"], value="(keiner)",
                        label="Texture-Boost Modell", scale=2)
                    tex_boost_alpha = gr.Slider(
                        0.0, 0.5, value=0.0, step=0.05,
                        label="Boost Alpha", scale=1)

                gen_script_btn = gr.Button(
                    "📝 Script generieren", variant="primary", size="lg")
                script_box = gr.Code(
                    label="Merge-Script", language="python", lines=25)

            # ── TAB 3: MERGE ──
            with gr.Tab("🔀 Merge"):
                merge_script = gr.Code(
                    label="Script (editierbar)", language="python", lines=28)
                with gr.Row():
                    out_name = gr.Textbox(
                        value="OrthoMerge_result.safetensors",
                        label="Output", scale=3)
                    merge_btn = gr.Button(
                        "🚀 Merge", variant="primary", scale=1)
                merge_log = gr.Textbox(
                    label="Log", lines=15, interactive=False)
                script_box.change(
                    lambda x: x, [script_box], [merge_script])

                def do_merge(s, o):
                    if not s.strip():
                        return "❌ Kein Script"
                    if not o.endswith(".safetensors"):
                        o += ".safetensors"
                    ok, out = run_merge(s, o)
                    return f"{'✅ OK' if ok else '❌ Fehler'}\n\n{out}"
                merge_btn.click(
                    do_merge, [merge_script, out_name], [merge_log])

            # ── TAB 4: PREVIEW ──
            with gr.Tab("🖼️ Preview"):
                gr.Markdown(
                    f"ComfyUI: "
                    f"{'🟢' if comfyui_ok() else '🔴'} {COMFYUI_API}")
                with gr.Row():
                    prev_m = gr.Dropdown([], label="Modell", scale=3)
                    prev_r = gr.Button("🔄", scale=1)
                prev_p = gr.Textbox(
                    value="A woman in sunlit garden, photorealistic, "
                          "detailed skin",
                    label="Prompt", lines=3)
                with gr.Row():
                    pw = gr.Number(value=1024, label="W", precision=0)
                    ph = gr.Number(value=1024, label="H", precision=0)
                    ps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    pc = gr.Slider(
                        0, 20, value=1.0, step=0.5, label="CFG")
                prev_btn = gr.Button(
                    "🎨 Generieren", variant="primary", size="lg")
                prev_img = gr.Image(label="Vorschau", height=512)
                prev_st = gr.Textbox(
                    label="Status", interactive=False)

                def rp(a, d):
                    d2 = d if a == "Custom" else DEFAULT_DIRS.get(a, "")
                    if not d2:
                        return gr.update(choices=[])
                    return gr.update(choices=[
                        os.path.basename(f) for f in sorted(
                            glob.glob(os.path.join(d2, "*.safetensors")))])
                prev_r.click(rp, [arch_dd, model_dir], [prev_m])

                def dp(a, d, m, p, w, h, s, c):
                    d2 = d if a == "Custom" else DEFAULT_DIRS.get(a, "")
                    fp = os.path.join(d2, m) if m else ""
                    if not os.path.exists(fp):
                        return None, "❌ Nicht gefunden"
                    from ortho_studio_v5 import gen_preview
                    return gen_preview(fp, p, w, h, s, c)
                prev_btn.click(
                    dp,
                    [arch_dd, model_dir, prev_m, prev_p, pw, ph, ps, pc],
                    [prev_img, prev_st])

        # ══════════════════════════════════════════════════════
        # ANALYSE → MIXER PREFILL
        # ══════════════════════════════════════════════════════

        def do_analysis(arch, mdir, base_name, selected, acfg,
                       autofix, n_rank):
            d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
            if not d or not os.path.isdir(d):
                empty_updates = (
                    [gr.update()] * (MAX_MIXER_ROWS * 6)
                    + [gr.update(), gr.update(), gr.update()]
                )
                return (
                    "❌ Verzeichnis nicht gefunden", "", "",
                    "{}", "[]", *empty_updates, gr.update()
                )

            base_path = os.path.join(d, base_name) if base_name else ""
            if not os.path.exists(base_path):
                empty_updates = (
                    [gr.update()] * (MAX_MIXER_ROWS * 6)
                    + [gr.update(), gr.update(), gr.update()]
                )
                return (
                    "❌ Base nicht gefunden", "", "",
                    "{}", "[]", *empty_updates, gr.update()
                )

            if selected and len(selected) >= 2:
                candidates = [os.path.join(d, m)
                             for m in selected if m != base_name]
            else:
                all_f = sorted(
                    glob.glob(os.path.join(d, "*.safetensors")))
                candidates = [
                    f for f in all_f
                    if os.path.abspath(f) != os.path.abspath(base_path)
                ]
            if len(candidates) < 2:
                empty_updates = (
                    [gr.update()] * (MAX_MIXER_ROWS * 6)
                    + [gr.update(), gr.update(), gr.update()]
                )
                return (
                    "❌ Mind. 2 Modelle nötig", "", "",
                    "{}", "[]", *empty_updates, gr.update()
                )

            log = []

            # Phase 1
            log.append("═══ PHASE 1: VALIDIERUNG ═══")
            base_f = safe_open(
                base_path, framework="pt", device="cpu")
            base_keys = list(base_f.keys())
            del base_f
            log.append(
                f"Base: {os.path.basename(base_path)} "
                f"({len(base_keys)} Keys)\n")

            valid = []
            for p in candidates:
                v = validate_model(p, base_keys)
                if v["fixes"]:
                    log.append(
                        f"🔧 {v['name']}: {' | '.join(v['issues'])}")
                    if autofix:
                        try:
                            fix_model(p, v["fixes"], base_keys)
                            valid.append(p)
                            log.append("   → Repariert")
                        except Exception as e:
                            log.append(f"   → Fix-Fehler: {e}")
                else:
                    log.append(f"✅ {v['name']}")
                    valid.append(p)

            if len(valid) < 2:
                empty_updates = (
                    [gr.update()] * (MAX_MIXER_ROWS * 6)
                    + [gr.update(), gr.update(), gr.update()]
                )
                return (
                    "\n".join(log) + "\n❌ Zu wenige Modelle",
                    "", "", "{}", "[]",
                    *empty_updates, gr.update()
                )

            # Phase 2
            log.append(
                f"\n═══ PHASE 2: FULL-PAIRWISE STREAMING "
                f"({len(valid)} Modelle) ═══")
            def log_fn(msg):
                log.append(msg)
                vlog(msg)
            raw = streaming_core(base_path, valid, log_fn=log_fn)

            # Phase 3
            log.append("\n═══ PHASE 3: SIGNALQUALITÄT ═══")
            rev_maps = [build_key_mapping(p, base_keys)
                       for p in valid]
            eff_ranks, snr_scores = compute_effective_rank(
                base_path, valid, rev_maps, base_keys,
                raw["key_sizes"],
                n_samples=int(n_rank), log_fn=log_fn)
            for mi, (er, snr) in enumerate(
                    zip(eff_ranks, snr_scores)):
                log.append(
                    f"  {raw['names'][mi+1]}: "
                    f"rank={er:.3f} SNR={snr:.1f}")

            # Phase 4
            log.append("\n═══ PHASE 4: METRIKEN ═══")
            derived = compute_derived(raw)
            quality_scores, quality_details = compute_quality_scores(
                derived, eff_ranks, snr_scores)
            names = derived["names"]

            for mi, (nm, q, qd) in enumerate(
                    zip(names, quality_scores, quality_details)):
                grade = ("A+" if q > 0.8 else "A" if q > 0.65
                        else "B" if q > 0.5 else "C" if q > 0.35
                        else "D")
                log.append(f"  {nm}: {grade}({q:.3f})")

            # Matrizen
            mat = fmt_matrix(
                derived["cos"], names, "COSINE SIMILARITY")
            mat += "\n\n" + fmt_matrix(
                derived["new_info"], names,
                "NEUE INFO (%)", fmt=".0%")
            mat += "\n\n" + fmt_matrix(
                derived["conf"], names,
                "CONFLICT RATIO", fmt=".1%")

            # Phase 5: EXHAUSTIVE OPTIMIERUNG
            log.append(
                "\n═══ PHASE 5: EXHAUSTIVE DELTA-BASE "
                "OPTIMIERUNG ═══")
            opt = exhaustive_optimize(
                raw, eff_ranks, snr_scores, quality_scores,
                log_fn=log_fn)

            # Details
            det_lines = ["OPTIMIERTE KONFIGURATION", "─" * 60]
            det_lines.append(
                f"Gesamt-Orthogonalität: "
                f"{opt['total_orthogonality']:.4f}")
            det_lines.append("")
            for row in opt["rows"]:
                status = "✅" if row["enabled"] else "❌"
                repeat_str = (f" ×{row['repeat']}"
                             if row["repeat"] > 1 else "")
                det_lines.append(
                    f"  {status} {row['model'][:25]:25s} "
                    f"← {row['delta_base']:15s} "
                    f"Q:{row['quality_grade']} "
                    f"mag:{row['magnitude_vs_parent']:.0f}"
                    f"{repeat_str}")
                det_lines.append(
                    f"      {row['reason']}")
                # Zeige Top-3 Alternativen
                alts = row.get("alternatives", [])[:3]
                if len(alts) > 1:
                    alt_str = " | ".join(
                        f"{a['parent']}({a['magnitude']:.0f})"
                        for a in alts)
                    det_lines.append(f"      Alternativen: {alt_str}")
                det_lines.append("")

            det_lines.append("LINEAGE-DETAILS")
            det_lines.append("─" * 60)
            for nm, li in derived["lineage"].items():
                extra = ""
                if (li["best_parent"] != "base"
                        and li["improvement"] > 5):
                    extra = (f" ← via '{li['best_parent']}' "
                            f"({li['improvement']:.0f}% besser)")
                det_lines.append(
                    f"  {nm[:28]:28s} "
                    f"dist={li['dist_to_base']:.0f}{extra}")

            det = "\n".join(det_lines)
            log.append("Analyse + Optimierung abgeschlossen.")

            # ── State für Mixer ──
            all_model_names = [
                os.path.basename(p).replace(".safetensors", "")
                for p in valid
            ]
            base_short = os.path.basename(
                base_path).replace(".safetensors", "")
            parent_choices = ["base"] + all_model_names

            state = json.dumps({
                "base_path": base_path,
                "model_dir": d,
                "arch": acfg,
                "model_names": all_model_names,
                "parent_choices": parent_choices,
            })

            mixer_rows_data = json.dumps(opt["rows"])

            # ── Mixer-UI Updates ──
            ui_updates = []
            n_active = len(opt["rows"])
            for ri in range(MAX_MIXER_ROWS):
                if ri < n_active:
                    row = opt["rows"][ri]
                    ui_updates.extend([
                        gr.update(value=row["enabled"]),         # enable
                        gr.update(                                # model
                            choices=all_model_names,
                            value=row["model"]),
                        gr.update(                                # delta_base
                            choices=parent_choices,
                            value=row["delta_base"]),
                        gr.update(value=row["alpha"]),            # alpha
                        gr.update(value=row["repeat"]),           # repeat
                        gr.update(value=row["reason"]),           # info
                    ])
                else:
                    ui_updates.extend([
                        gr.update(value=False),
                        gr.update(choices=all_model_names, value=None),
                        gr.update(choices=parent_choices,
                                 value="base"),
                        gr.update(value=1.0),
                        gr.update(value=1),
                        gr.update(value=""),
                    ])

            # Row visibility
            vis_updates = []
            for ri in range(MAX_MIXER_ROWS):
                vis_updates.append(
                    gr.update(visible=ri < n_active))

            # Texture boost dropdown
            tex_dd_update = gr.update(
                choices=["(keiner)"] + all_model_names,
                value="(keiner)")

            # num_visible
            num_vis_update = gr.update(value=n_active)

            return (
                "\n".join(log),  # log_box
                mat,             # matrix_box
                det,             # detail_box
                state,           # analysis_state
                mixer_rows_data, # mixer_json
                *ui_updates,     # 6 × MAX_MIXER_ROWS updates
                *vis_updates,    # MAX_MIXER_ROWS container updates
                tex_dd_update,   # tex_boost_dd
                num_vis_update,  # num_visible
            )

        # Build output list for analyze
        analyze_outputs = [
            log_box, matrix_box, detail_box,
            analysis_state, mixer_json,
        ]
        for r in mixer_rows_ui:
            analyze_outputs.extend([
                r["enable"], r["model"], r["delta_base"],
                r["alpha"], r["repeat"], r["info"],
            ])
        for r in mixer_rows_ui:
            analyze_outputs.append(r["container"])
        analyze_outputs.append(tex_boost_dd)
        analyze_outputs.append(num_visible)

        analyze_btn.click(
            do_analysis,
            [arch_dd, model_dir, base_dd, model_cb, arch_cfg,
             auto_fix, rank_samples],
            analyze_outputs,
        )

        # ══════════════════════════════════════════════════════
        # SCRIPT GENERIERUNG AUS MIXER
        # ══════════════════════════════════════════════════════

        def do_generate_script(state_str, ta, ca, dw,
                               tex_model, tex_alpha,
                               n_vis, *mixer_args):
            if not state_str or state_str == "{}":
                return "# Zuerst Analyse durchführen!"

            state = json.loads(state_str)
            n_vis = int(n_vis)

            # Parse mixer rows from UI
            # mixer_args: (enable, model, delta_base,
            #              alpha, repeat) × MAX_MIXER_ROWS
            rows = []
            for ri in range(min(n_vis, MAX_MIXER_ROWS)):
                offset = ri * 5
                if offset + 4 >= len(mixer_args):
                    break
                enabled = mixer_args[offset]
                model = mixer_args[offset + 1]
                delta_base = mixer_args[offset + 2]
                alpha = mixer_args[offset + 3]
                repeat = int(mixer_args[offset + 4])

                if model:
                    rows.append({
                        "model": model,
                        "delta_base": delta_base or "base",
                        "alpha": float(alpha),
                        "repeat": repeat,
                        "enabled": bool(enabled),
                    })

            tex_m = (tex_model
                    if tex_model and tex_model != "(keiner)"
                    else None)
            tex_a = float(tex_alpha) if tex_m else 0.0

            script = generate_script_from_mixer(
                state["base_path"],
                state["model_dir"],
                state["arch"],
                rows,
                theta_agg=ta,
                conflict_aware=ca,
                direction_weight=dw,
                texture_boost_model=tex_m,
                texture_boost_alpha=tex_a,
            )
            return script

        gen_script_inputs = [
            analysis_state, theta_agg, conflict_aware,
            direction_weight, tex_boost_dd, tex_boost_alpha,
            num_visible,
        ]
        for r in mixer_rows_ui:
            gen_script_inputs.extend([
                r["enable"], r["model"], r["delta_base"],
                r["alpha"], r["repeat"],
            ])

        gen_script_btn.click(
            do_generate_script,
            gen_script_inputs,
            [script_box],
        )

        # ── Initial Load ──
        app.load(on_refresh, [arch_dd, model_dir, base_dd],
                [model_cb, base_dd])

    return app


def gen_preview(model_path, prompt, w, h, steps, cfg):
    """ComfyUI-basierte Preview-Generierung."""
    try:
        import requests
        from PIL import Image
    except ImportError:
        return None, "pip install requests Pillow"
    if not comfyui_ok():
        return None, "ComfyUI nicht erreichbar!"
    rel = ""
    if "/diffusion_models/" in model_path:
        rel = model_path.split("/diffusion_models/")[1]
    wf = {
        "3": {"class_type": "UNETLoader",
              "inputs": {"unet_name": rel,
                        "weight_dtype": "default"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["11", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["13", 0], "vae": ["10", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0],
                        "filename_prefix": "ortho_preview"}},
        "10": {"class_type": "VAELoader",
               "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "DualCLIPLoader",
               "inputs": {"clip_name1": "clip_l.safetensors",
                          "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                          "type": "flux"}},
        "12": {"class_type": "EmptyLatentImage",
               "inputs": {"width": int(w), "height": int(h),
                          "batch_size": 1}},
        "13": {"class_type": "KSampler",
               "inputs": {
                   "model": ["3", 0], "positive": ["6", 0],
                   "negative": ["6", 0],
                   "latent_image": ["12", 0],
                   "seed": int(time.time()) % 2**32,
                   "steps": int(steps), "cfg": float(cfg),
                   "sampler_name": "euler", "scheduler": "simple",
                   "denoise": 1.0}},
    }
    try:
        import requests
        r = requests.post(
            f"{COMFYUI_API}/prompt",
            json={"prompt": wf}, timeout=10)
        if r.status_code != 200:
            return None, f"API: {r.text[:200]}"
        pid = r.json().get("prompt_id")
        for _ in range(300):
            time.sleep(1)
            h2 = requests.get(
                f"{COMFYUI_API}/history/{pid}", timeout=5)
            if h2.status_code == 200 and pid in h2.json():
                hist = h2.json()[pid]
                if "outputs" in hist:
                    for nid, out in hist["outputs"].items():
                        if "images" in out:
                            ii = out["images"][0]
                            url = (
                                f"{COMFYUI_API}/view?"
                                f"filename={ii['filename']}"
                                f"&type={ii.get('type','output')}"
                                f"&subfolder="
                                f"{ii.get('subfolder','')}")
                            ir = requests.get(url, timeout=10)
                            if ir.status_code == 200:
                                return (Image.open(
                                    io.BytesIO(ir.content)),
                                    "✅ OK")
                if (hist.get("status", {}).get("status_str")
                        == "error"):
                    return None, f"Fehler: {hist['status']}"
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


if __name__ == "__main__":
    if VERBOSE:
        print("OrthoMerge Studio v5 — Verbose")
    app = create_app()
    app.launch(
        server_name="0.0.0.0", server_port=7860,
        share=False, inbrowser=True)
