#!/usr/bin/env python3
"""
OrthoMerge Studio v4 — GUI
Starten: python ortho_studio.py [--verbose]
Braucht: analyze_deltas.py im selben Ordner
"""

import os, sys, glob, time, math, io, subprocess, itertools

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv or "--v" in sys.argv
COMFYUI_DIR = os.path.expanduser("~/ComfyUI")
COMFYUI_API = "http://127.0.0.1:8188"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_deltas import (
    validate_model, fix_model, streaming_core, compute_effective_rank,
    compute_derived, build_recommendation, build_key_mapping,
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
            elif fmt == ".2f":
                row += f"{m[i][j]:>{w}.2f}"
            elif fmt == ".0%":
                row += f"{m[i][j]:>{w}.0%}"
            else:
                row += f"{m[i][j]:>{w}{fmt}}"
        lines.append(row)
    return "\n".join(lines)


def run_merge(script_text, output_name):
    if output_name:
        script_text = script_text.replace("OrthoMerge_result.safetensors", output_name)
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
        return requests.get(f"{COMFYUI_API}/system_stats", timeout=2).status_code == 200
    except Exception:
        return False


def gen_preview(model_path, prompt, w, h, steps, cfg):
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
              "inputs": {"unet_name": rel, "weight_dtype": "default"}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["11", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["13", 0], "vae": ["10", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": "ortho_preview"}},
        "10": {"class_type": "VAELoader",
               "inputs": {"vae_name": "ae.safetensors"}},
        "11": {"class_type": "DualCLIPLoader",
               "inputs": {"clip_name1": "clip_l.safetensors",
                          "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux"}},
        "12": {"class_type": "EmptyLatentImage",
               "inputs": {"width": int(w), "height": int(h), "batch_size": 1}},
        "13": {"class_type": "KSampler",
               "inputs": {"model": ["3", 0], "positive": ["6", 0], "negative": ["6", 0],
                          "latent_image": ["12", 0], "seed": int(time.time()) % 2**32,
                          "steps": int(steps), "cfg": float(cfg),
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    }
    try:
        r = requests.post(f"{COMFYUI_API}/prompt", json={"prompt": wf}, timeout=10)
        if r.status_code != 200:
            return None, f"API: {r.text[:200]}"
        pid = r.json().get("prompt_id")
        for _ in range(300):
            time.sleep(1)
            h2 = requests.get(f"{COMFYUI_API}/history/{pid}", timeout=5)
            if h2.status_code == 200 and pid in h2.json():
                hist = h2.json()[pid]
                if "outputs" in hist:
                    for nid, out in hist["outputs"].items():
                        if "images" in out:
                            ii = out["images"][0]
                            url = (f"{COMFYUI_API}/view?filename={ii['filename']}"
                                   f"&type={ii.get('type','output')}"
                                   f"&subfolder={ii.get('subfolder','')}")
                            ir = requests.get(url, timeout=10)
                            if ir.status_code == 200:
                                return Image.open(io.BytesIO(ir.content)), "✅ OK"
                if hist.get("status", {}).get("status_str") == "error":
                    return None, f"Fehler: {hist['status']}"
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


# ─── Gradio ───────────────────────────────────────────────────────

def create_app():
    import gradio as gr

    with gr.Blocks(title="OrthoMerge Studio v4") as app:
        gr.Markdown("# 🔬 OrthoMerge Studio v4\n*Validierung → Analyse → Signalqualität → Lineage → Empfehlung → Merge → Preview*")

        with gr.Row():
            arch_dd = gr.Dropdown(list(DEFAULT_DIRS.keys()), value="Flux.2 Klein",
                                 label="Architektur", scale=1)
            arch_cfg = gr.Textbox(value="flux2-klein", label="Config", scale=1)
            model_dir = gr.Textbox(value=DEFAULT_DIRS["Flux.2 Klein"],
                                  label="Verzeichnis", scale=3)

        with gr.Row():
            base_dd = gr.Dropdown([], label="Base-Modell", scale=2)
            model_cb = gr.CheckboxGroup([], label="Modelle (leer = alle)", scale=3)
            refresh = gr.Button("🔄", scale=1)

        def on_arch(c):
            return DEFAULT_DIRS.get(c, ""), ARCH_CONFIGS.get(c, "")
        arch_dd.change(on_arch, [arch_dd], [model_dir, arch_cfg])

        def on_refresh(arch, mdir, cur):
            d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
            if not d or not os.path.isdir(d):
                return gr.update(choices=[]), gr.update(choices=[])
            files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))]
            return gr.update(choices=files, value=[]), gr.update(choices=files, value=cur if cur in files else None)
        refresh.click(on_refresh, [arch_dd, model_dir, base_dd], [model_cb, base_dd])

        with gr.Tabs():
            with gr.Tab("📊 Analyse"):
                with gr.Row():
                    auto_fix = gr.Checkbox(value=True, label="Auto-Fix")
                    rank_samples = gr.Number(value=5, label="Rank-Samples", precision=0)
                    analyze_btn = gr.Button("🔬 Analyse starten", variant="primary", size="lg")

                log_box = gr.Textbox(label="📋 Log (6 Phasen)", lines=18, interactive=False)
                with gr.Row():
                    matrix_box = gr.Textbox(label="Metriken", lines=20, interactive=False)
                    detail_box = gr.Textbox(label="Signalqualität & Lineage", lines=20, interactive=False)
                rec_box = gr.Textbox(label="📋 Empfehlung", lines=18, interactive=False)

                with gr.Row():
                    strategy_radio = gr.Radio(
                        choices=["A — Lineage-basiert", "B — Automatisch"],
                        value="A — Lineage-basiert",
                        label="Strategie wählen", scale=2)

                script_box = gr.Code(label="Merge-Script (mit Post-Merge-Fix)", language="python", lines=25)
                script_a_hidden = gr.Textbox(visible=False)
                script_b_hidden = gr.Textbox(visible=False)

                def do_analysis(arch, mdir, base_name, selected, acfg, autofix, n_rank):
                    d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
                    if not d or not os.path.isdir(d):
                        return "❌ Verzeichnis", "", "", "", "", "", ""
                    base_path = os.path.join(d, base_name) if base_name else ""
                    if not os.path.exists(base_path):
                        return "❌ Base nicht gefunden", "", "", "", "", "", ""

                    if selected and len(selected) >= 2:
                        candidates = [os.path.join(d, m) for m in selected if m != base_name]
                    else:
                        all_f = sorted(glob.glob(os.path.join(d, "*.safetensors")))
                        candidates = [f for f in all_f if os.path.abspath(f) != os.path.abspath(base_path)]
                    if len(candidates) < 2:
                        return "❌ Mind. 2 Modelle", "", "", "", "", "", ""

                    log = []

                    # Phase 1
                    log.append("═══ PHASE 1: VALIDIERUNG ═══")
                    base_f = safe_open(base_path, framework="pt", device="cpu")
                    base_keys = list(base_f.keys())
                    del base_f
                    log.append(f"Base: {os.path.basename(base_path)} ({len(base_keys)} Keys)\n")

                    valid = []
                    for p in candidates:
                        v = validate_model(p, base_keys)
                        if v["fixes"]:
                            log.append(f"🔧 {v['name']}: {' | '.join(v['issues'])}")
                            if autofix:
                                try:
                                    fix_model(p, v["fixes"], base_keys)
                                    valid.append(p)
                                    log.append(f"   → Repariert")
                                except Exception as e:
                                    log.append(f"   → Fehler: {e}")
                        else:
                            log.append(f"✅ {v['name']}")
                            valid.append(p)

                    if len(valid) < 2:
                        return "\n".join(log) + "\n\n❌ Zu wenige kompatible Modelle", "", "", "", "", "", ""

                    # Phase 2
                    log.append(f"\n═══ PHASE 2: STREAMING ({len(valid)} Modelle) ═══")
                    def log_fn(msg):
                        log.append(msg)
                        vlog(msg)
                    raw = streaming_core(base_path, valid, log_fn=log_fn)

                    # Phase 3
                    log.append(f"\n═══ PHASE 3: SIGNALQUALITÄT ═══")
                    rev_maps = [build_key_mapping(p, base_keys) for p in valid]
                    eff_ranks = compute_effective_rank(base_path, valid, rev_maps,
                                                       base_keys, raw["key_sizes"],
                                                       n_samples=int(n_rank), log_fn=log_fn)
                    for mi, (name, er) in enumerate(zip(raw["names"], eff_ranks)):
                        log.append(f"  {name}: eff_rank={er:.3f}")

                    # Phase 4
                    log.append(f"\n═══ PHASE 4: METRIKEN ═══")
                    derived = compute_derived(raw)
                    derived["outlier_scores"] = [
                        round(raw["outlier_max"][mi] / max(raw["outlier_mean_sum"][mi] / max(raw["outlier_count"][mi], 1), 1e-10), 1)
                        for mi in range(raw["n"])
                    ]
                    names = derived["names"]

                    # Matrizen formatieren
                    mat = fmt_matrix(derived["cos"], names, "COSINE SIMILARITY")
                    mat += "\n\n" + fmt_matrix(derived["l2"], names, "L2-DISTANZ", fmt=".2f")
                    mat += "\n\n" + fmt_matrix(derived["new_info"], names, "NEUE INFO (%)", fmt=".0%")
                    mat += "\n\n" + fmt_matrix(derived["spearman"], names, "SPEARMAN RANK")
                    mat += "\n\n" + fmt_matrix(derived["topk"], names, "TOP-5% OVERLAP")
                    mat += "\n\n" + fmt_matrix(derived["conf"], names, "CONFLICT RATIO", fmt=".1%")

                    # Details
                    det_lines = ["MAGNITUDE + SURVIVAL", "─" * 50]
                    mx = max(derived["mags"]) if derived["mags"] else 1
                    for mi, (nm, mg) in enumerate(zip(names, derived["mags"])):
                        bar = "█" * int(25 * mg / mx) if mx > 0 else ""
                        det_lines.append(f"  {nm[:28]:28s} mag:{mg:8.0f} surv:{derived['survival'][mi]:5.1f}% {bar}")

                    det_lines += ["", "SIGNALQUALITÄT", "─" * 50]
                    for mi, nm in enumerate(names):
                        er = eff_ranks[mi]
                        q = "★★★" if er < 0.3 else "★★" if er < 0.6 else "★"
                        ol = derived["outlier_scores"][mi]
                        ow = " ⚠outlier" if ol > 500 else ""
                        det_lines.append(f"  {nm[:28]:28s} Fokus:{q} rank={er:.2f}{ow}")

                    det_lines += ["", "LINEAGE", "─" * 50]
                    for nm, li in derived["lineage"].items():
                        extra = ""
                        if li["best_parent"] != "base" and li["improvement"] > 20:
                            extra = f" ← via '{li['best_parent']}' ({li['improvement']:.0f}% besser)"
                        det_lines.append(f"  {nm[:28]:28s} dist={li['dist_to_base']:.0f}{extra}")

                    det = "\n".join(det_lines)

                    # Phase 5
                    log.append(f"\n═══ PHASE 5: EMPFEHLUNG ═══")
                    rec, script_a, script_b, _ = build_recommendation(derived, eff_ranks, arch=acfg)
                    log.append("Fertig.")

                    return "\n".join(log), mat, det, rec, script_a, script_a, script_b

                analyze_btn.click(do_analysis,
                    [arch_dd, model_dir, base_dd, model_cb, arch_cfg, auto_fix, rank_samples],
                    [log_box, matrix_box, detail_box, rec_box, script_box, script_a_hidden, script_b_hidden])

                def on_strategy_change(choice, sa, sb):
                    if "A" in choice:
                        return sa
                    return sb

                strategy_radio.change(on_strategy_change,
                    [strategy_radio, script_a_hidden, script_b_hidden],
                    [script_box])

            with gr.Tab("🔀 Merge"):
                merge_script = gr.Code(label="Script (editierbar)", language="python", lines=28)
                with gr.Row():
                    out_name = gr.Textbox(value="OrthoMerge_result.safetensors", label="Output", scale=3)
                    merge_btn = gr.Button("🚀 Merge", variant="primary", scale=1)
                merge_log = gr.Textbox(label="Log", lines=15, interactive=False)
                script_box.change(lambda x: x, [script_box], [merge_script])

                def do_merge(s, o):
                    if not s.strip(): return "❌ Kein Script"
                    if not o.endswith(".safetensors"): o += ".safetensors"
                    ok, out = run_merge(s, o)
                    return f"{'✅ OK' if ok else '❌ Fehler'}\n\n{out}"
                merge_btn.click(do_merge, [merge_script, out_name], [merge_log])

            with gr.Tab("🖼️ Preview"):
                gr.Markdown(f"ComfyUI: {'🟢' if comfyui_ok() else '🔴'} {COMFYUI_API}")
                with gr.Row():
                    prev_m = gr.Dropdown([], label="Modell", scale=3)
                    prev_r = gr.Button("🔄", scale=1)
                prev_p = gr.Textbox(value="A woman in sunlit garden, photorealistic, detailed skin",
                                   label="Prompt", lines=3)
                with gr.Row():
                    pw = gr.Number(value=1024, label="W", precision=0)
                    ph = gr.Number(value=1024, label="H", precision=0)
                    ps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    pc = gr.Slider(0, 20, value=1.0, step=0.5, label="CFG")
                prev_btn = gr.Button("🎨 Generieren", variant="primary", size="lg")
                prev_img = gr.Image(label="Vorschau", height=512)
                prev_st = gr.Textbox(label="Status", interactive=False)

                def rp(a, d):
                    d2 = d if a == "Custom" else DEFAULT_DIRS.get(a, "")
                    if not d2: return gr.update(choices=[])
                    return gr.update(choices=[os.path.basename(f) for f in sorted(glob.glob(os.path.join(d2, "*.safetensors")))])
                prev_r.click(rp, [arch_dd, model_dir], [prev_m])

                def dp(a, d, m, p, w, h, s, c):
                    d2 = d if a == "Custom" else DEFAULT_DIRS.get(a, "")
                    fp = os.path.join(d2, m) if m else ""
                    if not os.path.exists(fp): return None, "❌ Nicht gefunden"
                    return gen_preview(fp, p, w, h, s, c)
                prev_btn.click(dp, [arch_dd, model_dir, prev_m, prev_p, pw, ph, ps, pc], [prev_img, prev_st])

        app.load(on_refresh, [arch_dd, model_dir, base_dd], [model_cb, base_dd])
    return app


if __name__ == "__main__":
    if VERBOSE: print("OrthoMerge Studio v4 — Verbose")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
