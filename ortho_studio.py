#!/usr/bin/env python3
"""
OrthoMerge Studio v3 — GUI mit Validierung, Multi-Metrik Analyse, Merge & Preview

Voraussetzung: analyze_deltas.py muss im selben Ordner liegen.
Starten: python ortho_studio.py [--verbose]
"""

import os, sys, glob, time, math, io, subprocess, itertools

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv or "--v" in sys.argv
COMFYUI_DIR = os.path.expanduser("~/ComfyUI")
COMFYUI_API = "http://127.0.0.1:8188"

# Import Analyse-Engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_deltas import (
    validate_model, fix_model, streaming_analysis,
    build_recommendation, classify_keys, compute_merge_score,
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


# ─── Formatierung ─────────────────────────────────────────────────

def fmt_matrix(m, names, title, fmt=".4f"):
    short = [n[:18] for n in names]
    w = max(len(s) for s in short) + 2
    lines = [title, "─" * 60]
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
            else:
                row += f"{m[i][j]:>{w}{fmt}}"
        lines.append(row)
    return "\n".join(lines)


def fmt_magnitudes(mags, names):
    lines = ["MAGNITUDE", "─" * 50]
    mx = max(mags) if mags else 1
    for name, mag in zip(names, mags):
        bar = "█" * int(30 * mag / mx) if mx > 0 else ""
        lines.append(f"  {name[:28]:28s}  {mag:10.2f}  {bar}")
    return "\n".join(lines)


def fmt_blocks(block_cos, names):
    n = len(names)
    lines = ["PER-BLOCK COSINE", "─" * 50]
    for block in sorted(block_cos.keys()):
        bc = block_cos[block]
        sims = [bc[i][j] for i, j in itertools.combinations(range(n), 2)]
        avg = sum(sims) / len(sims) if sims else 0
        lines.append(f"  {block:38s}  avg={avg:+.4f}")
    return "\n".join(lines)


# ─── Merge ────────────────────────────────────────────────────────

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
                          cwd=COMFYUI_DIR, timeout=7200)
        return r.returncode == 0, r.stdout + "\n" + r.stderr
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ─── Preview ──────────────────────────────────────────────────────

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


# ─── Gradio App ───────────────────────────────────────────────────

def create_app():
    import gradio as gr

    with gr.Blocks(title="OrthoMerge Studio") as app:

        gr.Markdown("# 🔬 OrthoMerge Studio v3\n*Validierung → Analyse → Merge → Preview*")

        with gr.Row():
            arch_dd = gr.Dropdown(list(DEFAULT_DIRS.keys()), value="Flux.2 Klein",
                                 label="Architektur", scale=1)
            arch_cfg = gr.Textbox(value="flux2-klein", label="sd_mecha Config", scale=1)
            model_dir = gr.Textbox(value=DEFAULT_DIRS["Flux.2 Klein"],
                                  label="Modellverzeichnis", scale=3)

        with gr.Row():
            base_dd = gr.Dropdown([], label="Base-Modell", scale=2)
            model_cb = gr.CheckboxGroup([], label="Modelle (leer = alle)", scale=3)
            refresh = gr.Button("🔄", scale=1)

        def on_arch(choice):
            return DEFAULT_DIRS.get(choice, ""), ARCH_CONFIGS.get(choice, "")

        arch_dd.change(on_arch, [arch_dd], [model_dir, arch_cfg])

        def on_refresh(arch, mdir, cur_base):
            d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
            if not d or not os.path.isdir(d):
                return gr.update(choices=[]), gr.update(choices=[])
            files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))]
            return (gr.update(choices=files, value=[]),
                    gr.update(choices=files, value=cur_base if cur_base in files else None))

        refresh.click(on_refresh, [arch_dd, model_dir, base_dd], [model_cb, base_dd])

        with gr.Tabs():

            # ── Tab 1: Analyse ──
            with gr.Tab("📊 Analyse & Validierung"):
                with gr.Row():
                    auto_fix = gr.Checkbox(value=True, label="Auto-Fix (Prefix/FP8/Norm-Keys/CLIP/VAE)")
                    analyze_btn = gr.Button("🔬 Validierung + Analyse", variant="primary", size="lg")

                log_box = gr.Textbox(label="📋 Log", lines=15, interactive=False)

                with gr.Row():
                    matrix_box = gr.Textbox(label="Metriken", lines=18, interactive=False)
                    detail_box = gr.Textbox(label="Magnitude & Blocks", lines=18, interactive=False)

                rec_box = gr.Textbox(label="📋 Empfehlung", lines=14, interactive=False)
                script_box = gr.Code(label="Merge-Script", language="python", lines=22)

                def do_analysis(arch, mdir, base_name, selected, acfg, autofix):
                    d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
                    if not d or not os.path.isdir(d):
                        return "❌ Verzeichnis nicht gefunden", "", "", "", ""

                    base_path = os.path.join(d, base_name) if base_name else ""
                    if not base_path or not os.path.exists(base_path):
                        return "❌ Base-Modell nicht gefunden", "", "", "", ""

                    if selected and len(selected) >= 2:
                        candidates = [os.path.join(d, m) for m in selected if m != base_name]
                    else:
                        all_f = sorted(glob.glob(os.path.join(d, "*.safetensors")))
                        candidates = [f for f in all_f
                                     if os.path.abspath(f) != os.path.abspath(base_path)]

                    if len(candidates) < 2:
                        return "❌ Mindestens 2 Modelle nötig", "", "", "", ""

                    # ── Validierung ──
                    log_lines = []
                    log_lines.append(f"═══ VALIDIERUNG ═══")
                    log_lines.append(f"Base: {os.path.basename(base_path)}")

                    base_f = safe_open(base_path, framework="pt", device="cpu")
                    base_keys = list(base_f.keys())
                    del base_f
                    log_lines.append(f"Base: {len(base_keys)} Keys\n")

                    valid_paths = []
                    for p in candidates:
                        v = validate_model(p, base_keys)
                        if v["fixes"]:
                            fixes_str = ", ".join(v["fixes"])
                            log_lines.append(f"🔧 {v['name']}: {fixes_str}")
                            if autofix:
                                try:
                                    fixed = fix_model(p, v["fixes"], base_keys)
                                    valid_paths.append(fixed)
                                    log_lines.append(f"   → Repariert")
                                except Exception as e:
                                    log_lines.append(f"   → Fix fehlgeschlagen: {e}")
                            else:
                                log_lines.append(f"   → Übersprungen (Auto-Fix aus)")
                        else:
                            log_lines.append(f"✅ {v['name']}")
                            valid_paths.append(p)

                    if len(valid_paths) < 2:
                        log_lines.append(f"\n❌ Nur {len(valid_paths)} kompatible Modelle.")
                        return "\n".join(log_lines), "", "", "", ""

                    # ── Analyse ──
                    log_lines.append(f"\n═══ ANALYSE ({len(valid_paths)} Modelle) ═══")

                    def log_cb(msg):
                        log_lines.append(msg)
                        vlog(msg)

                    results = streaming_analysis(base_path, valid_paths, log_fn=log_cb)
                    names = results["names"]

                    log_text = "\n".join(log_lines)

                    # Matrizen
                    mat = fmt_matrix(results["cos_matrix"], names, "COSINE SIMILARITY")
                    mat += "\n\n" + fmt_matrix(results["l2_matrix"], names, "L2-DISTANZ", fmt=".2f")
                    mat += "\n\n" + fmt_matrix(results["spearman_matrix"], names, "SPEARMAN RANK")
                    mat += "\n\n" + fmt_matrix(results["topk_matrix"], names,
                                              f"TOP-{int(results['topk_pct']*100)}% OVERLAP")
                    mat += "\n\n" + fmt_matrix(results["conf_matrix"], names, "CONFLICT RATIO", fmt=".1%")

                    # Details
                    det = fmt_magnitudes(results["mags"], names)
                    det += "\n\n" + fmt_blocks(results["block_cos"], names)

                    # Empfehlung
                    rec, script, _ = build_recommendation(results, arch=acfg)

                    return log_text, mat, det, rec, script

                analyze_btn.click(
                    do_analysis,
                    [arch_dd, model_dir, base_dd, model_cb, arch_cfg, auto_fix],
                    [log_box, matrix_box, detail_box, rec_box, script_box]
                )

            # ── Tab 2: Merge ──
            with gr.Tab("🔀 Merge"):
                gr.Markdown("Script aus der Analyse wird automatisch übernommen.")
                merge_script = gr.Code(label="Merge-Script (editierbar)", language="python", lines=25)
                with gr.Row():
                    out_name = gr.Textbox(value="OrthoMerge_result.safetensors",
                                         label="Output-Name", scale=3)
                    merge_btn = gr.Button("🚀 Merge", variant="primary", scale=1)
                merge_log = gr.Textbox(label="Merge-Log", lines=15, interactive=False)

                script_box.change(lambda x: x, [script_box], [merge_script])

                def do_merge(script, oname):
                    if not script.strip():
                        return "❌ Kein Script"
                    if not oname.endswith(".safetensors"):
                        oname += ".safetensors"
                    ok, out = run_merge(script, oname)
                    return f"{'✅ Erfolgreich' if ok else '❌ Fehlgeschlagen'}\n\n{out}"

                merge_btn.click(do_merge, [merge_script, out_name], [merge_log])

            # ── Tab 3: Preview ──
            with gr.Tab("🖼️ Preview"):
                running = comfyui_ok()
                gr.Markdown(f"ComfyUI: {'🟢' if running else '🔴'} {COMFYUI_API}")

                with gr.Row():
                    prev_model = gr.Dropdown([], label="Modell", scale=3)
                    prev_refresh = gr.Button("🔄", scale=1)

                prev_prompt = gr.Textbox(
                    value="A woman in sunlit garden, photorealistic, detailed skin",
                    label="Prompt", lines=3)

                with gr.Row():
                    pw = gr.Number(value=1024, label="W", precision=0)
                    ph = gr.Number(value=1024, label="H", precision=0)
                    ps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    pc = gr.Slider(0, 20, value=1.0, step=0.5, label="CFG")

                prev_btn = gr.Button("🎨 Generieren", variant="primary", size="lg")
                prev_img = gr.Image(label="Vorschau", height=512)
                prev_status = gr.Textbox(label="Status", interactive=False)

                def prev_refresh_fn(arch, mdir):
                    d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
                    if not d:
                        return gr.update(choices=[])
                    return gr.update(choices=[os.path.basename(f)
                                             for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))])

                prev_refresh.click(prev_refresh_fn, [arch_dd, model_dir], [prev_model])

                def do_preview(arch, mdir, mname, prompt, w, h, steps, cfg):
                    d = mdir if arch == "Custom" else DEFAULT_DIRS.get(arch, "")
                    p = os.path.join(d, mname) if mname else ""
                    if not os.path.exists(p):
                        return None, "❌ Modell nicht gefunden"
                    return gen_preview(p, prompt, w, h, steps, cfg)

                prev_btn.click(do_preview,
                              [arch_dd, model_dir, prev_model, prev_prompt, pw, ph, ps, pc],
                              [prev_img, prev_status])

        app.load(on_refresh, [arch_dd, model_dir, base_dd], [model_cb, base_dd])

    return app


if __name__ == "__main__":
    if VERBOSE:
        print("OrthoMerge Studio v3 — Verbose")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
