import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "-q"], check=True)

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
from huggingface_hub import hf_hub_download
from collections import defaultdict

RATINGS_REPO = "ghananlpcommunity/ghana-tts-ratings"

LANG_NAMES = {
    "acd": "Akan (Akuapem Twi)", "ada": "Dangme", "akp": "Siwu",
    "any": "Anyi", "avn": "Avatime", "bib": "Bisa",
    "bim": "Bimoba", "biv": "Birifor", "bov": "Tuwuli",
    "bud": "Ntcham", "bwu": "Buli", "dag": "Dagbani",
    "dga": "Dagaare", "en": "English", "ewe": "Ewe",
    "fat": "Fante", "ffm": "Moba", "gjn": "Gonja",
    "gur": "Frafra", "hau": "Hausa", "kbp": "Kabiyé",
    "kdh": "Tem", "kma": "Konni", "kus": "Kusaal",
    "lef": "Lelemi", "lip": "Sekpele", "maw": "Mampruli",
    "mzw": "Deg", "naw": "Nawuri", "ncu": "Chumburung",
    "nko": "Nkonya", "ntr": "Delo", "nzi": "Nzema",
    "sfw": "Sehwi", "sig": "Paasaal", "sil": "Sisaali",
    "snw": "Safaliba", "tpm": "Tampulma", "vag": "Vagla",
    "xon": "Konkomba", "xsm": "Kasem",
}

def load_ratings():
    try:
        path = hf_hub_download(RATINGS_REPO, "ratings.jsonl", repo_type="dataset")
    except Exception:
        return {}, 0
    scores = defaultdict(list)
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            count += 1
            lang = r.get("language", "unknown")
            rating_str = r.get("rating", "")
            if rating_str == "thumbs_up":
                scores[lang].append(1)
            elif rating_str == "thumbs_down":
                scores[lang].append(0)
            else:
                try:
                    val = int(rating_str.split("/")[0])
                    scores[lang].append(val / 5)
                except (ValueError, IndexError, AttributeError):
                    pass
    return scores, count

def plot_chart():
    scores, total = load_ratings()
    if not scores:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No ratings yet.\nSubmit ratings from the Ghana TTS app!", 
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig, f"Total ratings: 0", gr.update(choices=[]), gr.update(value=[])

    langs = sorted(scores.keys())
    names = [LANG_NAMES.get(l, l) for l in langs]
    pcts = [(sum(scores[l]) / len(scores[l])) * 100 for l in langs]

    colors = plt.cm.viridis([p / 100 for p in pcts])
    fig, ax = plt.subplots(figsize=(12, max(6, len(langs) * 0.35)))
    bars = ax.barh(names, pcts, color=colors)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Approval Rate (%)")
    ax.set_title(f"TTS Quality by Language (based on {total} ratings)")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    fig.tight_layout()

    lang_choices = [(f"{LANG_NAMES.get(l, l)} ({l})", l) for l in langs]
    return fig, f"Total ratings: {total}", gr.update(choices=lang_choices), gr.update(value=langs)

def plot_selected(selected):
    scores, total = load_ratings()
    if not scores or not selected:
        return plot_chart()[0]
    pcts = {}
    for l in selected:
        if l in scores:
            pcts[l] = (sum(scores[l]) / len(scores[l])) * 100
    langs = sorted(pcts.keys(), key=lambda l: pcts[l], reverse=True)
    names = [LANG_NAMES.get(l, l) for l in langs]
    vals = [pcts[l] for l in langs]
    colors = plt.cm.viridis([v / 100 for v in vals])
    fig, ax = plt.subplots(figsize=(10, max(5, len(langs) * 0.5)))
    bars = ax.barh(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Rating (%)")
    ax.set_title(f"TTS Quality by Language (selected)")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig

def refresh():
    return plot_chart()

with gr.Blocks(title="Ghana TTS Ratings Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Ghana TTS Ratings Dashboard\nRatings collected from users via the [Ghana TTS space](https://huggingface.co/spaces/ghananlpcommunity/ghana-tts).")
    
    with gr.Row():
        stats = gr.Textbox(label="Stats", interactive=False, show_label=False, scale=1)
        refresh_btn = gr.Button("🔄 Refresh", variant="primary", scale=0, size="sm")
    
    plot = gr.Plot(label="Ratings by Language")
    
    with gr.Accordion("Filter Languages", open=False):
        lang_filter = gr.CheckboxGroup(choices=[], label="Select languages to show", info="Leave empty to show all")
        update_btn = gr.Button("Update Chart", variant="secondary", size="sm")
    
    demo.load(refresh, outputs=[plot, stats, lang_filter, lang_filter])
    refresh_btn.click(refresh, outputs=[plot, stats, lang_filter, lang_filter])
    update_btn.click(plot_selected, inputs=[lang_filter], outputs=[plot])

if __name__ == "__main__":
    demo.launch()
