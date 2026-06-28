import subprocess, sys, os, random, json, pathlib, datetime, io, base64, wave

subprocess.run([sys.executable, "-m", "pip", "install",
    "git+https://github.com/OpenBMB/VoxCPM.git", "--no-deps", "-q"], check=True)

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download, HfApi
from voxcpm import VoxCPM

MODEL_ID = "ghananlpcommunity/ghana-tts-72k"

RATINGS_FILE = "/tmp/ratings.jsonl"
RATINGS_REPO = "ghananlpcommunity/ghana-tts-ratings"
_HF_TOKEN = os.environ.get("HF_TOKEN")

def upload_ratings():
    try:
        if os.path.exists(RATINGS_FILE):
            HfApi(token=_HF_TOKEN).upload_file(
                path_or_fileobj=RATINGS_FILE,
                path_in_repo="ratings.jsonl",
                repo_id=RATINGS_REPO,
                repo_type="dataset",
            )
    except Exception:
        pass

def save_rating(lang, input_text, prompt_audio, duration_s, cfg, steps, rating, comment=""):
    entry = {
        "language": lang,
        "input_text": input_text,
        "prompt_audio": prompt_audio or "",
        "duration_s": round(duration_s, 2),
        "cfg_scale": cfg,
        "inference_steps": steps,
        "rating": rating,
        "comment": comment.strip(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    os.makedirs(os.path.dirname(RATINGS_FILE) or ".", exist_ok=True)
    with open(RATINGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    upload_ratings()

LANGUAGE_EXAMPLES = {
    "acd": [
        "Abiidɛbo mɔ mɛ kpaa gyan wura gɛwi.",
        "Iken-alɛɛbo mɔ, mɛ tiri abiidɛbo de mɛ' baa biidɛ mɔmɔ-asawuu de iken lii gigengen so sa mɔmɔ.",
        "Gɛbii pubɔrɔ gumu gi bo dabɛ-dabɛ fɛɛ, gibono i kpa too gimɔ ayaa adɛ so.",
        "Yenwura gi dɔɔ nyɛ aterenbi ne ɔ yii-yii ibu dɛnsɛ-dɛnsɛ ɔsowolɛ belɛ mɔ so.",
    ],
    "ada": [
        "Adesahi tsuo ɔ, a bɔ mɛ nɛ nɔ fɛɛ nɔ e ye e he, nɛ nɔ tsuaa nɔsɔ ngɛ odehe si himi kɛ he blɔhi a blɔ fa mi.",
    ],
    "akp": [
        "Ne ɔso si mito mikparama kayi ne, miɣɛ sɔ, 'Bo Ɔse gɔ i kato, tã marɔ fɔ iyere sɛɛ.",
        "Fɔ sigara iɖe si siba i bo ndɛ̃. Maabara fɔ kuɖɔɛ i kayiiso lɛ kumɛgɔ masɛ mabara i kato.",
        "Tã bo nɔme ikpɛnɛ aɖera.",
        "Su bo akpi tsɛ bo lɛ kumɛgɔ bosɛ bosu botsɛ mma loɣɛrɛ bo.",
        "Daatã sɔ boabo i kalakanyɔ ame, ɣɛɛ ɖi bo bɔrɛgu i ikpi ame.",
    ],
    "any": [],
    "avn": [],
    "bib": [
        "A n a hɩ ŋ nɩ, a ʊ: K'a ɩ yaa dam, k'a hɩ, a ʊ: Zɩ, a ka kʊ n n'ɩbɩɩ mɩŋŋa dɔ Woso a deem, k'ɩ cirbəə n bʊr.",
        "Wɔɔ dɔmɩm biyəə hɔbɩrɛ ka wɔɔ ʊ, k'ɩ wɔɔ mimbʊnyaarɔ sugur ka wɔɔ ʊ.",
        "Amba wɔɔ mɩŋŋɔɔ y'a sugur kam bɔɔ ɩ kʊrɔma kan wɔɔ kɩ rɔ wɔɔ ʊ bɩ m.",
    ],
    "bim": [],
    "biv": [],
    "bov": [],
    "bud": [
        "Unimbɔti di kaa báaa? Ní ki náań binib ki duu líì kitiŋ pu. Binibee mà l ki butì pá aaa. U pé e nyáń dalbee, ní ki bí yii u làá yìiǹ kí cɔŋkì u-niboolii gbanti, ní ki kpántì unyaŋgbanti ki níń cá bi-cee. U pé e bí i yìiǹ ki tin kɔ́ ubɔ nì u-nimpuu pu.",
    ],
    "dag": [
        "Bɛhig' be sokam sanimi, din pa la amii. Suhizɔbo be sokam sani; ka nambɔxu beni. Suhubɔhibo mi bi lan kɔŋ yigunaadam kam sani. Dimbɔŋɔ zaa wuhiya ka di tu kamaata ka ti zaa yu tab' hali ni ti puuni.",
    ],
    "dga": [
        "Nengsaala zaa ba nang dɔge so la o menga, ka o ne o taaba zaa sengtaa noba emmo ane yɛlɛsoobo sobic poɔ. Ba dɔgɛɛ ba zaa ne yɛng ane yɛlɛ-iruu k'a da seng ka ba erɛ yɛlɛ korɔ taa a nga yɔɔmine.",
    ],
    "en": [
        "Government sitting on US$200m Akufo-Addo left to combat perennial flooding.",
        "Black Stars suffer 2-1 defeat in Group L at FIFA World Cup 2026.",
        "Police probe suspected foul play after woman found dead in Somanya.",
        "Maritime sector key to Ghana\u2019s growth.",
        "Pupils to recite daily anti-drug messages under intensified government campaign.",
        "Students now lacing ice cream with weed.",
    ],
    "ewe": [
        "Wodzi amegbetɔwo katã ablɔɖeviwoe eye wodzena bubu kple gomekpɔkpɔ ɔsɔe. Susu kple dzitsinya le wo dometɔ ɖesiaɖe si eyata wodze be woanɔ anyi le ɖekawɔwɔ blibo me.",
    ],
    "fat": [
        "Wɔwo ɑdɑsɑ nyinɑ to fɑhodzi mu, nɑ hɔn nyinɑ yɛ pɛr wɔ enyimnyɑm nɑ ndzinoɑ mu. Wɔmɑɑ hɔn nyinɑ ɑdwen nɑ tsibowɑ, nɑ ɔwɔ dɛ hɔn nkitɑhodzi mu ndzeyɛɛ dɑ no edzi dɛ wɔyɛ enuɑnom.",
    ],
    "ffm": [
        "Innama aadeeji fof poti, ndimɗidi e jibinannde to bannge hakkeeji. Eɓe ngoodi miijo e hakkilantaagal ete eɓe poti huufo ndirde e nder ɓ iynguyummaagu.",
    ],
    "gjn": [
        "Bu kurwe dimedi kikɛ mobe kumu so, nɛ mobe, eyilikpa, kesheŋ nɛ kashinteŋ maŋ kɔr eko peyɛ to. Nyinpela sa dimedi kikɛ lakal nɛ mfɛra fanɛ bu chena abarso kelepo so.",
    ],
    "gur": [
        "To Sɔ' n boe saazuo ha, ho yu'urɛ nara pɛnka, ho sɔ'ɔlom wa'ana, ho sunboolom tom teŋa wa zuo wo lan ane se'em saazuo ha la. Bo' to to zina dabaherɛ dia. Bahɛ to be'em bo to wo tomam n dite suguru bɔ'ɔra to taaba se'em la.",
    ],
    "hau": [
        "Su dai yan-adam, ana haifuwarsu ne duka yantattu, kuma kowannensu na da mutunci da hakkoki daidai da na kowa. Suna da hankali da tunani, saboda haka duk abin da za su aikata wa juna, ya kamata su yi shi a cikin yan-uwanci.",
    ],
    "kbp": [
        "Palʊlʊʊ ɛyaaa nɛ pa-tɩ yɔɔ wɛʊ kpaagbaa nɛ pɛwɛɛ kɩmaŋ wala ɛsɩndaa. Palʊlʊʊ-wɛ nɛ pɔ-lɔŋ nɛ pa-aɣzɩm; mbʊ yekina nɛ pɔsɔɔlɩ ɖama se pɛkɛ ɛyaa pa-tɩŋgɛ.",
    ],
    "kdh": [
        "Bánlʊrʊ́ʊ ɩrʊ́ báa weení na kezéńbíídi gɛ bɩka bɛdɛ́ɛ ɖɔɔzɩ́tɩ na yíkowá kɛgɛ́ɛ ɖéyí-ɖéyí gɛ. Bɔwɛná laakárɩ na ɩrʊ́tɩ bɩka bɩɩbɔ́ɔ́zɩ bɔcɔɔná ɖamá koobíre cɔwʊrɛ.",
    ],
    "kma": [],
    "kus": [],
    "lef": [],
    "lip": [
        "Mfó nya Yesu lɛtɛyi mǝ nkǝ, 'Lǝ bɛlɛ ola botoo, bɛtɛyi biǝnkǝ, 'Bo Anto, tǝ lǝ bakpasǝ fǝ diye. Fǝ sekadidi lǝ sibǝ. Beyifo lǝ fǝ lelabi ǝsuǝ lǝ kasɔ mfo fe kase inte kato.",
        "Nya lǝ efi bo abua atsyɛ bo, fe kase bo tsya leefi katsyɛ utidi saa wǝ laata bo kebu.",
    ],
    "maw": [
        "N\u2019i wunta\u014b\u014ba.",
        "Ani\u014b wula.",
        "I ba b\u025b wula?",
        "I ma b\u025b wula?",
        "I yuuri boonni la b\u0254?",
        "B\u0254 n\u014bwa?",
        "Ka ny\u025b la yiri.",
        "Ka ny\u025b la tiiya.",
        "N y\u025bl ka su\u014b\u014baa?",
    ],
    "mzw": [],
    "naw": [],
    "ncu": [
        "Kyo̱ŋbo̱ro̱ŋ awuye, mo̱ne̱ a ba, mo̱n(e̱) ne̱ mo̱ne̱ a le̱e̱ e̱maŋ se̱ mo̱ne̱ a ba-ɔ. Maŋ e̱ sa mo̱ne̱ aŋsɛ ooo.",
        "Ne̱ Yeesuu a be̱ŋŋaa mò̱ fe̱yɛ, 'Mo̱ne̱ e̱ ko̱re̱ ke̱bware̱ko̱re̱, mo̱nꞌ tɔwe̱ fe̱yɛ, 'Ane̱ se̱ Wuribware̱, sa a bo̱ kyo̱rɔ fo̱ ke̱nyare̱ timaa. Na fo̱ baa fo̱ kuwure‑o bo̱ gyi mfe̱e̱.",
    ],
    "nko": [
        "Yesu lɛbla amʋ́ ɔbɛɛ, 'Nɩ mlɔ́bɔ mpaɩ a, mlɩbɔ mʋ́ alɩ. 'Anɩ Sɩ, aha bʋbu fʋ ɩda. Ba begyi iwie.",
        "Ha anɩ atogyihɛá ɩbɔ́fʋn anɩ ekekegyiɛkɛ.",
        "Si anɩ lakpan kie anɩ, fɛ alɩá anɩtesikie aha ánɩ́ bʋtɔpʋ ɩla gyi anɩ. Mákpa anɩ wa ɩsɔkɩtɔ.",
    ],
    "ntr": [],
    "nzi": [
        "Menli muala di bɛ ti anwo na eza noko bɛsɛ wɔ dibilɛ nee adenlenyianlɛ nu. Bɛlɛ ndwenlenwo nee adwenle, yemɔti ɔwɔ kɛ bɛkile adiemayɛlɛ bɛmaa bɛ nwo ngoko.",
    ],
    "sfw": [],
    "sig": [
        "Ɛɛ rɛ Yesu basɩ tɩya ba a baa, 'Dɩ ma ko kɩ kyɛ dɩ ma kyʋwalɩ Wɩɩsɩ buloŋ, ma baa, 'Á Kuwo Wɩɩsɩ, leŋ dɩ ɩ feŋ gyɩŋ, leŋ dɩ ɩ koro hʋ ko.",
        "Tɩya ma á nyʋwa kɩdiilii kyɛɛ buloŋ.",
        "Kpa á wɩbɔmɔ kɩ kyɛ ma anɩɩ á mɛ aa kpaa á dɔŋtɩŋsɩ wɩbɔmɔ kɩ kyɛ ba gɛɛ.",
        "Aŋ lɩɩ ma wɩɩ buloŋ aa sɩ kaŋ ma we wɩbɔŋ yayɩ tɩyaŋ.",
    ],
    "sil": [],
    "snw": [],
    "tpm": [],
    "vag": [],
    "xon": [
        "Le Yesu bui bi, 'Ni yaa mee Uwumbɔr kan, ni bui ke, 'Tite Uwumbɔr, cha binib li san saayimbil; cha saanaan dan. Tiin timi din aawiin aajikaar. Cha timi aatunwanbir pinn timi; ba pu? ti mu di cha pinn binib bimɔk koo timi aataani ni na. Taa cha ti kan ntɔŋ.",
    ],
    "xsm": [
        "Ba loge nɔɔna maama se ba taa ye bedwe mo ba ŋwea de ba chega seini, ye fefeo teira kɔtaa. Wɛ pɛ ba swa de boboŋa mo se ba taa ye nubiu daane ye ba jege da ŋwaŋa.",
    ],
    "twi-asante": [
        "Nnipa nyinaa yɛ pɛ. Na wɔde adwene ne nyansa na abɔ obiara. Ɛno nti, ɛsɛ sɛ obiara dɔ ne yɔnko, bu ne yɔnko, di ne yɔnko ni.",
    ],
    "twi-akuapem": [
        "Wɔɑwo ɑdesɑmmɑ nyinɑɑ sɛ nnipɑ ɑ wɔwɔ ɑhofɑdi. Wɔn nyinɑɑ wɔ nidi ne kyɛfɑ koro. Wɔwɔ ɔdwene ne ɔhonim, nɑ ɛsɛ sɛ wobu wɔn ho wɔn ho sɛ ɔnuɑnom.",
    ],
    "bwu": [
        "Ka se-aa?",
        "Nalim nyini.",
        "Fi yue le boa?",
        "Ka boa ale nna?",
        "N baga a maari fu?",
        "Faa yaali k\u00e1 boa?",
        "Maa saalim, vongti mu.",
        "Kan namu.",
        "Cheng du!",
        "Jam de!",
    ],
}

LANG_NAMES = {
    "acd": "Akyode",
    "twi-asante": "Twi (Asante)",
    "twi-akuapem": "Twi (Akuapem)",
    "ada": "Dangme",
    "akp": "Siwu",
    "any": "Anyi",
    "avn": "Avatime",
    "bib": "Bimoba",
    "bim": "Bimoba",
    "biv": "Birifor",
    "bov": "Tuwuli",
    "bud": "Ntcham",
    "bwu": "Buli",
    "dag": "Dagbani",
    "dga": "Dagaare",
    "en": "English (Ghanaian)",
    "ewe": "Ewe",
    "fat": "Fante",
    "ffm": "Fulfulde (Maasina)",
    "gjn": "Gonja",
    "gur": "Farefare",
    "hau": "Hausa",
    "kbp": "Kabiyè",
    "kdh": "Koma",
    "kma": "Konni",
    "kus": "Kusaal",
    "lef": "Lelemi",
    "lip": "Sekpele",
    "maw": "Mampruli",
    "mzw": "Mo",
    "naw": "Nawuri",
    "ncu": "Chumburung",
    "nko": "Nkonya",
    "ntr": "Delo",
    "nzi": "Nzema",
    "sfw": "Sehwi",
    "sig": "Paasaal",
    "sil": "Sisaala",
    "snw": "Santrokofi",
    "tpm": "Tampulma",
    "vag": "Vagla",
    "xon": "Konkomba",
    "xsm": "Kasem",
}

LANG_CHOICES = sorted(LANGUAGE_EXAMPLES.keys(), key=lambda t: LANG_NAMES.get(t, t))

# Download prompt audio samples
PROMPT_DIR = "/tmp/prompt_audio"
PROMPT_MANIFEST = {}
def setup_prompt_audio():
    global PROMPT_MANIFEST
    os.makedirs(PROMPT_DIR, exist_ok=True)
    try:
        manifest_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="prompt_audio/manifest.json",
            repo_type="model",
        )
        with open(manifest_path) as f:
            PROMPT_MANIFEST = json.load(f)
        for tag, entries in PROMPT_MANIFEST.items():
            for e in entries:
                local_path = os.path.join(PROMPT_DIR, e["audio"])
                if not os.path.exists(local_path):
                    hf_hub_download(
                        repo_id=MODEL_ID,
                        filename=f"prompt_audio/{e['audio']}",
                        repo_type="model",
                        local_dir=PROMPT_DIR,
                        local_dir_use_symlinks=False,
                    )
    except Exception as ex:
        print(f"Prompt audio setup failed: {ex}")

setup_prompt_audio()

model = None

def load_model():
    global model
    if model is None:
        model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False, optimize=False)
    return model

def on_language_change(tag):
    examples = LANGUAGE_EXAMPLES.get(tag, [])
    if examples:
        return random.choice(examples)
    return ""

_CFG_SCALE = 2.0
_INF_STEPS = 10

def wav_to_b64(sr, wav_arr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wav_arr * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()

def build_survey_html(survey):
    if not survey:
        return '<p style="color:#6b7280; padding:1em 0;">No entries yet. Generate speech above to start rating.</p>'
    rows = []
    for e in survey:
        audio_tag = f'<audio controls src="data:audio/wav;base64,{e["audio_b64"]}" style="width:100%;"></audio>'
        prompt_label = f" + prompt" if e.get("prompt") and e["prompt"] != "None" else ""
        rid = e["id"]
        sel = e.get("rating", 0)
        rows.append(
            f"<tr><td><b>{e['lang_name']}</b><br><small>{e['lang']}{prompt_label}</small></td>"
            f"<td style='max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title='{e['text']}'>{e['text'][:50]}{'…' if len(e['text'])>50 else ''}</td>"
            f"<td style='width:180px;'>{audio_tag}</td>"
            f"<td style='white-space:nowrap; text-align:center;'>"
            f"<label class='thumb-radio{' active' if sel==1 else ''}'><input type='radio' name='rate_{rid}' value='1' onchange='rate(this)'{' checked' if sel==1 else ''}> 👍</label>"
            f"<label class='thumb-radio{' active' if sel==-1 else ''}'><input type='radio' name='rate_{rid}' value='-1' onchange='rate(this)'{' checked' if sel==-1 else ''}> 👎</label>"
            f"</td></tr>"
        )
    return f"""
<style>
.survey-table td, .survey-table th {{ padding:8px 10px; border-bottom:1px solid #e5e7eb; text-align:left; vertical-align:middle; }}
.survey-table th {{ background:#f3f4f6; font-weight:600; }}
.thumb-radio {{ cursor:pointer; font-size:20px; padding:4px 8px; user-select:none; white-space:nowrap; }}
.thumb-radio.active {{ background:#f0fdf4; border-radius:6px; font-weight:bold; }}
.thumb-radio input {{ margin-right:3px; cursor:pointer; }}
</style>
<table class="survey-table"><thead><tr><th>Language</th><th>Text</th><th>Audio</th><th>Rate</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<script>
function rate(el){{var id=el.name.split('_')[1];var v=parseInt(el.value);var r={{}};try{{r=JSON.parse(document.getElementById('ratings-data').value||'{{}}')}}catch(e){{}}r[id]=v;document.getElementById('ratings-data').value=JSON.stringify(r);}}
</script>"""

_NEXT_ID = 0

def synthesize(text, tag, prompt_choice, survey):
    global _NEXT_ID
    if not text.strip():
        return None, "Please enter some text.", build_survey_html(survey), survey
    final_text = f"<|lang:{tag}|> {text}"
    try:
        m = load_model()
        kwargs = dict(cfg_value=_CFG_SCALE, inference_timesteps=_INF_STEPS, normalize=False)
        prompt_wav = None
        if prompt_choice and prompt_choice != "None":
            for e in PROMPT_MANIFEST.get(tag, []):
                if e["audio"] == prompt_choice:
                    wav_path = os.path.join(PROMPT_DIR, e["audio"])
                    if os.path.exists(wav_path):
                        kwargs["prompt_wav_path"] = wav_path
                        kwargs["prompt_text"] = e.get("text", "")
                        prompt_wav = prompt_choice
                    break
        wav = m.generate(text=final_text, **kwargs)
        sr = m.tts_model.sample_rate
        duration = len(wav) / sr
        audio_b64 = wav_to_b64(sr, wav)
        entry = {
            "id": _NEXT_ID,
            "lang": tag,
            "lang_name": LANG_NAMES.get(tag, tag),
            "text": text,
            "prompt": prompt_wav,
            "duration": round(duration, 2),
            "audio_b64": audio_b64,
            "rating": 0,
        }
        _NEXT_ID += 1
        new_survey = survey + [entry]
        html = build_survey_html(new_survey)
        return (sr, wav), f"Generated {duration:.2f}s of speech. Added to survey.", html, new_survey
    except Exception as e:
        return None, f"Error: {e}", build_survey_html(survey), survey

def on_prompt_change(tag):
    prompts = PROMPT_MANIFEST.get(tag, [])
    choices = [("None (no prompt)", "None")]
    for i, e in enumerate(prompts):
        choices.append((f"Sample {i+1} ({e['duration']}s)", e["audio"]))
    return gr.Dropdown(choices=choices, value="None")

def submit_ratings(ratings_json, survey):
    if not survey:
        return "Nothing to submit.", build_survey_html([]), []
    try:
        ratings = json.loads(ratings_json) if ratings_json.strip() else {}
    except json.JSONDecodeError:
        ratings = {}
    submitted = 0
    for e in survey:
        rating_val = ratings.get(str(e["id"]), 0)
        if rating_val != 0:
            label = "thumbs_up" if rating_val > 0 else "thumbs_down"
            save_rating(e["lang"], e["text"], e.get("prompt"),
                        e["duration"], _CFG_SCALE, _INF_STEPS,
                        label, "")
            submitted += 1
    return f"Submitted {submitted} rating(s). Thank you!", build_survey_html([]), []

with gr.Blocks(title="Ghana TTS") as demo:
    survey_state = gr.State([])
    ratings_data = gr.Textbox(value="{}", visible=False, elem_id="ratings-data")
    gr.Markdown("# Ghana TTS\nText-to-speech for 41 Ghanaian languages.")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=400):
            tag = gr.Dropdown(
                choices=[(f"{LANG_NAMES.get(t, t)} ({t})", t) for t in LANG_CHOICES],
                value="acd", label="Language",
                info="Select a Ghanaian language"
            )
            prompt_audio = gr.Dropdown(
                choices=[("None (no prompt)", "None")], value="None",
                label="Prompt Audio (optional)",
                info="Select a voice reference sample to match speaking style"
            )
            text = gr.Textbox(label="Text to synthesize", placeholder="Enter text or click Change Example...", lines=4)
            with gr.Row():
                rand_btn = gr.Button("Change Example", variant="secondary", size="sm")
            btn = gr.Button("Generate Speech", variant="primary")
            audio = gr.Audio(label="Generated Speech", type="numpy")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=1, min_width=400):
            gr.Markdown("### 📋 Rating Survey\nListen, rate, then submit.")
            survey_html = gr.HTML(build_survey_html([]))
            with gr.Row():
                submit_btn = gr.Button("📤 Submit All Ratings", variant="primary", size="sm")
                clear_btn = gr.Button("🗑 Clear All", variant="secondary", size="sm")
            submit_status = gr.Textbox(label="", interactive=False, show_label=False)

    tag.change(on_language_change, inputs=[tag], outputs=[text])
    tag.change(on_prompt_change, inputs=[tag], outputs=[prompt_audio])
    rand_btn.click(on_language_change, inputs=[tag], outputs=[text])
    btn.click(synthesize, inputs=[text, tag, prompt_audio, survey_state], outputs=[audio, status, survey_html, survey_state])
    submit_btn.click(submit_ratings, inputs=[ratings_data, survey_state], outputs=[submit_status, survey_html, survey_state],
        js="(r, s) => { var ratings={}; document.querySelectorAll('.survey-table input[type=radio]:checked').forEach(function(el) { var id = el.name.split('_')[1]; ratings[id] = parseInt(el.value); }); return [JSON.stringify(ratings), s]; }")
    clear_btn.click(lambda: ("Cleared.", build_survey_html([]), []), outputs=[submit_status, survey_html, survey_state])
    demo.load(on_language_change, inputs=tag, outputs=[text])
    demo.load(on_prompt_change, inputs=tag, outputs=[prompt_audio])

if __name__ == "__main__":
    demo.launch()
