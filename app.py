from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from utils_shiprocket import prepare_data
import os
import json
from utils_shiprocket import AudioDataset, gru_collate_fn, AudioGRU
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

app = FastAPI()
features = ("log-mel-spectrogram",)
DIR = "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/val_samples"
VAL_FILE_MAPPINGS_TRANSCRIPTS_PATH = '/Users/akshat.khatri/PycharmProjects/Shiprocket_final/transcripts/merged_output_val.jsonl'
MODEL_DIR = "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/kaggle/working/muril-endpoint-clf/checkpoint-3500"
ALPHA = 0.5
BETA = 0.5
THRESHOLD = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

print('Starting building Application.. This may take a while.. before server starts...')

norm_stats = torch.load("/Users/akshat.khatri/PycharmProjects/Shiprocket_final/model/norm_stats.pt", map_location="cpu")
train_mean = norm_stats["train_mean"]
train_std = norm_stats["train_std"]

model = AudioGRU(128, train_mean, train_std, hidden_size=64, conv_channels=64)
checkpoint = torch.load("/Users/akshat.khatri/PycharmProjects/Shiprocket_final/model/interrupted_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
audio_model = model
audio_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

items = []
audio_paths_by_id = {}
for f in os.listdir(DIR):
    if f.endswith(".wav"):
        _id, rest = f.split("_label_")
        label = int(rest.replace(".wav", ""))
        items.append((_id, label))
        audio_paths_by_id[_id] = os.path.join(DIR, f)

_, val_df, _ = prepare_data(train_examples=None)

ids_wanted = {i for i, _ in items}
val_df_filtered = val_df.filter(lambda row: row["id"] in ids_wanted)

val_dataset = AudioDataset(val_df_filtered, features)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=gru_collate_fn,
)


def load_transcript_lookup(path):
    lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lookup[row["id"]] = row["text"]
    return lookup

def get_text_prob(text):
    with torch.no_grad():
        enc = tokenizer([text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        logits = text_model(**enc).logits
        p = torch.softmax(logits, dim=-1)
    return p[0, 1].item()


def precompute_predictions(loader, transcript_lookup):
    results = []
    ids = val_df_filtered["id"]
    idx = 0
    with torch.no_grad():
        for X, lengths, y in tqdm(loader, desc = 'Computing eval points ..'):
            logits = audio_model(X, lengths)
            audio_probs = torch.sigmoid(logits)

            for ap, true in zip(audio_probs.cpu().tolist(), y.cpu().tolist()):
                _id = ids[idx]
                text = transcript_lookup.get(_id)
                text_prob = get_text_prob(text) if text is not None else None

                if text_prob is not None:
                    combined = ALPHA * ap + BETA * text_prob
                else:
                    combined = ap

                results.append({
                    "id": _id,
                    "audio_prob": ap,
                    "text_prob": text_prob,
                    "combined_prob": combined,
                    "prediction": int(combined >= THRESHOLD),
                    "label": true,
                })
                idx += 1
    return results


transcript_lookup = load_transcript_lookup(VAL_FILE_MAPPINGS_TRANSCRIPTS_PATH)
PREDICTIONS = precompute_predictions(val_loader, transcript_lookup)
PREDICTIONS_BY_ID = {r["id"]: r for r in PREDICTIONS}
LABEL_BY_ID = {r["id"]: r["label"] for r in PREDICTIONS}

print('Application Build Done.. you can use endpoints Now...')

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/index.html")) as f:
        return f.read()


@app.get("/audio/{sample_id}")
def get_audio(sample_id: str):
    path = audio_paths_by_id.get(sample_id)
    if path is None:
        return {"error": "id not found"}
    return FileResponse(path, media_type="audio/wav")


@app.get("/predictions")
def get_all_predictions():
    return PREDICTIONS


@app.get("/predictions/{sample_id}")
def get_prediction(sample_id: str):
    return PREDICTIONS_BY_ID.get(sample_id, {"error": "id not found"})