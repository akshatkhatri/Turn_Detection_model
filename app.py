from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from utils_shiprocket import prepare_data
import os
import json
import secrets
import threading
from datetime import datetime, timezone
from utils_shiprocket import AudioDataset, gru_collate_fn, AudioGRU
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

features = ("log-mel-spectrogram",)

DIR = "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/val_samples"
VAL_FILE_MAPPINGS_TRANSCRIPTS_PATH = (
    "/Users/akshat.khatri/Desktop/Turn_Detection_model/transcripts/merged_output_val.jsonl"
)
MODEL_DIR = (
    "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/"
    "kaggle/working/muril-endpoint-clf/checkpoint-3500"
)
INDEX_HTML_PATH = "/Users/akshat.khatri/Desktop/Turn_Detection_model/index.html"
VISITOR_LOG_PATH = "/Users/akshat.khatri/Desktop/Turn_Detection_model/visitors.jsonl"
FEEDBACK_LOG_PATH = "/Users/akshat.khatri/Desktop/Turn_Detection_model/feedback.jsonl"
VISITORS_PASSWORD = os.environ.get("VISITORS_PASSWORD")

security = HTTPBasic(
    description=(
        "<b>Username is optional.</b> You can leave it blank or enter anything; "
        "only the password is validated."
    )
)


def require_visitors_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not VISITORS_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VISITORS_PASSWORD is not configured",
        )

    correct_password = secrets.compare_digest(
        credentials.password,
        VISITORS_PASSWORD,
    )

    if not correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


ALPHA = 0.5
BETA = 0.5
THRESHOLD = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Starting building Application.. This may take a while.. before server starts...")

norm_stats = torch.load(
    "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/model/norm_stats.pt",
    map_location="cpu",
)
train_mean = norm_stats["train_mean"]
train_std = norm_stats["train_std"]

model = AudioGRU(
    128,
    train_mean,
    train_std,
    hidden_size=64,
    conv_channels=64,
)
checkpoint = torch.load(
    "/Users/akshat.khatri/PycharmProjects/Shiprocket_final/model/interrupted_checkpoint.pt",
    map_location="cpu",
)
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
        enc = tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        logits = text_model(**enc).logits
        p = torch.softmax(logits, dim=-1)
    return p[0, 1].item()


def precompute_predictions(loader, transcript_lookup):
    results = []
    ids = val_df_filtered["id"]
    idx = 0

    with torch.no_grad():
        for X, lengths, y in tqdm(loader, desc="Computing eval points .."):
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

                results.append(
                    {
                        "id": _id,
                        "audio_prob": ap,
                        "text_prob": text_prob,
                        "combined_prob": combined,
                        "prediction": int(combined >= THRESHOLD),
                        "label": true,
                    }
                )
                idx += 1

    return results


transcript_lookup = load_transcript_lookup(VAL_FILE_MAPPINGS_TRANSCRIPTS_PATH)
PREDICTIONS = precompute_predictions(val_loader, transcript_lookup)
PREDICTIONS_BY_ID = {r["id"]: r for r in PREDICTIONS}
LABEL_BY_ID = {r["id"]: r["label"] for r in PREDICTIONS}

print("Application Build Done.. you can use endpoints Now...")


# -----------------------------------------------------------------------------
# Visitor + feedback tracking
# -----------------------------------------------------------------------------
#
# We intentionally do not use cookies here.
#
# When a visitor first arrives with ?visitor_name=..., the server creates a
# random opaque visitor_id and injects only that id into the HTML page. The
# feedback form later sends visitor_id + feedback. The server resolves the name
# from its own visitor index.
#
# Existing visitor log rows that do not have visitor_id are still readable from
# /visitors, but they cannot be used for new feedback association.

VISITORS_BY_ID = {}
log_lock = threading.Lock()


def load_visitor_index():
    if not os.path.exists(VISITOR_LOG_PATH):
        return

    try:
        with open(VISITOR_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                visitor_id = row.get("visitor_id")
                name = row.get("name")

                if visitor_id and name:
                    VISITORS_BY_ID[visitor_id] = {
                        "name": name,
                        "reason": row.get("reason"),
                    }
    except Exception as e:
        print(f"Failed to load visitor index: {e}")


def log_visitor(request: Request, visitor_id: str, name: str, reason: str | None):
    entry = {
        "visitor_id": visitor_id,
        "name": name,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }

    VISITORS_BY_ID[visitor_id] = {
        "name": name,
        "reason": reason,
    }

    try:
        with log_lock:
            with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to log visitor: {e}")


def get_visitor_by_id(visitor_id: str):
    visitor = VISITORS_BY_ID.get(visitor_id)
    if visitor is not None:
        return visitor

    # Fallback to disk so the lookup still works after a restart or when
    # another worker handled the original page request.
    if not os.path.exists(VISITOR_LOG_PATH):
        return None

    try:
        with open(VISITOR_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if row.get("visitor_id") == visitor_id and row.get("name"):
                    visitor = {
                        "name": row["name"],
                        "reason": row.get("reason"),
                    }
                    VISITORS_BY_ID[visitor_id] = visitor
                    return visitor
    except Exception as e:
        print(f"Failed to look up visitor: {e}")

    return None


def log_feedback(
    request: Request,
    visitor_id: str,
    visitor_name: str,
    feedback: str,
):
    entry = {
        "visitor_id": visitor_id,
        "name": visitor_name,
        "feedback": feedback,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }

    with log_lock:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


load_visitor_index()


class FeedbackPayload(BaseModel):
    visitor_id: str
    feedback: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    visitor_name = request.query_params.get("visitor_name")
    visitor_reason = request.query_params.get("visitor_reason")

    # If this request came from your existing name-entry flow, create a fresh,
    # unguessable id for this page session and remember the mapping server-side.
    if visitor_name and visitor_name.strip():
        visitor_name = visitor_name.strip()
        visitor_id = secrets.token_urlsafe(24)
        log_visitor(
            request=request,
            visitor_id=visitor_id,
            name=visitor_name,
            reason=visitor_reason,
        )
    else:
        # No name was supplied on this request, so we deliberately do not guess.
        visitor_id = ""

    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # token_urlsafe() only produces URL-safe ASCII, so it is safe to place in
    # this quoted meta attribute. An empty value means feedback cannot be tied
    # to a known visitor on this page load.
    html = html.replace("__VISITOR_ID__", visitor_id)

    return HTMLResponse(content=html)


@app.get("/visitors")
def get_visitors(username: str = Depends(require_visitors_auth)):
    if not os.path.exists(VISITOR_LOG_PATH):
        return []

    visitors = []
    with open(VISITOR_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                visitors.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return visitors


@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload, request: Request):
    visitor_id = payload.visitor_id.strip()
    feedback = payload.feedback.strip()

    if not visitor_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This page is not associated with a known visitor.",
        )

    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback cannot be empty.",
        )

    if len(feedback) > 5000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback must be 5000 characters or fewer.",
        )

    visitor = get_visitor_by_id(visitor_id)

    if visitor is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown or expired visitor id.",
        )

    try:
        log_feedback(
            request=request,
            visitor_id=visitor_id,
            visitor_name=visitor["name"],
            feedback=feedback,
        )
    except Exception as e:
        print(f"Failed to save feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save feedback.",
        )

    return {"success": True}


@app.get("/audio/{sample_id}")
def get_audio(sample_id: str):
    path = audio_paths_by_id.get(sample_id)
    if path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="id not found",
        )
    return FileResponse(path, media_type="audio/wav")


@app.get("/predictions")
def get_all_predictions():
    return PREDICTIONS


@app.get("/predictions/{sample_id}")
def get_prediction(sample_id: str):
    prediction = PREDICTIONS_BY_ID.get(sample_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="id not found",
        )
    return prediction