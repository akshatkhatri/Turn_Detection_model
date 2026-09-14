# Bilingual Turn Detection for Voice Assistants

An experimental end-of-turn detection system for English and Hindi speech. The project frames turn detection as binary classification: given an audio utterance, predict whether the speaker has completed their turn (`endpoint_bool = 1`) or is likely to continue (`endpoint_bool = 0`).

The central idea was to build the solution progressively—from a simple silence heuristic, through classical acoustic models and temporal deep learning, to a semantic model and finally a multimodal ensemble. Each stage tests what additional information is useful for deciding whether an utterance is complete.

## Results

All results below are from the validation split. F1-score is the primary comparison metric because it balances precision and recall for the positive endpoint class.

| Approach | Input signal | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| RMS silence heuristic | Trailing silence | 0.508 | 0.551 | 0.07 | **0.13** |
| Random Forest | Summarized acoustic features | 0.6468 | 0.6351 | 0.6492 | **0.6421** |
| XGBoost | Summarized acoustic features | 0.6691 | 0.6613 | 0.6597 | **0.6605** |
| CNN + GRU | Log-mel time series | 0.8410 | 0.8048 | 0.8688 | **0.8356** |
| MuRIL text classifier | Whisper transcript | 0.9375 | — | — | **0.9378** |
| **Hybrid soft-voting ensemble** | **Acoustic + semantic** | **0.9576** | **0.9444** | **0.9702** | **0.9571** |

The experiments show a consistent improvement as the representation becomes richer. XGBoost improves substantially over a pure silence rule; preserving the acoustic sequence with a CNN–GRU raises F1 to 0.8356; semantic modeling reaches 0.9378; and combining both signals produces the best validation F1 of **0.9571 (~0.96)**.

## Why combine acoustics and semantics?

Turn completion is not the same as silence detection. A pause may indicate the end of an utterance, but it may also be a hesitation, interruption, or brief gap inside an incomplete sentence.

The two final branches capture complementary evidence:

- The **acoustic branch** learns patterns in timing, energy, and the evolution of the speech signal.
- The **semantic branch** estimates whether the transcribed utterance is linguistically complete. For example, “find me a place to dine with my…” is less likely to be complete than “find me a place to dine with my friends.”

The final system combines their endpoint probabilities using soft voting:

```text
P(endpoint) = alpha * P_audio(endpoint) + beta * P_text(endpoint)

alpha = 0.5
beta  = 0.5
decision threshold = 0.5
```

This lets a confident acoustic prediction support an uncertain semantic prediction, and vice versa.

## Modeling approach

### 1. RMS-energy baseline

The baseline divides each waveform into short frames and measures RMS energy near the end of the clip. If at least 0.7 seconds of continuous trailing audio falls below the silence threshold, the sample is classified as an endpoint.

This establishes an interpretable lower bound. Its F1-score of 0.13 also demonstrates why conversational turn detection needs more than a fixed silence rule.

### 2. Classical ML on acoustic features

The next stage extracts signal-level features with `librosa`:

- 128-bin log-mel spectrogram
- 13 MFCC coefficients
- Zero-crossing rate

Because classical estimators expect a fixed-length vector while audio clips have different durations, every feature channel is summarized across time using five statistics: mean, standard deviation, minimum, maximum, and temporal trend. This converts 142 acoustic channels into a **710-dimensional vector** per utterance.

Random Forest and XGBoost were trained on these vectors. XGBoost was the strongest classical model with an F1-score of 0.6605. An online kernel-approximation classifier was also explored as a compute-aware path for incremental training while feature extraction was still being iterated.

### 3. Temporal acoustic modeling with CNN + GRU

Summary statistics are efficient, but they discard the order in which acoustic events occur. The temporal model therefore operates directly on variable-length log-mel sequences:

```text
Waveform
  -> 128-bin log-mel spectrogram
  -> feature normalization
  -> two Conv1D + ReLU blocks
  -> packed GRU sequence
  -> final hidden state
  -> linear binary classifier
```

The convolutional layers learn local spectro-temporal patterns, while the GRU retains longer-range context. Padding masks and packed sequences prevent padded frames from influencing the prediction. This approach reaches an F1-score of 0.8356 using only the log-mel representation.

### 4. Semantic endpoint modeling

The semantic path converts speech into text with **Whisper Large V3 Turbo**, selected for multilingual English, Hindi, and code-mixed speech support. The resulting transcripts are used to fine-tune **`google/muril-base-cased`** as a binary sequence classifier.

MuRIL is well aligned with this dataset because it was pretrained for English and multiple Indian languages. It can learn lexical and grammatical completion cues that are not available from signal energy alone. The model reaches **0.9378 F1** on the validation split at checkpoint 3500.

### 5. Acoustic–semantic fusion

The final system performs probability-level fusion between the CNN–GRU and MuRIL classifiers. With equal weights and a 0.5 decision threshold, it achieves the best overall result:

- Accuracy: **0.9576**
- Precision: **0.9444**
- Recall: **0.9702**
- F1-score: **0.9571**

The high recall is particularly useful in an endpoint detector, where missed turn completions can make a voice interface feel unresponsive.

## Data preparation

The source is the Hugging Face dataset **`pipecat-ai/smart-turn-data-v3.2-train`**. The pipeline:

1. Filters the original multilingual corpus to English (`eng`) and Hindi (`hin`).
2. Casts audio to a deferred-decoding representation to avoid loading every waveform at once.
3. Creates an 80/10/10 train, validation, and test split.
4. Stratifies each split by language so Hindi remains proportionally represented despite the English-heavy distribution.
5. Keeps the test split separate from model iteration; the reported experiment comparisons use validation data.

The filtered working dataset contains 77,808 samples:

| Split | Samples |
|---|---:|
| Train | 62,247 |
| Validation | 7,781 |
| Test | 7,780 |

The shared preparation code uses a fixed seed (`42`) for reproducible splits.

## Demo application

The repository includes a FastAPI evaluation interface for inspecting validation samples. Predictions are precomputed when the application starts, and the UI lets a reviewer:

- Play an audio sample.
- Inspect acoustic, semantic, and combined probabilities.
- Compare the model prediction with the ground-truth label.
- Filter samples and review agreements or mismatches.

Available endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Evaluation dashboard |
| `GET /audio/{sample_id}` | WAV audio for one sample |
| `GET /predictions` | All precomputed predictions |
| `GET /predictions/{sample_id}` | Prediction details for one sample |

## Repository structure

```text
.
├── app.py                                  # FastAPI inference and demo API
├── index.html                              # Interactive evaluation dashboard
├── utils_shiprocket.py                     # Data, feature, model, and evaluation utilities
├── model/
│   ├── acoustic_features_model.ipynb       # Classical acoustic experiments
│   ├── time_series_dnn_model.ipynb         # CNN–GRU training and evaluation
│   ├── semantic_text_models.ipynb           # MuRIL inference workflow
│   ├── hybrid_models.ipynb                  # Acoustic + semantic ensemble
│   ├── wav2vec2.ipynb                       # End-to-end speech-model experiments
│   ├── xgboost_model.json                   # Trained XGBoost artifact
│   └── non_linear_classic_ml_classifier.joblib
├── kaggle/working/muril-endpoint-clf/
│   └── checkpoint-3500/                    # Fine-tuned MuRIL checkpoint and logs
├── transcripts/
│   ├── merged_output_train.jsonl           # Training transcripts
│   └── merged_output_val.jsonl             # Validation transcripts
├── val_samples/                            # Audio used by the evaluation UI
├── full_sample.ipynb                       # Validation sample export workflow
├── requirements.txt
└── pyproject.toml
```

The notebooks are organized by experiment family so the full progression—from feature engineering to multimodal fusion—can be reviewed independently.

## Running the project

### Requirements

- Python 3.11+
- Sufficient memory to load the CNN–GRU and MuRIL models together
- Internet access on the first dataset load
- CUDA is optional; inference automatically uses CUDA when available and otherwise falls back to CPU

### Installation

```bash
git clone <repository-url>
cd Shiprocket_final

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure local assets

The demo consumes the checked-in/local model checkpoint, normalization statistics, validation WAV files, and validation transcripts. Set the path constants near the top of `app.py` to the corresponding locations on your machine:

```python
DIR = "<project-root>/val_samples"
VAL_FILE_MAPPINGS_TRANSCRIPTS_PATH = "<project-root>/transcripts/merged_output_val.jsonl"
MODEL_DIR = "<project-root>/kaggle/working/muril-endpoint-clf/checkpoint-3500"
```

Also update the two `torch.load(...)` paths for `model/norm_stats.pt` and `model/interrupted_checkpoint.pt`.

### Start the API and dashboard

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. Startup includes dataset preparation and prediction precomputation, so the first launch can take several minutes depending on hardware.

## Reproducing the experiments

Run the notebooks in the following order to follow the research path:

1. `model/acoustic_features_model.ipynb`
2. `model/time_series_dnn_model.ipynb`
3. `model/semantic_text_models.ipynb`
4. `model/hybrid_models.ipynb`

`model/wav2vec2.ipynb` contains an additional end-to-end representation-learning experiment. Transcription and transformer training were carried out with GPU acceleration; saved transcripts and checkpoints allow the downstream semantic and hybrid stages to be inspected without repeating the entire preprocessing workload.

## Key design decisions

- **Language-stratified splitting:** protects bilingual evaluation coverage in an imbalanced corpus.
- **F1-first evaluation:** balances false endpoint triggers against missed endpoints.
- **Progressive baselines:** makes the value of each modeling choice measurable.
- **Sequence preservation:** the CNN–GRU avoids collapsing all temporal behavior into global statistics.
- **Multilingual semantics:** Whisper and MuRIL support English, Hindi, and code-mixed conversational input.
- **Late fusion:** probability-level ensembling keeps both specialist branches interpretable and independently testable.

## Author

**Akshat Khatri**  
Turn Detection assignment submission for Shiprocket.
