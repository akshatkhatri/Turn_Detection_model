import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from datasets import Audio
import librosa
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from typing import Any
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

def stratified_split(dataset, stratify_col: str, test_size: float, seed: int = 42):
    """Shuffle within each stratum, then split into (train, test)."""
    rng = random.Random(seed)
    groups = {}
    for i, val in enumerate(dataset[stratify_col]):
        groups.setdefault(val, []).append(i)

    train_idx, test_idx = [], []
    for val, idx_list in groups.items():
        idx_list = idx_list[:]  # copy
        rng.shuffle(idx_list)
        n = len(idx_list)
        n_test = max(1, int(n * test_size))
        n_test = min(n_test, n - 1) if n > 1 else 0
        test_idx.extend(idx_list[:n_test])
        train_idx.extend(idx_list[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return dataset.select(train_idx), dataset.select(test_idx)

def evaluate_binary_classifier(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


from datasets import load_dataset, Audio


def prepare_data(path='pipecat-ai/smart-turn-data-v3.2-train', languages=['eng', 'hin'], train_examples=10000, seed=42):
    df = load_dataset(path)['train']
    df = df.cast_column("audio", Audio(decode=False))

    if languages:
        df = df.filter(lambda example: example['language'] in languages)

    if train_examples is not None:
        train_examples = min(train_examples, len(df))
        df = df.shuffle(seed=seed).select(range(train_examples))

    train_df, temp_df = stratified_split(df, "language", test_size=0.2, seed=seed)
    val_df, test_df = stratified_split(temp_df, "language", test_size=0.5, seed=seed)

    return train_df, val_df, test_df

def give_item_y_and_sr(item, sr = 22050):
    if item.get('bytes'):
        y, sr = librosa.load(io.BytesIO(item['bytes']), sr=sr)
        return y, sr

    audio_info = item['audio']
    y, sr = librosa.load(io.BytesIO(audio_info['bytes']), sr=sr)
    return y,sr


def transcribe_item(item,stt_model, beam_size = 5):
    y, sr = give_item_y_and_sr(item, sr=16000)
    true_label = item['endpoint_bool']

    segments, info = stt_model.transcribe(
        y,
        task="transcribe",
        beam_size=beam_size
    )

    text = " ".join(segment.text.strip() for segment in segments)

    return {
        "text": text,
        "language": info.language,
        "language_probability": info.language_probability,
        "true_label": true_label,
    }



class FeatureStore:
    def __init__(
        self,
        audios : list[Any],audio_ids : list[str], sample_rate: int = 22050, hop_length: int = 512,
        n_fft: int = 2048, n_mels: int = 128, n_mfcc: int = 13,
        features: tuple = (
            "log-mel-spectrogram", "mfcc", 'zcr',
            "mfcc-delta", "mfcc-delta2", "pyin","yin")
        ):

        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sample_rate

        if len(audios) == 0:
            raise AttributeError('No audio files provided... , provide at least one..')
        if len(audios) != len(audio_ids):
            raise AttributeError('Number of audio files dont match their ids provided.')

        self.audios = audios
        self.audio_ids = audio_ids

        self.features = features
        self.feature_store = []

    def extract_all(self , verbose = True) -> list[Any]:
        self.feature_store = []
        for a, audio_id in tqdm(zip(self.audios,self.audio_ids), desc = "Extracting Acoustic features", total = len(self.audio_ids), disable = not verbose):
            audio , _ = give_item_y_and_sr(a)
            aux_features_dict = dict()
            # aux_features_dict["audio_id"] = audio_id    # can enable but commented for easier implementation
            # print('Checking and finding MFCC ..')
            need_mfcc = any(
                f in self.features
                for f in ("mfcc", "mfcc-delta", "mfcc-delta2")
            )
            base_mfcc = self._get_mfcc(audio) if need_mfcc else None

            for feature in self.features:
                if feature == "log-mel-spectrogram":
                    # print('Checking and finding log-mel-spectogram..')
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio,
                        sr=self.sr,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        n_mels=self.n_mels,
                    )
                    aux_features_dict["log-mel-spectrogram"] = librosa.power_to_db(
                        mel_spec, ref=np.max
                    )

                elif feature == "mfcc":
                    # print('Checking and finding mfcc..')
                    aux_features_dict["mfcc"] = base_mfcc

                elif feature == "mfcc-delta":
                    # print('Checking and finding mfcc delta..')
                    aux_features_dict["mfcc-delta"] = librosa.feature.delta(base_mfcc)

                elif feature == "mfcc-delta2":
                    # print('Checking and finding mfcc-delta-delta..')
                    aux_features_dict["mfcc-delta2"] = librosa.feature.delta(
                        base_mfcc, order=2
                    )

                elif feature == "pyin":
                    # print('Checking and finding pyin..')
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        y= audio,
                        fmin=librosa.note_to_hz("C1"),
                        fmax=librosa.note_to_hz("C7"),
                        sr=self.sr,
                        frame_length=self.n_fft,
                        hop_length=self.hop_length,
                    )
                    aux_features_dict["f0"] = f0.reshape(1, -1)
                    aux_features_dict["voiced_flag"] = voiced_flag.reshape(1, -1)
                    aux_features_dict["voiced_probs"] = voiced_probs.reshape(1, -1)

                elif feature == "yin":
                    # print('Checking and finding yin..')
                    f0 = librosa.yin(
                        y=audio,
                        fmin=librosa.note_to_hz("C1"),
                        fmax=librosa.note_to_hz("C7"),
                        sr=self.sr,
                        frame_length=self.n_fft,
                        hop_length=self.hop_length,
                    )
                    f0= np.nan_to_num(f0, nan=0.0)
                    aux_features_dict["yin_f0"] = f0.reshape(1, -1)

                elif feature == "zcr":
                    # print('Checking and finding zcr..')
                    aux_features_dict["zcr"] = librosa.feature.zero_crossing_rate(
                        y=audio,
                        frame_length=self.n_fft,
                        hop_length=self.hop_length,
                    )
            self.feature_store.append(aux_features_dict)

        return self.feature_store

    def extract_one(self, item) -> dict:
        audio, _ = give_item_y_and_sr(item)
        aux_features_dict = {}

        need_mfcc = any(f in self.features for f in ("mfcc", "mfcc-delta", "mfcc-delta2"))
        base_mfcc = self._get_mfcc(audio) if need_mfcc else None

        for feature in self.features:
            if feature == "log-mel-spectrogram":
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
                aux_features_dict["log-mel-spectrogram"] = librosa.power_to_db(mel_spec, ref=np.max)

            elif feature == "mfcc":
                aux_features_dict["mfcc"] = base_mfcc

            elif feature == "mfcc-delta":
                aux_features_dict["mfcc-delta"] = librosa.feature.delta(base_mfcc)

            elif feature == "mfcc-delta2":
                aux_features_dict["mfcc-delta2"] = librosa.feature.delta(base_mfcc, order=2)

            elif feature == "pyin":
                f0, voiced_flag, voiced_probs = librosa.pyin(y=audio, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C7"), sr=self.sr, frame_length=self.n_fft, hop_length=self.hop_length)

                aux_features_dict["f0"] = np.nan_to_num(f0, nan=0.0).reshape(1, -1)
                aux_features_dict["voiced_flag"] = voiced_flag.reshape(1, -1)
                aux_features_dict["voiced_probs"] = np.nan_to_num(voiced_probs, nan=0.0).reshape(1, -1)

            elif feature == "yin":
                f0 = librosa.yin(y=audio, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C7"), sr=self.sr, frame_length=self.n_fft, hop_length=self.hop_length)
                aux_features_dict["yin_f0"] = np.nan_to_num(f0, nan=0.0).reshape(1, -1)

            elif feature == "zcr":
                aux_features_dict["zcr"] = librosa.feature.zero_crossing_rate(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)

        return aux_features_dict

    def _get_mfcc(self,audio) -> np.ndarray:
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc,
        )

    def vertical_stack_features_summary_stats(self, verbose = True):
        features_per_item = self.extract_all(verbose)
        summary_stats_features = []
        for features_of_item in tqdm(features_per_item, disable= not verbose):
            stacked_features = np.vstack(list(features_of_item.values()))
            stacked_features = np.nan_to_num(
            stacked_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
            summary_stats_features.append(np.array([
                np.mean(stacked_features, axis=1),
                np.std(stacked_features,axis = 1),
                np.min(stacked_features,axis = 1),
                np.max(stacked_features,axis = 1),
               np.polyfit(np.arange(stacked_features.shape[1]), stacked_features.T, 1)[0], # trend direction
            ]))

        summary_stats_features = np.array(summary_stats_features)
        summary_stats_features = summary_stats_features.reshape(summary_stats_features.shape[0], -1)

        # print(summary_stats_features.shape)
        return summary_stats_features

    def get_temporal_feature(self, item):
        features_of_item = self.extract_one(item)

        if not features_of_item:
            raise ValueError(f"No features extracted. Requested: {self.features}")

        if len(features_of_item) == 1:
            stacked_features = next(iter(features_of_item.values())).T
        else:
            stacked_features = np.vstack(list(features_of_item.values())).T

        return np.nan_to_num(stacked_features, nan=0.0, posinf=0.0, neginf=0.0)

class AudioDataset(Dataset):
    def __init__(self,df,features_lst):
        self.df = df
        self.features = features_lst

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df[idx]
        fs = FeatureStore([item['audio']],[item['id']], features=self.features)
        X = fs.get_temporal_feature(item)
        Y = item['endpoint_bool']

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def gru_collate_fn(batch):
    X, y = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in X], dtype=torch.long) # lenght is neccesary so model ignores it
    X = pad_sequence(X, batch_first=True, padding_value=0.0)
    y = torch.stack(y)

    return X, lengths, y


class AudioGRU(nn.Module):
    def __init__(self, input_size, mean, std, hidden_size=64, conv_channels=64):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, 5, padding=2), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1), nn.ReLU()
        )
        self.gru = nn.GRU(conv_channels, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths.to(x.device)[:, None]
        x = (x - self.mean) / self.std
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(1)


import io
import os
import soundfile as sf
from tqdm import tqdm


def save_val_samples(val_df, out_dir="val_samples", n_samples=1000):
    """Saves audio samples from a validation Dataset/DataFrame to disk.

    Parameters:
    - val_df: Dataset or list-like collection of audio examples.
    - out_dir (str): Directory where the output WAV files will be saved.
    - n_samples (int): Maximum number of requested samples to process.
    """
    os.makedirs(out_dir, exist_ok=True)
    total_val = len(val_df)

    # Cap n_samples to the length of val_df if it exceeds it
    if n_samples > total_val:
        print(
            f"Requested {n_samples} samples, but dataset only has {total_val}. Capping to {total_val}."
        )
        n_samples = total_val

    for i in tqdm(range(n_samples), desc="Saving audio samples"):
        example = val_df[i]
        audio = example["audio"]
        label = int(example["endpoint_bool"])

        # Use original filename (without extension) if available, else fallback to index
        if audio.get("path"):
            base_name = os.path.splitext(os.path.basename(audio["path"]))[0]
        else:
            base_name = f"sample_{i}"

        out_path = os.path.join(out_dir, f"{base_name}_label_{label}.wav")

        data, samplerate = sf.read(io.BytesIO(audio["bytes"]))
        sf.write(out_path, data, samplerate)

    print(f"\nDone — {n_samples} files written to ./{out_dir}")
