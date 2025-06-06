#!/usr/bin/env python3
import os
import json
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from speechbrain.pretrained import EncoderClassifier  # Para ECAPA-TDNN

import matplotlib.pyplot as plt

"""
tts_lora/
├── DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py  # versión modificada con spk_proj
├── tts_embeddings.py
├── hparams.json      # JSON con tus hiperparámetros para Tacotron2
├── checkpoints/      # carpeta vacía o con checkpoints previos
└── data/
    └── LJSpeech/     # dataset de LJSpeech con metadata.csv y wavs/
    └── my_voice/     # tu dataset personal (opcional, para inferencia)
        └── wavs/
            └── Voz1.wav

Uso:

python tts_embeddings.py train \
  --data_dir data/LJSpeech \
  --hparams_path hparams.json \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --num_workers 4 \
  --checkpoints_dir checkpoints \
  --spk_emb_size 192 \
  --vis_dir visualizations \
  --mlflow-host http://admin:mlflow_password@mlflow.100.106.150.12.nip.io/

python tts_embeddings.py infer \
  --hparams_path hparams.json \
  --checkpoint_path checkpoints/best_tts_zero_shot.pth \
  --reference_wav data/my_voice/wavs/Voz1.wav \
  --text "Hello, this is a zero-shot voice cloning test." \
  --output_dir samples_cloned \
  --spk_emb_size 192 \
  --sample_rate 22050
"""

# --------------------------------------------------------------------------------
# 0) UTILIDADES: cargar hparams y extraer speaker embedding
# --------------------------------------------------------------------------------

def load_hparams(hparams_path: str) -> SimpleNamespace:
    """
    Carga hiperparámetros desde un JSON. Debe contener al menos los campos de Tacotron2:
      - mask_padding
      - n_mel_channels
      - symbols_embedding_dim
      - encoder_kernel_size
      - encoder_n_convolutions
      - encoder_embedding_dim
      - attention_rnn_dim
      - attention_dim
      - attention_location_n_filters
      - attention_location_kernel_size
      - n_frames_per_step
      - decoder_rnn_dim
      - prenet_dim
      - max_decoder_steps
      - gate_threshold
      - p_attention_dropout
      - p_decoder_dropout
      - postnet_embedding_dim
      - postnet_kernel_size
      - postnet_n_convolutions
      - decoder_no_early_stopping
      - symbols  (lista de strings)
    Adicionalmente, puede incluir (opcional):
      - sample_rate (predeterminado: 22050)
      - n_fft       (predeterminado: 1024)
      - hop_length  (predeterminado: 256)
      - win_length  (predeterminado: 1024)
    """
    with open(hparams_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimpleNamespace(**data)


def get_spk_embedding(wav_path: str, device: torch.device) -> torch.Tensor:
    """
    Extrae el speaker embedding ECAPA-TDNN desde un archivo WAV.
    - Re-muestrea a 16 kHz mono.
    - Retorna un Tensor [192] en CPU.
    """
    # ECAPA se mantiene en CPU para evitar mismatch de tipos
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    ecapa.eval()

    wav, sr = torchaudio.load(wav_path)  # [1, T] o [2, T]
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    wav = wav.squeeze(0).unsqueeze(0)  # [1, T] en CPU

    with torch.no_grad():
        emb = ecapa.encode_batch(wav)  # [1, 192]
    emb = emb.squeeze(0).cpu()  # [192]
    return emb


# --------------------------------------------------------------------------------
# 1) CLASE DE DATASET PARA LJSPEECH
# --------------------------------------------------------------------------------

class LJSpeechDataset(Dataset):
    """
    Dataset para LJSpeech (u otro con metadata.csv en formato fileid|transcript|normalized_transcript).
    Archivos .wav en subcarpeta 'wavs/'. Retorna:
      - text (string)
      - mel (Tensor [n_mels, T_frames])
      - spk_emb (Tensor [192]) de ECAPA-TDNN
    """
    def __init__(self, data_dir, hparams, ecapa_model, device):
        super().__init__()
        self.data_dir = data_dir
        self.wav_dir = os.path.join(data_dir, "wavs")
        metadata_path = os.path.join(data_dir, "metadata.csv")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"No se encuentra {metadata_path}")
        lines = open(metadata_path, "r", encoding="utf-8").read().strip().split("\n")
        self.entries = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            fileid, _, norm = parts
            wav_path = os.path.join(self.wav_dir, f"{fileid}.wav")
            if os.path.isfile(wav_path):
                self.entries.append({"wav_path": wav_path, "text": norm.lower()})

        # Parámetros de audio/mel: usar defaults si no están en hparams
        self.sample_rate = getattr(hparams, "sample_rate", 22050)
        self.n_mels = hparams.n_mel_channels
        self.n_fft = getattr(hparams, "n_fft", 1024)
        self.hop_length = getattr(hparams, "hop_length", 256)
        self.win_length = getattr(hparams, "win_length", 1024)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
        )
        self.ecapa = ecapa_model  # ECAPA en CPU
        self.device = device

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav_path = entry["wav_path"]
        text = entry["text"]

        # 1) Cargar audio y calcular mel-spectrogram (en CPU)
        wav, sr = torchaudio.load(wav_path)
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        mel_spec = self.mel_transform(wav).squeeze(0)  # [n_mels, T_frames]
        # NORMALIZE: always between 0 and 1 (or you could use a fixed max like 32_768, but this works for most cases)
        mel_spec = mel_spec / (mel_spec.max() + 1e-9)

        # 2) Obtener embedding de hablante (ECAPA en CPU)
        wav_ecapa, sr2 = torchaudio.load(wav_path)
        if wav_ecapa.dim() == 2 and wav_ecapa.shape[0] > 1:
            wav_ecapa = torch.mean(wav_ecapa, dim=0, keepdim=True)
        if sr2 != 16000:
            wav_ecapa = torchaudio.functional.resample(wav_ecapa, orig_freq=sr2, new_freq=16000)
        wav_ecapa = wav_ecapa.squeeze(0).unsqueeze(0)  # [1, T] en CPU
        with torch.no_grad():
            spk_emb = self.ecapa.encode_batch(wav_ecapa)  # [1, 192]
        spk_emb = spk_emb.squeeze(0).cpu()  # [192]

        return {"text": text, "mel": mel_spec, "spk_emb": spk_emb}


def collate_fn(batch):
    """
    Recibe lista de dicts {"text": str, "mel": Tensor[n_mels, T], "spk_emb": Tensor[192]}.
    Retorna:
      - texts: [str]
      - padded_mels: Tensor [B, n_mels, T_max]
      - mel_lengths: LongTensor [B]
      - spk_embs: Tensor [B, 192]
    """
    texts = [item["text"] for item in batch]
    mels = [item["mel"] for item in batch]
    spk_embs = [item["spk_emb"] for item in batch]

    mel_lengths = torch.LongTensor([mel.size(1) for mel in mels])
    max_mel_len = int(mel_lengths.max().item())
    n_mels = mels[0].size(0)
    padded_mels = torch.full((len(mels), n_mels, max_mel_len), fill_value=-100.0)
    for i, mel in enumerate(mels):
        T = mel.size(1)
        padded_mels[i, :, :T] = mel

    spk_embs = torch.stack(spk_embs, dim=0)  # [B, 192]

    return texts, padded_mels, mel_lengths, spk_embs


# --------------------------------------------------------------------------------
# 2) FUNCIONES DE ENTRENAMIENTO E INFERENCIA
# --------------------------------------------------------------------------------
import mlflow

from sklearn.model_selection import train_test_split

def split_dataset(dataset, val_ratio=0.1, seed=42):
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

def evaluate(model, dataloader, criterion, tts_utils, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for texts, mels, mel_lengths, spk_embs in dataloader:
            sequences, lengths = tts_utils.prepare_input_sequence(texts)
            sequences, lengths = sequences.to(device), lengths.to(device)
            mels, mel_lengths, spk_embs = mels.to(device), mel_lengths.to(device), spk_embs.to(device)

            B, _, T = mels.size()
            gate_padded = torch.zeros((B, T), dtype=torch.float32).to(device)
            for i, L in enumerate(mel_lengths):
                if L < T:
                    gate_padded[i, L:] = 1.0

            inputs, targets, spk_batch = model.parse_batch(
                (sequences, lengths, mels, gate_padded, mel_lengths, spk_embs)
            )
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(inputs, spk_batch)

            T_min = min(mel_outputs_postnet.size(2), mels.size(2))
            loss = criterion(
                mel_outputs_postnet[:, :, :T_min],
                mels[:, :, :T_min],
            )
            val_loss += loss.item()
    return val_loss / len(dataloader)

def guided_attention_loss(attention_weights, input_lengths, output_lengths, sigma=0.2):
    """
    attention_weights: Tensor de forma [B, T_decoder, T_encoder]
    input_lengths: LongTensor de forma [B]
    output_lengths: LongTensor de forma [B]
    sigma: Parámetro que controla la anchura de la banda diagonal
    """
    B, N, T = attention_weights.size()
    guided_masks = torch.zeros_like(attention_weights)
    for b in range(B):
        N_b = input_lengths[b].item()
        T_b = output_lengths[b].item()
        grid_i = torch.arange(T_b).unsqueeze(1).float() / T_b
        grid_j = torch.arange(N_b).unsqueeze(0).float() / N_b
        mask = 1.0 - torch.exp(-((grid_i - grid_j) ** 2) / (2 * sigma ** 2))
        guided_masks[b, :T_b, :N_b] = mask
    loss = torch.mean(attention_weights * guided_masks)
    return loss

class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    See https://github.com/espnet/espnet/blob/e962a3c609ad535cd7fb9649f9f9e9e0a2a27291/espnet/nets/pytorch_backend/e2e_tts_tacotron2.py#L25
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = self.make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = self.make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


    def make_non_pad_mask(self, lengths, xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, xs, length_dim)


    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Iniciando entrenamiento en dispositivo: {device}")

    use_mlflow = args.mlflow_host is not None and args.mlflow_host.startswith("http")
    if use_mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_host)
        mlflow.set_experiment("Tacotron2_SpeakerEmbeddings")
        mlflow_run = mlflow.start_run()
        mlflow.log_params(vars(args))
    else:
        print("[INFO] MLflow no está habilitado. Entrenamiento local solamente.")

    hparams = load_hparams(args.hparams_path)

    BASE_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "DeepLearningExamples", "PyTorch", "SpeechSynthesis", "Tacotron2", "tacotron2"
    ))
    import sys
    sys.path.append(BASE_DIR)
    from model import Tacotron2

    model = Tacotron2(
        mask_padding=hparams.mask_padding,
        n_mel_channels=hparams.n_mel_channels,
        n_symbols=len(hparams.symbols),
        symbols_embedding_dim=hparams.symbols_embedding_dim,
        encoder_kernel_size=hparams.encoder_kernel_size,
        encoder_n_convolutions=hparams.encoder_n_convolutions,
        encoder_embedding_dim=hparams.encoder_embedding_dim,
        attention_rnn_dim=hparams.attention_rnn_dim,
        attention_dim=hparams.attention_dim,
        attention_location_n_filters=hparams.attention_location_n_filters,
        attention_location_kernel_size=hparams.attention_location_kernel_size,
        n_frames_per_step=hparams.n_frames_per_step,
        decoder_rnn_dim=hparams.decoder_rnn_dim,
        prenet_dim=hparams.prenet_dim,
        max_decoder_steps=hparams.max_decoder_steps,
        gate_threshold=hparams.gate_threshold,
        p_attention_dropout=hparams.p_attention_dropout,
        p_decoder_dropout=hparams.p_decoder_dropout,
        postnet_embedding_dim=hparams.postnet_embedding_dim,
        postnet_kernel_size=hparams.postnet_kernel_size,
        postnet_n_convolutions=hparams.postnet_n_convolutions,
        decoder_no_early_stopping=hparams.decoder_no_early_stopping,
        spk_emb_size=args.spk_emb_size,
    ).to(device)

    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    ecapa.eval()

    full_dataset = LJSpeechDataset(args.data_dir, hparams, ecapa, device)
    train_dataset, val_dataset = split_dataset(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    criterion = nn.MSELoss()

    tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")

    best_val_loss = float("inf")
    train_losses_epoch = []
    val_losses_epoch = []
    loss_train_steps = []

    guided_attn_criterion = GuidedAttentionLoss(sigma=0.4, alpha=1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (texts, mels, mel_lengths, spk_embs) in enumerate(train_loader):
            sequences, lengths = tts_utils.prepare_input_sequence(texts)
            sequences, lengths = sequences.to(device), lengths.to(device)
            mels, mel_lengths, spk_embs = mels.to(device), mel_lengths.to(device), spk_embs.to(device)

            B, _, T = mels.size()
            gate_padded = torch.zeros((B, T), dtype=torch.float32).to(device)
            for i, L in enumerate(mel_lengths):
                if L < T:
                    gate_padded[i, L:] = 1.0

            inputs, targets, spk_batch = model.parse_batch((sequences, lengths, mels, gate_padded, mel_lengths, spk_embs))
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = model(inputs, spk_batch)

            # SAVE MEL COMPARISON IMAGE
            orig_mel_norm = mels[0].detach().cpu().numpy()
            pred_mel_norm = mel_outputs_postnet[0].detach().cpu().numpy()

            def to_db(mel):
                mel = np.maximum(mel, 1e-5)
                return 20 * np.log10(mel)

            orig_mel_db = to_db(orig_mel_norm)
            pred_mel_db = to_db(pred_mel_norm)

            plt.figure(figsize=(12, 5))
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(orig_mel_db, origin='lower', aspect='auto')
            ax1.set_title("MEL Original (dB)")
            ax1.set_xlabel("Frames")
            ax1.set_ylabel("Mel bins")
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(pred_mel_db, origin='lower', aspect='auto')
            ax2.set_title("MEL Predicho (Postnet, dB)")
            ax2.set_xlabel("Frames")
            ax2.set_ylabel("Mel bins")
            plt.suptitle(f"Comparativa MEL - Epoch {epoch:03d} / Batch {batch_idx:04d}")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            comp_path = os.path.join(
                args.vis_dir,
                f"compare_ep{epoch:03d}_batch{batch_idx:04d}.png"
            )
            plt.savefig(comp_path, dpi=150)
            plt.close()

            T_min = min(mel_outputs_postnet.size(2), mels.size(2))
            # Calcular la pérdida de reconstrucción
            mel_loss = criterion(mel_outputs, mels) + criterion(mel_outputs_postnet, mels)

            # Nueva pérdida de atención
            align_loss = guided_attn_criterion(alignments, lengths, mel_lengths // hparams.n_frames_per_step)
            gate_loss = F.binary_cross_entropy_with_logits(gate_outputs, gate_padded)

            # Combinación
            lambda_align = 1.0
            loss = mel_loss + gate_loss + lambda_align * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"[DEBUG] Epoch {epoch}/{args.epochs} ; Step {batch_idx+1}/{len(train_loader)} ; Loss {loss.item():.4f}", flush=True)
            loss_train_steps.append(loss.item())
             # Guardar curva de pérdida por step
            plt.figure(figsize=(8, 5))
            plt.plot(loss_train_steps, label='Train Loss per Step', color='green')
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Evolución del Loss por Step")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_step_path = os.path.join(args.checkpoints_dir, "loss_curve_steps.png")
            plt.savefig(plot_step_path)
            plt.close()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = evaluate(model, val_loader, criterion, tts_utils, device)

        train_losses_epoch.append(epoch_train_loss)
        val_losses_epoch.append(epoch_val_loss)

        print(f"[INFO] Epoch {epoch} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        if use_mlflow:
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

        scheduler.step()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(args.checkpoints_dir, "best_tts_zero_shot.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Nuevo mejor modelo guardado en: {ckpt_path}")

        # Guardar curva combinada
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses_epoch, label='Train Loss', color='blue')
        plt.plot(val_losses_epoch, label='Val Loss', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Curvas de pérdida: Train vs Validation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        plot_path = os.path.join(args.checkpoints_dir, "loss_curve_train_val.png")
        plt.savefig(plot_path)
        plt.close()

    if use_mlflow:
        mlflow.pytorch.log_model(model, "tacotron2_model")
        mlflow.end_run()

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # 1) Cargar hiperparámetros
    hparams = load_hparams(args.hparams_path)

    # 2) Instanciar Tacotron2 modificado con speaker embeddings
    BASE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "DeepLearningExamples",
            "PyTorch",
            "SpeechSynthesis",
            "Tacotron2",
            "tacotron2",
        )
    )
    import sys
    sys.path.append(BASE_DIR)
    from model import Tacotron2  # Versión modificada con spk_proj

    model = Tacotron2(
        mask_padding=hparams.mask_padding,
        n_mel_channels=hparams.n_mel_channels,
        n_symbols=len(hparams.symbols),
        symbols_embedding_dim=hparams.symbols_embedding_dim,
        encoder_kernel_size=hparams.encoder_kernel_size,
        encoder_n_convolutions=hparams.encoder_n_convolutions,
        encoder_embedding_dim=hparams.encoder_embedding_dim,
        attention_rnn_dim=hparams.attention_rnn_dim,
        attention_dim=hparams.attention_dim,
        attention_location_n_filters=hparams.attention_location_n_filters,
        attention_location_kernel_size=hparams.attention_location_kernel_size,
        n_frames_per_step=hparams.n_frames_per_step,
        decoder_rnn_dim=hparams.decoder_rnn_dim,
        prenet_dim=hparams.prenet_dim,
        max_decoder_steps=hparams.max_decoder_steps,
        gate_threshold=hparams.gate_threshold,
        p_attention_dropout=hparams.p_attention_dropout,
        p_decoder_dropout=hparams.p_decoder_dropout,
        postnet_embedding_dim=hparams.postnet_embedding_dim,
        postnet_kernel_size=hparams.postnet_kernel_size,
        postnet_n_convolutions=hparams.postnet_n_convolutions,
        decoder_no_early_stopping=hparams.decoder_no_early_stopping,
        spk_emb_size=args.spk_emb_size,
    )
    model = model.to(device)

    # 3) Cargar checkpoint entrenado
    print(f"[INFO] Cargando checkpoint: {args.checkpoint_path}")
    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("[INFO] Checkpoint cargado correctamente.")

    # 4) Extraer speaker embedding de referencia
    print(f"[INFO] Extrayendo speaker embedding de: {args.reference_wav}")
    spk_emb = get_spk_embedding(args.reference_wav, device)  # [192]
    spk_emb = spk_emb.unsqueeze(0).to(device)  # [1, 192]

    # 5) Convertir texto a IDs y longitudes
    print(f"[INFO] Procesando texto: \"{args.text}\"")
    tts_utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils")
    text_input = [args.text.lower()]
    sequences, lengths = tts_utils.prepare_input_sequence(text_input)
    sequences = sequences.to(device)
    lengths = lengths.to(device)

    # 6) Generar mel-spectrogram con Tacotron2 modificado
    print("[INFO] Generando mel-spectrogram con Tacotron2 modificado...")
    with torch.no_grad():
        mel_spec, mel_lengths, alignments = model.infer(sequences, lengths, spk_emb)

    # 7) Cargar vocoder WaveGlow y generar audio
    print("[INFO] Cargando WaveGlow desde TorchHub...")
    waveglow = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_waveglow")
    waveglow = waveglow.remove_weightnorm(waveglow).to(device)
    waveglow.eval()
    for p in waveglow.parameters():
        p.requires_grad = False

    print("[INFO] Convirtiendo mel a audio con WaveGlow...")
    with torch.no_grad():
        try:
            audio = waveglow.infer(mel_spec)
        except RuntimeError:
            audio = waveglow.infer(mel_spec.unsqueeze(1))
    audio = audio.squeeze().cpu()

    # 8) Guardar WAV resultante
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"{int(torch.randint(0, int(1e9), (1,)).item())}_clone.wav"
    )
    torchaudio.save(output_path, audio.unsqueeze(0), sample_rate=args.sample_rate)
    print(f"[INFO] Audio sintetizado guardado en: {output_path}")


# --------------------------------------------------------------------------------
# 3) PUNTO DE ENTRADA: argparse con subcomandos "train" e "infer"
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TTS con Tacotron2 + speaker embeddings (zero-shot integrado)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando "train"
    train_parser = subparsers.add_parser("train", help="Entrenar Tacotron2 con embeddings de speaker")
    train_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directorio LJSpeech con metadata.csv y carpeta wavs/"
    )
    train_parser.add_argument(
        "--hparams_path", type=str, required=True,
        help="Ruta al JSON de hiperparámetros de Tacotron2"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100,
        help="Número de épocas de entrenamiento"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Tamaño de batch"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate para Adam"
    )
    train_parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Número de workers para DataLoader"
    )
    train_parser.add_argument(
        "--checkpoints_dir", type=str, default="checkpoints",
        help="Directorio donde se guardarán checkpoints"
    )
    train_parser.add_argument(
        "--spk_emb_size", type=int, default=192,
        help="Dimensión del speaker embedding (ECAPA-TDNN = 192)"
    )
    # Añadir en el bloque train_parser:
    train_parser.add_argument(
        "--mlflow_host", type=str, default=None,
        help="Dirección de servidor MLflow (ej: http://admin:mlflow_password@mlflow.ip.nip.io). Si no se especifica, no se usa MLflow."
    )
    train_parser.add_argument(
        "--vis_dir", type=str, default="mel_vis",
        help="Directorio donde guardar comparativas de mel por step"
    )



    # Subcomando "infer"
    infer_parser = subparsers.add_parser("infer", help="Inferir TTS usando modelo entrenado")
    infer_parser.add_argument(
        "--hparams_path", type=str, required=True,
        help="Ruta al JSON de hiperparámetros de Tacotron2"
    )
    infer_parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Ruta al checkpoint .pth generado en entrenamiento"
    )
    infer_parser.add_argument(
        "--reference_wav", type=str, required=True,
        help="Archivo .wav de referencia (ej: data/my_voice/wavs/Voz1.wav)"
    )
    infer_parser.add_argument(
        "--text", type=str, required=True,
        help="Texto a sintetizar (minúsculas con puntuación en inglés)"
    )
    infer_parser.add_argument(
        "--output_dir", type=str, default="samples_cloned",
        help="Directorio donde se guardará el audio generado"
    )
    infer_parser.add_argument(
        "--spk_emb_size", type=int, default=192,
        help="Dimensión del speaker embedding (ECAPA-TDNN = 192)"
    )
    infer_parser.add_argument(
        "--sample_rate", type=int, default=22050,
        help="Frecuencia de muestreo para el WAV de salida"
    )

    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
