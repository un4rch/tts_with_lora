#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tunea Tacotron2 usando LoRA para adaptar el modelo a una voz específica.
"""

import os
import json
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchaudio.pipelines import TACOTRON2_WAVERNN_CHAR_LJSPEECH
from torchaudio.datasets import LJSPEECH

# Parámetros de mel-spectrogram
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 80
F_MIN = 0

# Carga robusta de WAV (si falla torchaudio.load, usa soundfile)
_orig_load = torchaudio.load
def _safe_load(path, *args, **kwargs):
    try:
        return _orig_load(path, *args, **kwargs)
    except:
        import soundfile as sf
        wav_np, sr = sf.read(path, dtype="float32", always_2d=False)
        # soundfile.read (always_2d=False) devuelve (N,) para mono o (N, C) para multi
        # Convertimos a tensor con shape [C, N] para asemejar a torchaudio.load
        if wav_np.ndim == 1:
            wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)  # [1, N]
        else:
            # wav_np.shape == (N, C) -> queremos (C, N)
            wav_tensor = torch.from_numpy(wav_np.T)  # [C, N]
        return wav_tensor, sr

torchaudio.load = _safe_load

# Pipeline base y transformaciones de mel-spectrogram
bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
_vocoder_for_rate = bundle.get_vocoder()
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=_vocoder_for_rate.sample_rate,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    f_min=F_MIN,
    f_max=_vocoder_for_rate.sample_rate // 2,
    n_mels=N_MELS,
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')


class SpeechCollate:
    """
    Collate para textos y ondas. Transforma cada wav a mono, luego a mel-spectrograma y aplica padding.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # batch: lista de tuplas (waveform, sample_rate, text, ...)
        target_sr = batch[0][1]
        texts, waves = [], []
        for wav, sr, txt, *_ in batch:
            # 1) Resample si hace falta
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            # wav: tensor [C, N] (C = canales). Queremos mono [N].
            if wav.ndim > 1 and wav.shape[0] > 1:
                # Promediar canales: (C, N) -> (N,)
                wav_mono = wav.mean(dim=0)
            else:
                # Ya es [1, N] o [N]; forzamos a [N]
                wav_mono = wav.squeeze(0)
            texts.append(txt)
            waves.append(wav_mono)

        # Tokenización de texto
        tokens, tok_lens = self.processor(texts)

        # Cálculo de mel-spectrogramas y sus longitudes
        mel_specs = []
        mel_lens = []
        for w in waves:
            # w: [N], mel_transform espera [..., time], así que produce [n_mels, T]
            m = mel_transform(w)
            m = amplitude_to_db(m)
            mel_specs.append(m)
            mel_lens.append(m.shape[1])

        # Padding a la longitud máxima en el batch
        max_len = max(mel_lens)
        mel_padded = [F.pad(m, (0, max_len - m.shape[1])) for m in mel_specs]
        mel_batch = torch.stack(mel_padded)         # [batch, n_mels, max_len]
        mel_lens = torch.tensor(mel_lens, dtype=torch.int64)

        # Orden descendente por longitud de tokens
        tok_lens, perm = tok_lens.sort(descending=True)
        tokens = tokens[perm]
        mel_batch = mel_batch[perm]
        mel_lens = mel_lens[perm]
        return tokens, tok_lens, mel_batch, mel_lens


class LoraLinear(nn.Module):
    """
    Módulo LoRA para reemplazar un nn.Linear original.
    Mantiene el peso original congelado y añade parámetros de baja-rank (A y B).
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        # Propiedades del layer original
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Copiamos referencia al peso y sesgo originales
        self.weight = original_linear.weight
        self.bias = original_linear.bias

        # Congelamos los parámetros originales
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Inicializamos parámetros LoRA (A y B)
        # A: (r, in_features), B: (out_features, r)
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Inicialización recomendada (He/Kaiming) para A, ceros para B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computa salida original
        original_out = F.linear(x, self.weight, self.bias)
        # x: [batch, in_features]
        # LoRA: x @ A^T --> [batch, r]; luego @ B^T --> [batch, out_features]
        lora_intermediate = x @ self.lora_A.T                   # [batch, r]
        lora_out = lora_intermediate @ self.lora_B.T             # [batch, out_features]
        lora_out = lora_out * self.scaling
        return original_out + lora_out


def apply_lora_to_model(model: nn.Module, r: int, alpha: float):
    """
    Recorrer el modelo y reemplazar cada nn.Linear por un LoraLinear que
    contiene los mismos pesos originales + parámetros LoRA.
    """
    # Primero obtenemos la lista de rutas (names) de todos los nn.Linear
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            to_replace.append(name)

    for name in to_replace:
        # Navegamos hasta el módulo padre
        parent = model
        components = name.split('.')
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        attr = components[-1]
        orig_linear: nn.Linear = getattr(parent, attr)
        # Creamos el LoraLinear con el mismo peso
        lora_linear = LoraLinear(orig_linear, r=r, alpha=alpha)
        # Reemplazamos en el padre
        setattr(parent, attr, lora_linear)


def save_sample(model: nn.Module, vocoder: nn.Module, processor, device, out_path: str, text: str = "Hola mundo"):
    """
    Genera un sample de audio a partir de un texto de prueba y lo guarda en disco.
    """
    model.eval()
    vocoder.eval()
    with torch.inference_mode():
        tokens, lengths = processor([text])
        tokens, lengths = tokens.to(device), lengths.to(device)
        # Inferencia con Tacotron2 + LoRA
        mel_pred, mel_lens, _ = model.infer(tokens, lengths)
        mel_pred = mel_pred.to(device)
        mel_lens = mel_lens.to(device)
        # Vocoder convierte mel a waveform
        wav_pred, _ = vocoder(mel_pred, mel_lens)
        wav = wav_pred[0].unsqueeze(0).cpu()
        torchaudio.save(out_path, wav, vocoder.sample_rate)


def train(args):
    # Configurar dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out, exist_ok=True)

    # Cargamos el modelo Tacotron2 base
    tacotron2 = bundle.get_tacotron2()
    # Cargamos pesos desde el checkpoint
    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        tacotron2.load_state_dict(state_dict)
    else:
        raise ValueError("Debe especificar --ckpt con el checkpoint de Tacotron2 pre-entrenado.")

    # Aplicamos LoRA a todas las capas lineales
    apply_lora_to_model(tacotron2, r=args.lora_rank, alpha=args.lora_alpha)
    tacotron2 = tacotron2.to(device)

    # Vocoder para convertir mel a wav
    vocoder = bundle.get_vocoder().to(device)

    # Procesador de texto (tokenización)
    processor = bundle.get_text_processor()

    # Dataset LJSPEECH (o similar con tu propia voz)
    ds = LJSPEECH(root=args.dataset, download=False)
    collate = SpeechCollate(processor)
    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    # Criterios de pérdida
    mel_criterion = torch.nn.MSELoss()
    gate_criterion = torch.nn.BCEWithLogitsLoss()

    # Solo optimizamos parámetros LoRA (aquellos con requires_grad=True)
    lora_parameters = [p for p in tacotron2.parameters() if p.requires_grad]
    if not lora_parameters:
        raise RuntimeError("No se encontraron parámetros LoRA para optimizar. ¿Se aplicó LoRA correctamente?")

    optim = torch.optim.AdamW(lora_parameters, lr=args.lr)

    # Bucle de entrenamiento
    for epoch in range(1, args.epochs + 1):
        tacotron2.train()
        total_loss = 0.0
        steps = 0
        total_batches = len(loader)

        print(f"\n========== EMPEZANDO EPOCH {epoch}/{args.epochs} ==========", flush=True)
        for batch_idx, (tokens, tok_lens, mel_specs, mel_lens) in enumerate(loader, start=1):
            # Mover tensores a dispositivo
            tokens = tokens.to(device)
            tok_lens = tok_lens.to(device)
            mel_specs = mel_specs.to(device)
            mel_lens = mel_lens.to(device)

            optim.zero_grad()
            # Forward Tacotron2 (ahora con LoRA)
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2(
                tokens, tok_lens, mel_specs, mel_lens
            )
            # Pérdida de mel
            mel_loss = (
                mel_criterion(mel_outputs, mel_specs)
                + mel_criterion(mel_outputs_postnet, mel_specs)
            )
            # Pérdida de gate (stop token)
            gate_targets = torch.zeros_like(gate_outputs)
            gate_targets[:, -1] = 1.0
            gate_loss = gate_criterion(gate_outputs, gate_targets)

            loss = mel_loss + gate_loss
            loss.backward()
            # Clip de gradiente para estabilidad
            torch.nn.utils.clip_grad_norm_(lora_parameters, 1.0)
            optim.step()

            total_loss += loss.item()
            steps += 1
            if batch_idx % 50 == 0 or batch_idx == total_batches:
                avg_batch = total_loss / steps if steps else 0.0
                print(f"[DEBUG] Epoch {epoch} ; Step {batch_idx}/{total_batches} ; Avg-loss hasta ahora: {avg_batch:.4f}", flush=True)

        avg_epoch = total_loss / steps if steps else 0.0
        print(f"→ Epoch {epoch} terminado. Avg-loss={avg_epoch:.4f}", flush=True)

        # Guardamos pesos LoRA al final de la época
        lora_state = {
            k: v.cpu()
            for k, v in tacotron2.state_dict().items()
            if "lora_" in k
        }
        torch.save(lora_state, os.path.join(args.out, f"lora_epoch_{epoch:03d}.pth"))

        # Guardamos muestra de audio generada con el modelo adaptado
        """sample_path = os.path.join(args.out, f"sample_epoch_{epoch:03d}.wav")
        save_sample(
            tacotron2,
            vocoder,
            processor,
            device,
            sample_path,
            text=args.sample_text,
        )"""

    # Guardamos configuración final
    cfg = {
        "dataset": args.dataset,
        "out": args.out,
        "epochs": args.epochs,
        "bs": args.bs,
        "lr": args.lr,
        "ckpt": args.ckpt,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "sample_text": args.sample_text,
    }
    with open(os.path.join(args.out, "config_finetune_lora.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fine-tunea Tacotron2 con LoRA para adaptación a voz específica"
    )
    ap.add_argument(
        "--dataset", required=True, help="Ruta al dataset LJSpeech (o similar) de la voz específica"
    )
    ap.add_argument(
        "--out", required=True, help="Directorio de salida donde se guardarán pesos LoRA y muestras"
    )
    ap.add_argument(
        "--epochs", type=int, default=100, help="Número de épocas de fine-tuning"
    )
    ap.add_argument(
        "--bs", type=int, default=16, help="Tamaño de batch para entrenamiento"
    )
    ap.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate para parámetros LoRA"
    )
    ap.add_argument(
        "--ckpt", required=True, help="Checkpoint pre-entrenado de Tacotron2 (state_dict .pth)"
    )
    ap.add_argument(
        "--lora_rank", type=int, default=8, help="Rango bajo (rank) para LoRA"
    )
    ap.add_argument(
        "--lora_alpha", type=float, default=32.0, help="Factor de escala alpha para LoRA"
    )
    ap.add_argument(
        "--sample_text",
        type=str,
        default="Este es un ejemplo de fine-tuning con LoRA.",
        help="Texto de prueba para generar muestras de audio durante el entrenamiento",
    )
    args = ap.parse_args()
    train(args)