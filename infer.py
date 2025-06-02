#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer.py

Script de inferencia TTS: carga Tacotron2 + Vocoder + pesos LoRA
y genera un archivo WAV a partir de un texto.
"""

import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.pipelines import TACOTRON2_WAVERNN_CHAR_LJSPEECH
from torchaudio.datasets import LJSPEECH


# -------------------------------
# Reproducimos aquí las mismas definiciones de LoRA usadas en train
# -------------------------------
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
        # LoRA: x @ A^T --> [batch, r]; luego @ B^T --> [batch, out_features]
        lora_intermediate = x @ self.lora_A.T                   # [batch, r]
        lora_out = lora_intermediate @ self.lora_B.T             # [batch, out_features]
        lora_out = lora_out * self.scaling
        return original_out + lora_out


def apply_lora_to_model(model: nn.Module, r: int, alpha: float):
    """
    Recorre el modelo y reemplaza cada nn.Linear por un LoraLinear que
    contiene los mismos pesos originales + parámetros LoRA.
    """
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            to_replace.append(name)

    for name in to_replace:
        parent = model
        components = name.split('.')
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        attr = components[-1]
        orig_linear: nn.Linear = getattr(parent, attr)
        lora_linear = LoraLinear(orig_linear, r=r, alpha=alpha)
        setattr(parent, attr, lora_linear)


def save_waveform(wav_tensor: torch.Tensor, sample_rate: int, out_path: str):
    """
    Guarda un tensor de forma [1, N] como WAV.
    """
    # Aseguramos que la carpeta exista
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, wav_tensor, sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia TTS con Tacotron2 + Vocoder + LoRA"
    )
    parser.add_argument(
        "--base_ckpt",
        required=True,
        help="Checkpoint base de Tacotron2 (state_dict .pth) pre-entrenado",
    )
    parser.add_argument(
        "--lora_ckpt",
        required=True,
        help="Checkpoint con los pesos LoRA generados por finetune_lora.py",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        required=True,
        help="Rank (r) que se usó durante el entrenamiento LoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        required=True,
        help="Alpha que se usó durante el entrenamiento LoRA",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Texto a sintetizar (en inglés o español según el modelo)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Ruta al archivo WAV de salida (por ejemplo: out/sample.wav)",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1) Cargamos Tacotron2 base + aplicamos LoRA wrappers
    # --------------------------------------------------
    bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
    tacotron2 = bundle.get_tacotron2()

    # Cargamos pesos base de Tacotron2
    state_dict_base = torch.load(args.base_ckpt, map_location="cpu")
    tacotron2.load_state_dict(state_dict_base)

    # Envolvemos cada nn.Linear con LoRA (debe usar el mismo r y alpha del entrenamiento)
    apply_lora_to_model(tacotron2, r=args.lora_rank, alpha=args.lora_alpha)

    # Cargamos pesos LoRA (solo "lora_A" y "lora_B") con strict=False
    lora_state = torch.load(args.lora_ckpt, map_location="cpu")
    # Esto rellenará únicamente los parámetros cuyo nombre empiece por "lora_..."
    tacotron2.load_state_dict(lora_state, strict=False)

    # Lo llevamos a GPU (si disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tacotron2 = tacotron2.to(device).eval()

    # ------------------------------------
    # 2) Cargamos el vocoder de WaveRNN
    # ------------------------------------
    vocoder = bundle.get_vocoder().to(device).eval()

    # ------------------------------------
    # 3) Cargamos el procesador de texto
    # ------------------------------------
    processor = bundle.get_text_processor()

    # --------------------------------------------------
    # 4) Tokenizamos el texto y hacemos inferencia Tacotron2
    # --------------------------------------------------
    with torch.inference_mode():
        tokens, lengths = processor([args.text])
        tokens = tokens.to(device)
        lengths = lengths.to(device)

        # Inferencia: produce mel-spectrogram + longitudes
        mel_pred, mel_lens, _ = tacotron2.infer(tokens, lengths)
        mel_pred = mel_pred.to(device)
        mel_lens = mel_lens.to(device)

        # ------------------------------------
        # 5) Vocoder: convierte mel -> waveform
        # ------------------------------------
        wav_pred, _ = vocoder(mel_pred, mel_lens)
        # Normalmente wav_pred tiene forma [batch=1, T], float32
        wav = wav_pred[0].unsqueeze(0).cpu()

    # -------------------------------
    # 6) Guardamos el WAV resultante
    # -------------------------------
    sample_rate = _get_sample_rate(vocoder)
    save_waveform(wav, sample_rate, args.out)
    print(f"✔️  Archivo sintetizado guardado en: {args.out}")


def _get_sample_rate(vocoder_model):
    """
    Extrae el sample_rate del vocoder (bundle.get_vocoder()).
    """
    try:
        return vocoder_model.sample_rate
    except AttributeError:
        return 22050


if __name__ == "__main__":
    main()
