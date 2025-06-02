#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de inferencia para Tacotron2 + WaveRNN entrenados.
"""

import os
import argparse
import torch
import torchaudio
from torchaudio.pipelines import TACOTRON2_WAVERNN_CHAR_LJSPEECH

def load_model(checkpoint_path, device):
    bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
    model = bundle.get_tacotron2().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer(text, tacotron2, vocoder, processor, device, output_path):
    tacotron2.eval()
    vocoder.eval()

    with torch.inference_mode():
        tokens, lengths = processor([text])
        tokens = tokens.to(device)
        lengths = lengths.to(device)

        mel_outputs, mel_lengths, _ = tacotron2.infer(tokens, lengths)
        mel_outputs = mel_outputs.to(device)
        mel_lengths = mel_lengths.to(device)

        wav, _ = vocoder(mel_outputs, mel_lengths)
        wav = wav[0].unsqueeze(0).cpu()

        torchaudio.save(output_path, wav, sample_rate=vocoder.sample_rate)
        print(f"✅ Audio generado en: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Ruta al checkpoint tacotron2_XXXX.pth")
    parser.add_argument("--text", required=True, help="Texto a sintetizar")
    parser.add_argument("--out", default="output.wav", help="Archivo de salida WAV")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo")
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()
    vocoder = bundle.get_vocoder().to(device)

    tacotron2 = load_model(args.ckpt, device)
    infer(args.text, tacotron2, vocoder, processor, device, args.out)

if __name__ == "__main__":
    main()