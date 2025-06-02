#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inferencia con modelo base + LoRA.
"""

import os
import argparse
import torch
import torchaudio
from torchaudio.pipelines import TACOTRON2_WAVERNN_CHAR_LJSPEECH
from peft import PeftModel

def load_model_with_lora(base_ckpt, lora_dir, device):
    bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
    model = bundle.get_tacotron2().to(device)
    model.load_state_dict(torch.load(base_ckpt, map_location=device))
    model.eval()
    model = PeftModel.from_pretrained(model, lora_dir).to(device)
    return model

def infer(text, model, vocoder, processor, device, output_path):
    model.eval()
    vocoder.eval()
    with torch.inference_mode():
        tokens, lengths = processor([text])
        tokens, lengths = tokens.to(device), lengths.to(device)
        mel_pred, mel_lens, _ = model.infer(tokens, lengths)
        wav, _ = vocoder(mel_pred, mel_lens)
        torchaudio.save(output_path, wav[0].unsqueeze(0).cpu(), vocoder.sample_rate)
        print(f"✅ Audio generado: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True, help="Checkpoint tacotron2 base")
    parser.add_argument("--lora_dir", required=True, help="Directorio con LoRA fine-tuneado")
    parser.add_argument("--text", required=True, help="Texto a sintetizar")
    parser.add_argument("--out", default="output.wav", help="Archivo de salida")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()
    vocoder = bundle.get_vocoder().to(device)

    model = load_model_with_lora(args.base_ckpt, args.lora_dir, device)
    infer(args.text, model, vocoder, processor, device, args.out)

if __name__ == "__main__":
    main()