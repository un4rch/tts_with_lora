#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning Tacotron2 con LoRA usando PEFT y un dataset personalizado tipo LJSpeech.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import LoraLayer
from torchaudio.pipelines import TACOTRON2_WAVERNN_CHAR_LJSPEECH
from torchaudio.datasets import LJSPEECH
from torch.utils.data import DataLoader


# ✅ Collate con resample y conversión a mono
class SpeechCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        import torchaudio
        from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

        target_sr = batch[0][1]
        texts, waves = [], []
        for wav, sr, txt, *_ in batch:
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            waves.append(wav.squeeze(0))
            texts.append(txt)

        tokens, tok_lens = self.processor(texts)

        mel_transform = MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=target_sr // 2,
            n_mels=80,
        )
        amplitude_to_db = AmplitudeToDB(stype="power")

        mel_specs = []
        mel_lens = []
        for w in waves:
            mel = mel_transform(w)
            mel = amplitude_to_db(mel)
            mel_specs.append(mel)
            mel_lens.append(mel.shape[1])

        max_len = max(mel_lens)
        mel_padded = [F.pad(mel, (0, max_len - mel.shape[1])) for mel in mel_specs]
        mel_batch = torch.stack(mel_padded)
        mel_lens = torch.tensor(mel_lens, dtype=torch.int64)

        tok_lens, perm = tok_lens.sort(descending=True)
        tokens = tokens[perm]
        mel_batch = mel_batch[perm]
        mel_lens = mel_lens[perm]
        return tokens, tok_lens, mel_batch, mel_lens


# ✅ Solución robusta para aplicar LoRA sin romper Tacotron2.forward
def mark_lora(module):
    if isinstance(module, LoraLayer):
        module._is_lora_layer = True
    for child in module.children():
        mark_lora(child)

def apply_lora(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "decoder.rnn",
            "decoder.linear_projection",
            "encoder.convolutions.0.0",
            "encoder.convolutions.1.0",
            "encoder.convolutions.2.0"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION  # requerido aunque no se use forward wrapper
    )

    peft_model = get_peft_model(model, config)
    mark_lora(peft_model)  # evitar override del forward()
    return peft_model


def train_lora(model, loader, device, output_dir, epochs=10, lr=1e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for tokens, tok_lens, mel_specs, mel_lens in loader:
            tokens, tok_lens = tokens.to(device), tok_lens.to(device)
            mel_specs, mel_lens = mel_specs.to(device), mel_lens.to(device)

            optimizer.zero_grad()
            mel_out, mel_post, gate_out, _ = model(tokens, tok_lens, mel_specs, mel_lens)
            loss = criterion(mel_post, mel_specs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"[EPOCH {epoch}] Loss promedio: {avg:.4f}")

    # Guardar solo los LoRA adapters
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"✅ LoRA guardado en: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True, help="Ruta al checkpoint base de tacotron2")
    parser.add_argument("--data", default="./data/my_voice", help="Ruta al dataset tipo LJSpeech")
    parser.add_argument("--out", default="./checkpoints/lora_myvoice", help="Ruta para guardar LoRA")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle = TACOTRON2_WAVERNN_CHAR_LJSPEECH
    processor = bundle.get_text_processor()

    # Cargar modelo base
    model = bundle.get_tacotron2().to(device)
    model.load_state_dict(torch.load(args.base_ckpt, map_location=device))

    print("✅ Modelo base cargado")
    print(model.encoder.convolutions)

    # Aplicar LoRA
    model = apply_lora(model).to(device)

    # Dataset
    ds = LJSPEECH(root=args.data, download=False)
    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=SpeechCollate(processor)
    )

    # Entrenamiento
    train_lora(model, loader, device, args.out, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()