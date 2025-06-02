#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena Tacotron2 + WaveRNN sobre LJSpeech sin adapters LoRA usando DDP.
"""

import os
import json
import argparse
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchaudio.pipelines import TACOTRON2_WAVERNN_CHAR_LJSPEECH
from torchaudio.datasets import LJSPEECH

# Parámetros de mel-spectrogram
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 80
F_MIN = 0

# Pipeline base
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

# Carga robusta de WAV
_orig_load = torchaudio.load
def _safe_load(path, *args, **kwargs):
    try:
        return _orig_load(path, *args, **kwargs)
    except:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        return torch.from_numpy(wav).unsqueeze(0), sr
torchaudio.load = _safe_load

class SpeechCollate:
    def __init__(self, processor, vis_dir=None):
        self.processor = processor
        self.vis_dir = vis_dir
        self.batch_counter = 0
        if self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(self, batch):
        target_sr = batch[0][1]
        texts, waves = [], []
        for wav, sr, txt, *_ in batch:
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            texts.append(txt)
            waves.append(wav.squeeze(0))
        tokens, tok_lens = self.processor(texts)

        mel_specs = []
        mel_lens = []
        for w in waves:
            mel = mel_transform(w)
            mel = amplitude_to_db(mel)
            mel_specs.append(mel)
            mel_lens.append(mel.shape[1])

        if self.vis_dir:
            for idx, mel in enumerate(mel_specs):
                plt.figure(figsize=(10, 4))
                mel_np = mel.detach().cpu().numpy()
                plt.imshow(mel_np, aspect='auto', origin='lower')
                plt.xlabel("Frames")
                plt.ylabel("Mel bins")
                plt.title(f"MEL Spectrogram - batch {self.batch_counter}, sample {idx}")
                plt.colorbar(format="%+2.0f dB")
                plt.tight_layout()
                foto_path = os.path.join(self.vis_dir, f"mel_{self.batch_counter:04d}_{idx:02d}.png")
                plt.savefig(foto_path, dpi=150)
                plt.close()
            self.batch_counter += 1

        max_len = max(mel_lens)
        mel_padded = [F.pad(mel, (0, max_len - mel.shape[1])) for mel in mel_specs]
        mel_batch = torch.stack(mel_padded)
        mel_lens = torch.tensor(mel_lens, dtype=torch.int64)

        tok_lens, perm = tok_lens.sort(descending=True)
        tokens = tokens[perm]
        mel_batch = mel_batch[perm]
        mel_lens = mel_lens[perm]
        return tokens, tok_lens, mel_batch, mel_lens

def save_sample(model, vocoder, processor, device, out_path, text="Hola mundo"):
    model.eval()
    vocoder.eval()
    with torch.inference_mode():
        tokens, lengths = processor([text])
        tokens, lengths = tokens.to(device), lengths.to(device)
        mel_pred, mel_lens, _ = model.infer(tokens, lengths)
        mel_pred = mel_pred.to(device)
        mel_lens = mel_lens.to(device)
        wav_pred, _ = vocoder(mel_pred, mel_lens)
        wav = wav_pred[0].unsqueeze(0).cpu()
        torchaudio.save(out_path, wav, vocoder.sample_rate)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train(args):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(args.out, exist_ok=True)
    vis_dir = os.path.join(args.out, "visualizations") if local_rank == 0 else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    tacotron2 = bundle.get_tacotron2().to(device)
    vocoder = bundle.get_vocoder().to(device)

    if args.ckpt:
        tacotron2.load_state_dict(torch.load(args.ckpt, map_location=device))

    tacotron2 = torch.nn.parallel.DistributedDataParallel(tacotron2, device_ids=[local_rank])

    mel_criterion = torch.nn.MSELoss()
    gate_criterion = torch.nn.BCEWithLogitsLoss()

    os.makedirs(args.dataset, exist_ok=True)
    ds = LJSPEECH(root=args.dataset, download=args.download)
    sampler = DistributedSampler(ds)
    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        collate_fn=SpeechCollate(bundle.get_text_processor(), vis_dir=vis_dir),
    )

    optim = torch.optim.AdamW(tacotron2.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tacotron2.train()
        sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"\n========== EMPEZANDO EPOCH {epoch}/{args.epochs} ==========", flush=True)

        total_loss = 0.0
        steps = 0
        total_batches = len(loader)

        for batch_idx, (tokens, tok_lens, mel_specs, mel_lens) in enumerate(loader, start=1):
            if local_rank == 0:
                print(f"[DEBUG] Epoch {epoch}/{args.epochs} ; Step {batch_idx}/{total_batches}", flush=True)

            tokens = tokens.to(device)
            tok_lens = tok_lens.to(device)
            mel_specs = mel_specs.to(device)
            mel_lens = mel_lens.to(device)

            optim.zero_grad()
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2(
                tokens, tok_lens, mel_specs, mel_lens
            )
            mel_loss = (
                mel_criterion(mel_outputs, mel_specs)
                + mel_criterion(mel_outputs_postnet, mel_specs)
            )
            gate_targets = torch.zeros_like(gate_outputs)
            gate_targets[:, -1] = 1.0
            gate_loss = gate_criterion(gate_outputs, gate_targets)

            loss = mel_loss + gate_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tacotron2.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            steps += 1

        avg = total_loss / steps if steps else 0.0
        if local_rank == 0:
            print(f"→ Epoch {epoch} terminado. Avg-loss={avg:.3f}", flush=True)
            torch.save(tacotron2.module.state_dict(), f"{args.out}/tacotron2_{epoch:04d}.pth")
            torch.save(optim.state_dict(), f"{args.out}/optimizer_{epoch:04d}.pth")
            save_sample(
                tacotron2.module,
                vocoder,
                bundle.get_text_processor(),
                device,
                f"{args.out}/sample_{epoch:04d}.wav",
            )

    if local_rank == 0:
        with open(f"{args.out}/config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Ruta a LJSpeech")
    ap.add_argument("--out", required=True, help="Directorio de salida")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", help="Checkpoint para reanudar")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    train(args)