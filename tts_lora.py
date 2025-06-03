#!/usr/bin/env python3
import os
import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
python tts_lora.py train \
    --data_dir data/my_voice \
    --loras_dir loras \
    --vis_dir visualizations \
    --epochs 100000 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --early_stopping_patience 2000 \
    --vis_interval 200 \
    --num_workers 4 \
    --sample_rate 22050 \
    --n_mels 80 \
    --n_fft 1024 \
    --hop_length 256 \
    --win_length 1024 \
    --seed 42 \
    --train_frac 0.8 \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --lr_factor 0.5 \
    --lr_patience 1000

python tts_lora.py infer \
  --text "This is a speech synthesis test using text to speech model" \
  --lora_path loras/best_lora.pth \
  --output_dir samples \
  --lora_rank 8 \
  --lora_alpha 32
"""

# --------------------------------------------------------------------------------
# 1) UTILIDADES GENERALES
# --------------------------------------------------------------------------------

# Mantendremos tts_utils en None hasta que lo carguemos en train/infer.
tts_utils = None


# --------------------------------------------------------------------------------
# 2) CLASE LoRALinear
# --------------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, orig_module: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = orig_module.in_features
        self.out_features = orig_module.out_features
        self.r = r
        self.alpha = alpha

        # Guardamos peso y bias originales, congelados
        self.weight = orig_module.weight
        self.bias = orig_module.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Adaptadores A y B
        self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_features, r))
        self.scaling = self.alpha / self.r

    def forward(self, x):
        orig = nn.functional.linear(x, self.weight, self.bias)
        lora_part = (x @ self.A.t()) @ self.B.t()
        return orig + lora_part * self.scaling


# --------------------------------------------------------------------------------
# 3) FUNCIÓN RECURSIVA PARA INYECTAR LoRA
# --------------------------------------------------------------------------------

def replace_linear_with_lora(model: nn.Module, r: int, alpha: float):
    """
    Recorre recursivamente los submódulos de `model`.
    Cada vez que encuentra un nn.Linear, lo reemplaza por LoRALinear(orig, r, alpha).
    Retorna cuántas capas lineales fueron reemplazadas.
    """
    replaced = 0
    for name, child in list(model._modules.items()):
        if child is None:
            continue
        if isinstance(child, nn.Linear):
            lora_mod = LoRALinear(child, r=r, alpha=alpha)
            device = child.weight.device
            lora_mod.to(device)
            model._modules[name] = lora_mod
            replaced += 1
        else:
            replaced += replace_linear_with_lora(child, r, alpha)
    return replaced


# --------------------------------------------------------------------------------
# 4) DIVISIÓN DE metadata.csv
# --------------------------------------------------------------------------------

def split_metadata_file(metadata_path: str,
                        output_dir: str,
                        seed: int = 1234,
                        train_frac: float = 0.8,
                        val_frac: float = 0.1,
                        test_frac: float = 0.1):
    """
    Divide el archivo metadata.csv en:
      - train_metadata.csv
      - val_metadata.csv
      - test_metadata.csv
    según las fracciones indicadas. Mezcla líneas con seed fijo.
    """
    random.seed(seed)
    p = Path(metadata_path)
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.shuffle(lines)

    n = len(lines)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_lines = lines[:n_train]
    val_lines = lines[n_train:n_train + n_val]
    test_lines = lines[n_train + n_val:]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_metadata.csv").write_text("\n".join(train_lines), encoding="utf-8")
    (out_dir / "val_metadata.csv").write_text("\n".join(val_lines), encoding="utf-8")
    (out_dir / "test_metadata.csv").write_text("\n".join(test_lines), encoding="utf-8")
    print(f"[INFO] División: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test.")


# --------------------------------------------------------------------------------
# 5) DATASET PERSONALIZADO PARA VOCES
# --------------------------------------------------------------------------------

class VoiceDataset(Dataset):
    def __init__(self,
                 data_dir,
                 metadata_filename="train_metadata.csv",
                 sample_rate=22050,
                 n_mels=80,
                 n_fft=1024,
                 hop_length=256,
                 win_length=1024):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.wav_dir = self.data_dir / "wavs"
        self.metadata_path = self.data_dir / metadata_filename
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"No se encontró {self.metadata_path}")

        lines = self.metadata_path.read_text(encoding="utf-8").strip().split("\n")
        self.entries = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            fileid, transcript, norm_transcript = parts
            wav_path = self.wav_dir / f"{fileid}.wav"
            if wav_path.exists():
                self.entries.append({
                    "wav_path": wav_path,
                    "text": norm_transcript.lower()
                })

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav, sr = torchaudio.load(str(entry["wav_path"]))  # [n_channels, T]
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # Convertir a mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        mel_spec = self.mel_transform(wav)  # [1, n_mels, T_frames]
        mel_spec_db = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
        )
        return {
            "text": entry["text"],
            "mel": mel_spec_db.squeeze(0)  # [n_mels, T_frames]
        }


def collate_fn(batch):
    """
    Recibe una lista de diccionarios {"text": str, "mel": Tensor[n_mels, T]}.
    Devuelve:
      - texts: lista de strings
      - padded_mels: Tensor [B, n_mels, T_max]
      - mel_lengths: Tensor [B]
    Conversion a secuencias se hará en el training loop (para evitar CUDA en workers).
    """
    texts = [item["text"] for item in batch]
    mels = [item["mel"] for item in batch]

    mel_lengths = torch.LongTensor([mel.size(1) for mel in mels])
    max_mel_len = int(mel_lengths.max().item())
    n_mels = mels[0].size(0)
    padded_mels = torch.full((len(mels), n_mels, max_mel_len), fill_value=-100.0)
    for i, mel in enumerate(mels):
        T = mel.size(1)
        padded_mels[i, :, :T] = mel

    return texts, padded_mels, mel_lengths


# --------------------------------------------------------------------------------
# 6) MÉTRICAS OBJETIVAS PARA EVALUACIÓN
# --------------------------------------------------------------------------------

def compute_mcd(mel_ref, mel_pred, sample_rate=22050, n_mfcc=13):
    """
    Aproximación de MCD: extraemos MFCC de cada mel-spectrogram (convertido a wav
    a través de Griffin-Lim). No es idéntico a MCD clásico, pero sirve como proxy.

    - mel_ref, mel_pred: tensores [n_mels, T_frames] en escala dB.
    - Retorna un float: MCD promedio (dB).
    """
    n_mels = mel_ref.size(0)
    n_fft = 1024
    hop_length = 256
    win_length = 1024

    mel_to_spec = torchaudio.transforms.InverseMelScale(
        n_stft=(n_fft // 2) + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=0.0,
        f_max=sample_rate / 2.0
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )

    # dB → potencia
    ref_power = torchaudio.functional.DB_to_amplitude(mel_ref, ref=1.0, power=1.0)
    pred_power = torchaudio.functional.DB_to_amplitude(mel_pred, ref=1.0, power=1.0)

    spec_ref = mel_to_spec(ref_power)    # [n_stft, T]
    spec_pred = mel_to_spec(pred_power)  # [n_stft, T]

    wav_ref = griffin_lim(spec_ref.unsqueeze(0)).squeeze(0)   # [T_audio]
    wav_pred = griffin_lim(spec_pred.unsqueeze(0)).squeeze(0) # [T_audio]

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "win_length": win_length
        }
    )
    mfcc_ref = mfcc_transform(wav_ref.unsqueeze(0))   # [1, n_mfcc, T_mfcc]
    mfcc_pred = mfcc_transform(wav_pred.unsqueeze(0)) # [1, n_mfcc, T_mfcc]

    min_frames = min(mfcc_ref.size(2), mfcc_pred.size(2))
    mfcc_ref, mfcc_pred = mfcc_ref[:, :, :min_frames], mfcc_pred[:, :, :min_frames]

    diff = mfcc_ref - mfcc_pred  # [1, n_mfcc, T]
    diff_sq = diff.pow(2).sum(dim=1)  # [1, T]
    mcd_frames = (10.0 / torch.log(torch.tensor(10.0))) * torch.sqrt(2.0 * diff_sq)
    return float(mcd_frames.mean().item())


# --------------------------------------------------------------------------------
# 7) EVALUACIÓN (val/test)
# --------------------------------------------------------------------------------

def evaluate_model(tacotron2, waveglow, dataset, device):
    """
    Recorre `dataset` (VoiceDataset con split “val” o “test”), genera mel con Tacotron2
    y compara con mel real, calculando:
      - MSE en mel-spectrogram
      - MCD aproximado
    Retorna un dict con promedios.
    """
    tacotron2.eval()
    waveglow.eval()
    mse_criterion = nn.MSELoss(reduction="mean")

    total_mse = 0.0
    total_mcd = 0.0
    n_samples = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for (texts, mels, mel_lengths) in tqdm(loader, desc="Evaluación"):
            # Preparar secuencias con tts_utils (en CPU; luego movemos a device)
            sequences, lengths = tts_utils.prepare_input_sequence(texts)
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)

            # Inference Tacotron2
            mel_pred_tuple = tacotron2.infer(sequences, lengths)
            if isinstance(mel_pred_tuple, (list, tuple)):
                mel_pred = None
                for o in mel_pred_tuple:
                    if torch.is_tensor(o) and o.dim() >= 2:
                        mel_pred = o
                        break
            else:
                mel_pred = mel_pred_tuple

            if mel_pred.dim() == 2:
                mel_pred = mel_pred.unsqueeze(0)
            mel_pred = mel_pred.float()

            # Ajustar longitudes
            T_pred = mel_pred.size(2)
            T_target = mels.size(2)
            T_min = min(T_pred, T_target)
            mel_ref = mels[:, :, :T_min].squeeze(0).cpu()     # [n_mels, T_min]
            mel_out = mel_pred[:, :, :T_min].squeeze(0).cpu() # [n_mels, T_min]

            # 1) MSE mel
            mse_val = mse_criterion(mel_out, mel_ref).item()
            total_mse += mse_val

            # 2) MCD aproximado
            mcd_val = compute_mcd(mel_ref, mel_out, sample_rate=dataset.sample_rate)
            total_mcd += mcd_val

            n_samples += 1

    avg_mse = total_mse / max(n_samples, 1)
    avg_mcd = total_mcd / max(n_samples, 1)

    return {
        "mel_mse": avg_mse,
        "mcd": avg_mcd,
        "n_samples": n_samples
    }


# --------------------------------------------------------------------------------
# 8) ENTRENAMIENTO E INFERENCIA
# --------------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # 1) Cargar tts_utils (CPU)
    global tts_utils
    print("[INFO] Cargando utils para preparar texto...")
    tts_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    # 2) Dividir metadata.csv
    split_metadata_file(
        metadata_path=os.path.join(args.data_dir, "metadata.csv"),
        output_dir=args.data_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac
    )

    # 3) Crear datasets y dataloaders
    train_dataset = VoiceDataset(
        data_dir=args.data_dir,
        metadata_filename="train_metadata.csv",
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )
    val_dataset = VoiceDataset(
        data_dir=args.data_dir,
        metadata_filename="val_metadata.csv",
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True
    )

    # 4) Cargar Tacotron2 preentrenado
    print("[INFO] Cargando Tacotron2 preentrenado desde TorchHub...")
    tacotron2 = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_tacotron2',
        model_math='fp16' if device.type == 'cuda' else 'fp32'
    )
    tacotron2 = tacotron2.to(device)
    tacotron2.train()

    # 5) Inyectar LoRA en todas las nn.Linear
    print(f"[INFO] Reemplazando capas nn.Linear por LoRALinear (r={args.lora_rank}, alpha={args.lora_alpha})...")
    n_replaced = replace_linear_with_lora(tacotron2, r=args.lora_rank, alpha=args.lora_alpha)
    print(f"[INFO] Se reemplazaron {n_replaced} capas lineales con LoRALinear.")

    # 6) Congelar parámetros originales; entrenar solo LoRA
    lora_params = []
    for name, param in tacotron2.named_parameters():
        if 'A' in name or 'B' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    total_lora = sum(p.numel() for p in lora_params)
    print(f"[INFO] Parámetros totales a entrenar (solo LoRA): {total_lora} parámetros.")

    # 7) Cargar WaveGlow (solo para sampleos)
    print("[INFO] Cargando WaveGlow preentrenado desde TorchHub...")
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.to(device)
    waveglow.eval()
    for p in waveglow.parameters():
        p.requires_grad = False

    # 8) Optimizer, scheduler y pérdida
    optimizer = torch.optim.Adam(lora_params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Crear carpetas
    os.makedirs(args.loras_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # 9) Loop de entrenamiento
    for epoch in range(1, args.epochs + 1):
        tacotron2.train()
        running_loss = 0.0
        n_batches = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, (texts, mels, mel_lengths) in enumerate(loop, start=1):
            # Preparar secuencias en el proceso principal
            sequences, lengths = tts_utils.prepare_input_sequence(texts)
            sequences = sequences.to(device)
            lengths = lengths.to(device)

            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)

            # Forward Tacotron2
            max_len = int(mel_lengths.max().item())
            mel_outputs, mel_outputs_postnet, _, _ = tacotron2((sequences, lengths, mels, max_len, mel_lengths))

            # Calcular MSE en mel
            T_pred = mel_outputs_postnet.size(2)
            T_target = mels.size(2)
            T_min = min(T_pred, T_target)
            loss = criterion(
                mel_outputs_postnet[:, :, :T_min],
                mels[:, :, :T_min]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

            # Imprimir LR actual en DEBUG
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[DEBUG] Epoch {epoch} ; Step {batch_idx} ; Loss {loss.item():.6f} ; LR {current_lr:.6e}")

            # Visualización MEL cada vis_interval pasos
            if batch_idx % args.vis_interval == 0:
                with torch.no_grad():
                    orig_mel_db = mels[0].detach().cpu()
                    pred_mel_db = mel_outputs_postnet[0].detach().cpu()
                    plt.figure(figsize=(12, 5))
                    ax1 = plt.subplot(1, 2, 1)
                    ax1.imshow(orig_mel_db.numpy(), origin='lower', aspect='auto')
                    ax1.set_title("MEL Original (dB)")
                    ax1.set_xlabel("Frames")
                    ax1.set_ylabel("Mel bins")
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.imshow(pred_mel_db.numpy(), origin='lower', aspect='auto')
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

        epoch_loss = running_loss / n_batches
        print(f"[INFO] Epoch {epoch} completada. Loss promedio: {epoch_loss:.6f}")

        # 10) Evaluación en validación con .infer()
        tacotron2.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for (texts_v, mels_v, mel_lengths_v) in val_loader:
                sequences_v, lengths_v = tts_utils.prepare_input_sequence(texts_v)
                sequences_v = sequences_v.to(device)
                lengths_v = lengths_v.to(device)

                mels_v = mels_v.to(device)
                mel_lengths_v = mel_lengths_v.to(device)

                mel_pred_tuple = tacotron2.infer(sequences_v, lengths_v)
                if isinstance(mel_pred_tuple, (list, tuple)):
                    mel_outputs_postnet_v = None
                    for o in mel_pred_tuple:
                        if torch.is_tensor(o) and o.dim() >= 2:
                            mel_outputs_postnet_v = o
                            break
                else:
                    mel_outputs_postnet_v = mel_pred_tuple

                if mel_outputs_postnet_v.dim() == 2:
                    mel_outputs_postnet_v = mel_outputs_postnet_v.unsqueeze(0)

                T_pred_v = mel_outputs_postnet_v.size(2)
                T_target_v = mels_v.size(2)
                T_min_v = min(T_pred_v, T_target_v)
                loss_v = criterion(
                    mel_outputs_postnet_v[:, :, :T_min_v],
                    mels_v[:, :, :T_min_v]
                )
                val_running_loss += float(loss_v.item())
                val_batches += 1

        val_loss = val_running_loss / max(val_batches, 1)
        print(f"[INFO] Validación Epoch {epoch} – MSE mel: {val_loss:.6f}")

        # Ajustar LR scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[INFO] LR actual después de ReduceLROnPlateau: {current_lr:.6e}")

        # Visualización final de la epoch
        with torch.no_grad():
            orig_mel_db = mels[0].detach().cpu()
            pred_mel_db = mel_outputs_postnet[0, :, :T_min].detach().cpu()
            plt.figure(figsize=(12, 5))
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(orig_mel_db.numpy(), origin='lower', aspect='auto')
            ax1.set_title("MEL Original (dB)")
            ax1.set_xlabel("Frames")
            ax1.set_ylabel("Mel bins")
            ax2 = plt.subplot(1, 2, 2)
            ax2.imshow(pred_mel_db.numpy(), origin='lower', aspect='auto')
            ax2.set_title("MEL Predicho (Postnet, dB)")
            ax2.set_xlabel("Frames")
            ax2.set_ylabel("Mel bins")
            plt.suptitle(f"Comparativa MEL - Epoch {epoch:03d} / End")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            comp_path_epoch = os.path.join(
                args.vis_dir,
                f"compare_ep{epoch:03d}_end.png"
            )
            plt.savefig(comp_path_epoch, dpi=150)
            plt.close()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(args.loras_dir, "best_lora.pth")
            lora_state = {k: v.cpu() for k, v in tacotron2.state_dict().items() if ("A." in k or "B." in k)}
            torch.save(lora_state, best_path)
            print(f"[INFO] Nueva mejor validación: {best_val_loss:.6f}. LoRA guardado en: {best_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No mejoró en validación ({epochs_no_improve}/{args.early_stopping_patience})")
            if epochs_no_improve >= args.early_stopping_patience:
                print("[INFO] Early stopping activado. Terminando entrenamiento.")
                break

    print("[INFO] Entrenamiento finalizado.")

    # 11) Evaluation Report
    print("\n" + "="*40 + " EVALUATION REPORT " + "="*40)
    print("[INFO] Cargando mejor LoRA para evaluación final...")
    tacotron2.eval()
    best_lora_state = torch.load(os.path.join(args.loras_dir, "best_lora.pth"), map_location="cpu")
    full_state = tacotron2.state_dict()
    for k in best_lora_state:
        full_state[k] = best_lora_state[k]
    tacotron2.load_state_dict(full_state)

    val_dataset = VoiceDataset(
        data_dir=args.data_dir,
        metadata_filename="val_metadata.csv",
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )
    test_dataset = VoiceDataset(
        data_dir=args.data_dir,
        metadata_filename="test_metadata.csv",
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length
    )

    print("[INFO] Evaluando en VALIDACIÓN...")
    val_metrics = evaluate_model(tacotron2, waveglow, val_dataset, device)
    print(f"→ VALIDATION SET:\n"
          f"   • MSE (mel): {val_metrics['mel_mse']:.6f}\n"
          f"   • MCD aprox: {val_metrics['mcd']:.4f} dB\n"
          f"   • Ejemplos: {val_metrics['n_samples']}")

    print("[INFO] Evaluando en TEST...")
    test_metrics = evaluate_model(tacotron2, waveglow, test_dataset, device)
    print(f"→ TEST SET:\n"
          f"   • MSE (mel): {test_metrics['mel_mse']:.6f}\n"
          f"   • MCD aprox: {test_metrics['mcd']:.4f} dB\n"
          f"   • Ejemplos: {test_metrics['n_samples']}")

    print("="*100 + "\n")


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Cargando Tacotron2 para inferencia...")
    tacotron2 = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_tacotron2',
        model_math='fp16' if device.type == 'cuda' else 'fp32'
    )
    tacotron2 = tacotron2.to(device)
    tacotron2.eval()

    print(f"[INFO] Inyectando LoRA (r={args.lora_rank}, alpha={args.lora_alpha}) en Tacotron2...")
    replace_linear_with_lora(tacotron2, r=args.lora_rank, alpha=args.lora_alpha)

    if not os.path.isfile(args.lora_path):
        raise FileNotFoundError(f"No se encontró el archivo LoRA en {args.lora_path}")
    print(f"[INFO] Cargando pesos LoRA desde {args.lora_path} ...")
    lora_state = torch.load(args.lora_path, map_location="cpu")
    full_state = tacotron2.state_dict()
    for k in lora_state:
        if k in full_state and lora_state[k].shape == full_state[k].shape:
            full_state[k] = lora_state[k]
        else:
            raise RuntimeError(
                f"Shape mismatch para clave '{k}': checkpoint {tuple(lora_state[k].shape)} vs "
                f"modelo {tuple(full_state.get(k, torch.tensor([])).shape)}. "
                "Verifica que --lora_rank y --lora_alpha coincidan con el entrenamiento."
            )
    tacotron2.load_state_dict(full_state)
    tacotron2.eval()

    print("[INFO] Cargando WaveGlow para vocoder...")
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device)
    waveglow.eval()
    for p in waveglow.parameters():
        p.requires_grad = False

    print("[INFO] Cargando utils para preparar texto...")
    global tts_utils
    tts_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    # 1) Preparar secuencia con pipeline oficial NVIDIA
    sequences, lengths = tts_utils.prepare_input_sequence([args.text])
    sequences = sequences.to(device)
    lengths = lengths.to(device)

    # 2) Llamada a tacotron2.infer
    with torch.no_grad():
        mel_tuple = tacotron2.infer(sequences, lengths)
        mel = None
        if isinstance(mel_tuple, (list, tuple)):
            for o in mel_tuple:
                if torch.is_tensor(o) and o.dim() >= 2:
                    mel = o
                    break
        else:
            mel = mel_tuple

    if mel is None:
        raise RuntimeError(f"Salida inesperada de tacotron2.infer: {type(mel_tuple)}")

    # 3) Asegurar forma [B, n_mels, T]
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    elif mel.dim() == 3:
        pass
    else:
        raise RuntimeError(f"Dimensiones inesperadas de mel luego de infer: {mel.shape}")

    mel = mel.float()

    # 4) Generar audio con WaveGlow
    with torch.no_grad():
        try:
            audio = waveglow.infer(mel)
        except RuntimeError:
            audio = waveglow.infer(mel.unsqueeze(1))

    audio = audio.squeeze().cpu()
    if audio.dim() == 2:
        audio = audio[0]

    # 5) Guardar WAV
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = int(time.time())
    out_path = os.path.join(args.output_dir, f"infer_{timestamp}.wav")
    torchaudio.save(out_path, audio.unsqueeze(0), sample_rate=args.sample_rate)
    print(f"[INFO] Archivo WAV generado en: {out_path}")


# --------------------------------------------------------------------------------
# 9) PARSING DE ARGUMENTOS PRINCIPAL
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Script para entrenar LoRA en Tacotron2 y hacer inferencia TTS."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- SUBCOMANDO TRAIN ----------
    train_parser = subparsers.add_parser("train", help="Entrenar LoRA sobre Tacotron2")
    train_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directorio raíz que contiene 'metadata.csv' y carpeta 'wavs/'"
    )
    train_parser.add_argument(
        "--loras_dir", type=str, default="loras",
        help="Directorio donde se guardarán los checkpoints LoRA (.pth)"
    )
    train_parser.add_argument(
        "--vis_dir", type=str, default="visualization",
        help="Directorio donde se guardarán las imágenes de comparación de mels"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100,
        help="Número máximo de épocas de entrenamiento"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Tamaño de batch"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate para optimizador Adam (solo LoRA)"
    )
    train_parser.add_argument(
        "--lora_rank", type=int, default=4,
        help="Rango r para LoRA"
    )
    train_parser.add_argument(
        "--lora_alpha", type=float, default=1.0,
        help="Alpha (factor de escala) para LoRA"
    )
    train_parser.add_argument(
        "--early_stopping_patience", type=int, default=5,
        help="Número de épocas sin mejora para activar early stopping"
    )
    train_parser.add_argument(
        "--vis_interval", type=int, default=200,
        help="Frecuencia (en batches) con la que se guardan visualizaciones MEL"
    )
    train_parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Número de workers para DataLoader"
    )
    train_parser.add_argument(
        "--sample_rate", type=int, default=22050,
        help="Frecuencia de muestreo para los audios"
    )
    train_parser.add_argument(
        "--n_mels", type=int, default=80,
        help="Número de bins Mel"
    )
    train_parser.add_argument(
        "--n_fft", type=int, default=1024,
        help="Tamaño de FFT para espectrograma"
    )
    train_parser.add_argument(
        "--hop_length", type=int, default=256,
        help="Hop length para espectrograma"
    )
    train_parser.add_argument(
        "--win_length", type=int, default=1024,
        help="Window length para espectrograma"
    )
    train_parser.add_argument(
        "--seed", type=int, default=1234,
        help="Semilla para dividir aleatoriamente los datos"
    )
    train_parser.add_argument(
        "--train_frac", type=float, default=0.8,
        help="Fracción de datos para entrenamiento"
    )
    train_parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fracción de datos para validación"
    )
    train_parser.add_argument(
        "--test_frac", type=float, default=0.1,
        help="Fracción de datos para prueba"
    )
    train_parser.add_argument(
        "--lr_factor", type=float, default=0.5,
        help="Factor de reducción de LR en ReduceLROnPlateau"
    )
    train_parser.add_argument(
        "--lr_patience", type=int, default=2,
        help="Paciencia (epochs) para ReduceLROnPlateau"
    )

    # ---------- SUBCOMANDO INFER ----------
    infer_parser = subparsers.add_parser("infer", help="Inferir TTS con Tacotron2 + LoRA + vocoder")
    infer_parser.add_argument(
        "--text", type=str, required=True,
        help="Texto a sintetizar (inglés, con mayúsculas/minúsculas/puntuación)"
    )
    infer_parser.add_argument(
        "--lora_path", type=str, required=True,
        help="Ruta al checkpoint LoRA (.pth) generado en entrenamiento"
    )
    infer_parser.add_argument(
        "--output_dir", type=str, default="samples",
        help="Directorio donde se guardan audios WAV generados"
    )
    infer_parser.add_argument(
        "--lora_rank", type=int, default=4,
        help="Rango r para LoRA (igual que en entrenamiento)"
    )
    infer_parser.add_argument(
        "--lora_alpha", type=float, default=1.0,
        help="Alpha para LoRA (igual que en entrenamiento)"
    )
    infer_parser.add_argument(
        "--sample_rate", type=int, default=22050,
        help="Frecuencia de muestreo para salida"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
