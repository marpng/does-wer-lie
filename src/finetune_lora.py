"""LoRA fine-tune Whisper on the FLEURS train split of one language and export it for faster-whisper

    python src/finetune_lora.py --manifest data/manifests/fleurs_el_gr.csv --name small-lora-el
    python src/finetune_lora.py --manifest data/manifests/fleurs_el_gr.csv --name smoke --max-train 32 --epochs 1

Uses CUDA or Apple MPS if available and the CPU otherwise
Sentences in the evaluation set, the rated set or the dev and test splits are never trained on
--spec-augment masks parts of the input while training
--lora-targets sets which layers get adapters (default q_proj and v_proj)
Targets are the references as published with case and punctuation so a gain cannot come from lowercasing
Use --target reference for the corpus lowercased version

Writes models/{name}/ (not committed) with adapter/, merged/, ct2/, training_config.json and training_log.csv
Then transcribe with the same decoder as the baseline

    python src/transcribe.py --manifest data/manifests/fleurs_el_gr.csv --models {name}=models/{name}/ct2 \
        --out-dir results/finetuned
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from normalise import clean, normalisers

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval_sets"
RATED_IDS = ROOT / "ratings" / "rated_set_ids.csv"
SAMPLE_RATE = 16_000
MAX_LABEL_TOKENS = 448


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def held_out_sentences(manifest: pd.DataFrame, language: str) -> set[str]:
    """Sentence IDs never trained on: evaluation set, rated set and all dev and test sentences"""
    held = set(manifest.loc[manifest["split"] != "train", "sentence_id"].astype(str))
    eval_path = EVAL_DIR / f"{language}.csv"
    if eval_path.exists():
        held |= set(pd.read_csv(eval_path, dtype=str)["sentence_id"])
    if RATED_IDS.exists():
        rated = pd.read_csv(RATED_IDS, dtype=str)
        held |= set(rated.loc[rated["language"] == language, "sentence_id"])
    return held


class Utterances(Dataset):
    def __init__(self, rows: pd.DataFrame, processor, text_column: str, whisper_language: str):
        self.rows = rows.reset_index(drop=True)
        self.processor = processor
        self.text_column = text_column
        processor.tokenizer.set_prefix_tokens(language=whisper_language, task="transcribe")
        self.start_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows.iloc[i]
        audio, sample_rate = sf.read(ROOT / row["audio_path"], dtype="float32")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"{row['audio_path']} is not 16 kHz mono.")
        features = self.processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features[0]
        labels = self.processor.tokenizer(row[self.text_column].strip()).input_ids
        if labels[0] == self.start_token:  # the model adds it when shifting the labels
            labels = labels[1:]
        return {"input_features": features, "labels": labels[:MAX_LABEL_TOKENS], "id": row["id"], "reference": row["reference_raw"]}


def collate(batch: list[dict]) -> dict:
    longest = -(-max(len(b["labels"]) for b in batch) // 16) * 16  # few distinct shapes keep GPU memory from growing
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
    return {
        "input_features": torch.tensor(np.stack([b["input_features"] for b in batch])),
        "labels": labels,
        "references": [b["reference"] for b in batch],
    }


@torch.no_grad()
def validation_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        out = model(input_features=batch["input_features"].to(device), labels=batch["labels"].to(device))
        n = int((batch["labels"] != -100).sum())
        total += out.loss.item() * n
        count += n
    return total / count


@torch.no_grad()
def dev_wer(model, processor, loader: DataLoader, device: torch.device, whisper_language: str, language: str) -> float:
    """Greedy WER on a few dev utterances with standard normalisation as a quick check"""
    import jiwer

    model.eval()
    normalise = normalisers(language)["standard"]
    references, hypotheses = [], []
    for batch in loader:
        ids = model.generate(input_features=batch["input_features"].to(device), language=whisper_language, task="transcribe",
                             num_beams=1, max_new_tokens=200, use_cache=True)
        texts = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        references += [normalise(clean(r)).strip() for r in batch["references"]]
        hypotheses += [normalise(clean(t)).strip() for t in texts]
    keep = [i for i, r in enumerate(references) if r]
    return jiwer.wer([references[i] for i in keep], [hypotheses[i] for i in keep])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--name", required=True, help="output folder name under models/")
    parser.add_argument("--base", default="openai/whisper-small")
    parser.add_argument("--whisper-language", help="default is the manifest language, use de for Swiss German")
    parser.add_argument("--target", default="reference_raw", choices=["reference_raw", "reference"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="per forward pass; 8 needs about 40 GB on the Apple GPU")
    parser.add_argument("--grad-accum", type=int, default=4, help="batch-size x grad-accum is the effective batch")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,v_proj",
                        help="comma separated layer names, q_proj,k_proj,v_proj,out_proj,fc1,fc2 is all linear layers")
    parser.add_argument("--spec-augment", action="store_true", help="mask random time and mel-frequency blocks of the input while training")
    parser.add_argument("--mask-time-prob", type=float, default=0.05)
    parser.add_argument("--mask-feature-prob", type=float, default=0.05)
    parser.add_argument("--max-train", type=int, help="use only this many training utterances (smoke test)")
    parser.add_argument("--dev-wer-items", type=int, default=64, help="dev utterances for the per-epoch WER check (0 = skip)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="cuda, mps or cpu, default is the best available")
    args = parser.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    import peft
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    out_dir = ROOT / "models" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, keep_default_na=False, dtype={"sentence_id": str})
    language = manifest["language"].iloc[0]
    whisper_language = args.whisper_language or language
    held = held_out_sentences(manifest, language)

    train = manifest[(manifest["split"] == "train") & (manifest["audio_path"] != "")]
    leaked = train["sentence_id"].isin(held)
    print(f"train utterances with audio: {len(train)}, dropped because their sentence is held out: {int(leaked.sum())}")
    train = train[~leaked]
    if args.max_train:
        train = train.iloc[rng.permutation(len(train))[: args.max_train]]
    dev = manifest[(manifest["split"] == "dev") & (manifest["audio_path"] != "")]
    if train.empty or dev.empty:
        raise SystemExit("No train or dev audio. Run: python src/prepare_fleurs.py --langs <lang> --splits train dev test")
    hours = train["duration"].astype(float).sum() / 3600
    print(f"training on {len(train)} utterances ({hours:.1f} h), validating on {len(dev)} dev utterances, device {device}")

    processor = WhisperProcessor.from_pretrained(args.base)
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    if args.spec_augment:  # training only so dev loss and decoding see unmasked input
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
        model.config.mask_feature_prob = args.mask_feature_prob
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=args.lora_targets.split(","), bias="none"))
    model.print_trainable_parameters()
    model.to(device)

    train_set = Utterances(train, processor, args.target, whisper_language)
    dev_set = Utterances(dev, processor, args.target, whisper_language)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.workers, generator=generator)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers)
    wer_rows = dev.iloc[rng.permutation(len(dev))[: args.dev_wer_items]] if args.dev_wer_items else dev.iloc[:0]
    wer_loader = DataLoader(Utterances(wer_rows, processor, args.target, whisper_language), batch_size=args.batch_size, collate_fn=collate)

    steps_per_epoch = -(-len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min((s + 1) / args.warmup_steps, max(0.0, (total_steps - s) / max(1, total_steps - args.warmup_steps)))
    )
    use_cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_cuda_amp)

    log = []
    if len(wer_loader.dataset):
        start_wer = dev_wer(model, processor, wer_loader, device, whisper_language, language)
        print(f"before training: dev WER (greedy, standard) {start_wer:.3f} on {len(wer_loader.dataset)} utterances")
        log.append({"epoch": 0, "step": 0, "train_loss": None, "val_loss": validation_loss(model, dev_loader, device), "dev_wer": start_wer})

    step, started = 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for i, batch in enumerate(train_loader, start=1):
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda_amp):
                loss = model(input_features=batch["input_features"].to(device), labels=batch["labels"].to(device)).loss
            scaler.scale(loss / args.grad_accum).backward()
            running, seen = running + loss.item(), seen + 1
            if i % args.grad_accum == 0 or i == len(train_loader):
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                schedule.step()
                step += 1
                if device.type == "mps":
                    torch.mps.empty_cache()
                if step % 10 == 0:
                    elapsed = time.perf_counter() - started
                    memory = f", {torch.mps.driver_allocated_memory() / 2**30:.0f} GB" if device.type == "mps" else ""
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running / seen:.3f} lr {schedule.get_last_lr()[0]:.2e} "
                          f"{elapsed / 60:.1f} min elapsed, ~{elapsed / step * (total_steps - step) / 60:.1f} min left{memory}", flush=True)
        entry = {"epoch": epoch, "step": step, "train_loss": running / seen, "val_loss": validation_loss(model, dev_loader, device), "dev_wer": None}
        if len(wer_loader.dataset):
            entry["dev_wer"] = dev_wer(model, processor, wer_loader, device, whisper_language, language)
        log.append(entry)
        print(f"== epoch {epoch}: train loss {entry['train_loss']:.3f}, dev loss {entry['val_loss']:.3f}, dev WER {entry['dev_wer']}", flush=True)
        pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)

    model.save_pretrained(out_dir / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out_dir / "merged")
    processor.save_pretrained(out_dir / "merged")

    import ctranslate2.converters

    ctranslate2.converters.TransformersConverter(
        str(out_dir / "merged"), copy_files=["tokenizer.json"]
    ).convert(str(out_dir / "ct2"), force=True)  # keeps float32 weights like the CPU baseline

    config = {**vars(args), "manifest": str(args.manifest), "language": language, "whisper_language": whisper_language, "device": str(device),
              "train_utterances": len(train), "train_hours": round(hours, 2), "dropped_held_out": int(leaked.sum()),
              "train_minutes": round((time.perf_counter() - started) / 60, 1), "versions": {
                  "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}}
    (out_dir / "training_config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"saved to {out_dir.relative_to(ROOT)} (adapter/, merged/, ct2/)")


if __name__ == "__main__":
    main()
