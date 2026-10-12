# Does WER Lie? Auditing Whisper Across Five Languages with Human Evaluation

Course project for *Essentials in Text and Speech Processing* (UZH, Fall 2026).
Linda Mazzone and Maria Christina Panagiotopoulou.

Word Error Rate treats every differing word as an equal mistake. We check whether it agrees with
human-perceived transcription quality for OpenAI's Whisper on five languages that form a difficulty
gradient: English, Italian, German, Greek and Swiss German.

## Research questions

- **Q1** How well does WER correlate with human quality judgments in each language?
- **Q2** Does that correlation degrade down the resource ladder, especially for Swiss German, whose
  references are Standard German translations rather than literal transcripts?
- **Q3** After LoRA fine-tuning a small Whisper on the weakest languages, does the WER gain match a
  human-perceived gain, or is it partly cosmetic?

## Repository layout

```
src/                      pipeline scripts
  prepare_fleurs.py       download FLEURS + write manifests
  explore_data.py         data exploration report
reports/data_exploration/ generated report (tables, figure, per-file audio checks)
data/                     NOT committed: raw audio and manifests
results/                  model outputs and metric tables (to come)
ratings/                  frozen rated-set IDs and human scores (to come)
guidelines/               rating guidelines (to come)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # evaluation pipeline
pip install -r requirements-finetune.txt   # only for LoRA fine-tuning (Colab)
```

## Running

```bash
# 1. Download data and build manifests (dev + test audio, transcripts for all splits)
python src/prepare_fleurs.py --langs en_us el_gr          # Maritina
python src/prepare_fleurs.py --langs it_it de_de          # Linda

# 2. Explore whatever manifests are present
python src/explore_data.py
```

`prepare_fleurs.py --splits train dev test` also fetches the training audio (about 1.5–2 GB per language).

## Manifest format

Every dataset is converted to one CSV per language in `data/manifests/`, with 16 kHz mono audio:

| column | meaning |
|---|---|
| `id` | unique utterance ID, `{dataset}_{lang}_{split}_{recording}` |
| `language` | `en`, `it`, `de`, `el`, `gsw` |
| `dataset`, `split` | source corpus and its split |
| `audio_path` | path relative to the repo root (empty if the audio was not downloaded) |
| `reference_raw` | reference as published: cased and punctuated |
| `reference` | the corpus' own lowercased version (not fully normalised, see below) |
| `duration` | seconds |
| `sentence_id` | ID of the read sentence; shared across FLEURS languages |
| `gender`, `dialect` | speaker metadata where available |

## Data notes (FLEURS English and Greek)

Full numbers are in [`reports/data_exploration/summary.md`](reports/data_exploration/summary.md).

- The test sets hold about 650 recordings but only 350 (EN) / 333 (EL) **unique sentences**; most
  sentences are read by two or three speakers. All 333 Greek test sentences also exist in English.
- Utterances are very uniform: median 9.5 s / 21 words (EN) and 10.1 s / 22 words (EL), so a fixed
  number of sentences gives nearly the same audio time and word count in both languages.
- About 70% of the English recordings (all female speakers) are recorded at very low gain. Whisper
  is unaffected, but audio must be peak-normalised before it is played to raters.
- FLEURS' own "normalised" transcript still contains digits, quotes and some punctuation. Always
  normalise reference and hypothesis with the same function, starting from `reference_raw`.
- Around 20% of utterances contain digits and 9% of Greek utterances contain Latin-script words,
  both of which produce WER errors that are not meaning errors.

## Data licences

FLEURS is CC BY 4.0. STT4SG-350 and SDS-200 require a signed licence and must not be redistributed:
this repository only ever contains IDs, model outputs and ratings for Swiss German, never audio or
reference transcripts.
