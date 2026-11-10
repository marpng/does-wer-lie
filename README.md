# Does WER Lie? Auditing Whisper Across Five Languages with Human Evaluation

Course project for Essentials in Text and Speech Processing (UZH, Fall 2026)
Linda Mazzone and Maria Christina Panagiotopoulou

Word Error Rate counts every differing word as an equal mistake
We test whether it agrees with human judgments of Whisper transcripts in five languages
English, Italian, German, Greek and Swiss German

## Research questions

- Q1 How well does WER correlate with human quality judgments in each language
- Q2 Does the correlation get worse down the resource ladder, especially for Swiss German
- Q3 After LoRA fine-tuning a small Whisper, does the WER gain hold up or is it partly cosmetic

For Q3 we fine-tune on Greek and compare WER and CER before and after under different normalisations
There is no second human rating round for Q3

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # evaluation pipeline
pip install -r requirements-finetune.txt   # only for fine-tuning
```

Python 3.10 or newer
Fine-tuning needs a GPU, we used a Colab T4 (see `notebooks/`)

## Scripts

All scripts are in `src/` and run from the repository root

| Script | What it does |
|---|---|
| `prepare_fleurs.py` | downloads FLEURS and writes one manifest per language |
| `explore_data.py` | writes the data exploration report |
| `select_eval_set.py` | picks the evaluation set, one recording per test sentence |
| `transcribe.py` | transcribes the evaluation set with Whisper small, medium and large-v3 or with a fine-tuned model |
| `normalise.py` | text normalisers used for scoring (raw, standard, standard_en, content) |
| `compute_metrics.py` | WER and CER per utterance and as tables |
| `sample_rated_set.py` | samples the items humans rate, stratified by WER |
| `build_rating_sheet.py` | builds an offline HTML rating sheet with audio |
| `analyse_ratings.py` | rater agreement and correlation of human scores with WER and CER (Q1 and Q2) |
| `finetune_lora.py` | LoRA fine-tunes Whisper small on FLEURS train and exports a faster-whisper model |
| `analyse_finetuning.py` | WER and CER gain from fine-tuning with bootstrap intervals (Q3) |
| `analyse_error_types.py` | which error types fine-tuning removed (Q3) |

`rating_sheet_template.html` is the page used by `build_rating_sheet.py`

## Running

```bash
# 1 data and manifests
python src/prepare_fleurs.py --langs en_us el_gr
python src/explore_data.py

# 2 fixed evaluation set per language
python src/select_eval_set.py --manifest data/manifests/fleurs_en_us.csv
python src/select_eval_set.py --manifest data/manifests/fleurs_el_gr.csv

# 3 zero-shot transcription and metrics
python src/transcribe.py --manifest data/manifests/fleurs_en_us.csv
python src/transcribe.py --manifest data/manifests/fleurs_el_gr.csv
python src/compute_metrics.py

# 4 human rating (run the sampling once, the file is frozen)
python src/sample_rated_set.py --parallel en it de el --overlap-language en --overlap-size 30
python src/build_rating_sheet.py --language el
python src/build_rating_sheet.py --language en --overlap-only
python src/analyse_ratings.py

# 5 fine-tuning (Q3)
python src/prepare_fleurs.py --langs el_gr --splits train dev test
python src/finetune_lora.py --manifest data/manifests/fleurs_el_gr.csv --name small-lora-el
python src/transcribe.py --manifest data/manifests/fleurs_el_gr.csv --models small-lora-el=models/small-lora-el/ct2 --out-dir results/finetuned
python src/compute_metrics.py --results-dir results/finetuned --name finetuned
python src/analyse_finetuning.py --language el --tuned small-lora-el
python src/analyse_error_types.py --language el --baseline small --tuned small-lora-el
```

Decoding uses beam size 5, no VAD and no conditioning on previous text
The `compute_type` column of each output file stores the precision used

## Fine-tuning runs

All runs use Whisper small, FLEURS Greek train (3,215 recordings, 10 h) and the same seed
Sentences from the evaluation set, rated set, dev and test splits are never trained on

| Run | LoRA layers | Epochs | Learning rate | SpecAugment | Notebook |
|---|---|---|---|---|---|
| `small-lora-el` | q and v | 3 | 5e-4 | no | `finetune_colab.ipynb` |
| `small-lora-el-all` | all linear | 3 | 5e-4 | no | `finetune_colab_all_linear.ipynb` |
| `small-lora-el-reg` | q and v | 2 | 2e-4 | yes | `finetune_colab_sensitivity.ipynb` |

Summary of all runs is in `results/metrics/finetuning_summary.md`

## Repository layout

```
src/                       scripts
notebooks/                 Colab notebooks for the three fine-tuning runs
eval_sets/                 IDs of the evaluation utterances per language
reports/data_exploration/  data exploration report
results/baseline/          zero-shot transcripts, one file per model and language
results/finetuned/         transcripts of the fine-tuned models
results/metrics/           per-utterance metrics and result tables
results/analysis/          Q1 and Q2 agreement and correlations for all languages
results/analysis_q3/       Q3 gains and error types for small-lora-el
results/analysis_q3_all/   same for small-lora-el-all
results/analysis_q3_reg/   same for small-lora-el-reg
ratings/rated_set_ids.csv  frozen list of rated items
ratings/scores/           downloaded human ratings, one file per language and rater
data/                      not committed, audio and manifests
models/                    not committed, adapters and converted models
ratings/sheets/            not committed, rating sheets with audio
```

## Manifest format

One CSV per language in `data/manifests/` with 16 kHz mono audio

| column | meaning |
|---|---|
| `id` | utterance ID as dataset_lang_split_recording |
| `language` | en, it, de, el, gsw |
| `dataset`, `split` | source corpus and split |
| `audio_path` | path relative to the repo root, empty if the audio was not downloaded |
| `reference_raw` | reference as published with case and punctuation |
| `reference` | the corpus own lowercased version |
| `duration` | seconds |
| `sentence_id` | ID of the read sentence, shared across FLEURS languages |
| `gender`, `dialect` | speaker metadata where available |

## Data notes

Full numbers are in `reports/data_exploration/summary.md`

- Test sets have about 650 recordings but only 350 (EN) and 333 (EL) unique sentences
- Utterances are uniform, median 9.5 s and 21 words (EN), 10.1 s and 22 words (EL)
- About 70% of English recordings are very quiet, so audio is peak normalised for raters
- The FLEURS normalised transcript still has digits and punctuation, so always start from `reference_raw`
- About 20% of utterances contain digits and 9% of Greek ones contain Latin words, which cause WER errors that are not meaning errors

## Data licences

FLEURS is CC BY 4.0
STT4SG-350 and SDS-200 need a signed licence and must not be redistributed
For Swiss German this repository only holds IDs, model outputs and ratings, never audio or reference text
