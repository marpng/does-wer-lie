# Project diary

Running log of what was done, found and decided, as raw material for the final report.
Newest entry at the bottom.

---

## 12.10.2026

### Done

- Created the private GitHub repo `marpng/does-wer-lie` with README, `requirements.txt`
  (evaluation pipeline), `requirements-finetune.txt` (LoRA, for Colab) and `.gitignore`
  (no audio or data is committed).
- Wrote `src/prepare_fleurs.py`: downloads FLEURS from Hugging Face (`google/fleurs`) and writes one
  manifest per language in the shared format (`id, language, dataset, split, audio_path, reference,
  reference_raw, duration, sentence_id, gender, dialect`). Works for any FLEURS language code, so
  it can be reused for Italian and German.
- Downloaded FLEURS English (`en_us`) and Greek (`el_gr`): dev and test audio, transcripts for all
  three splits. Training audio not downloaded yet.
- Wrote `src/explore_data.py` and ran it. Output is in `reports/data_exploration/`
  (`summary.md`, `overview.csv`, `audio_health.csv`, `distributions_test.png`).

### What the data looks like

Test split:

| | English | Greek |
|---|---|---|
| Recordings | 647 | 650 |
| Unique sentences | 350 | 333 |
| Hours | 1.77 | 1.91 |
| Reference words | 14,344 | 15,106 |
| Median duration | 9.5 s | 10.1 s |
| Duration, 5th–95th percentile | 5.2–16.5 s | 5.8–16.8 s |
| Median words per utterance | 21 | 22 |
| Median speaking rate | 2.3 words/s | 2.2 words/s |
| Speakers female / male (recordings) | 412 / 235 | 325 / 325 |
| Utterances with digits | 18.9% | 20.0% |
| Utterances with quotes or brackets | 16.5% | 15.1% |
| Utterances with hyphen or apostrophe inside a word | 23.3% | 3.5% |
| Utterances with words in another script | 0% | 9.2% |

All splits:

| Language | Split | Recordings | Unique sentences | Hours |
|---|---|---|---|---|
| English | train | 2602 | 1476 | 7.49 |
| English | dev | 394 | 150 | 1.05 |
| English | test | 647 | 350 | 1.77 |
| Greek | train | 3215 | 1496 | 10.04 |
| Greek | dev | 271 | 140 | 0.77 |
| Greek | test | 650 | 333 | 1.91 |

### Findings

- **Fewer distinct utterances than planned.** Each sentence is read by one to three speakers
  (EN: 53 sentences once, 297 twice; EL: 100 once, 149 twice, 84 three times). The plan of 500
  evaluation utterances per language is only reachable by repeating sentences. The realistic
  evaluation set is about 330 unique sentences per language.
- **The sentences are parallel across languages.** All 333 Greek test sentences also exist in the
  English test set (same `sentence_id`). Italian and German should share them too; not checked yet.
- **Utterances are very uniform in length** and similar in the two languages (see table), so the
  number of utterances, the audio time and the number of reference words are almost
  interchangeable: one recording per unique sentence gives 58.1 min / 7,788 words in English and
  60.5 min / 7,938 words in Greek.
- **English audio is recorded at very low gain.** 451 of 647 English test recordings (70%) have an
  RMS level below −50 dBFS (median peak amplitude 0.009 of full scale). This affects every female
  speaker and about 17% of male speakers, so gain is confounded with gender. Greek has no such
  files. Speech is present in all of them (no empty files in either language).
- **Low gain does not hurt Whisper.** Quick check with Whisper small on 40 quiet and 40 normal
  English test recordings: mean WER 9.1% vs 7.4%; after peak-normalising the audio 8.6% vs 7.8%.
  Normalising gain makes no real difference for the model. It does matter for human raters, who
  would barely hear the quiet files.
- **The FLEURS "normalised" transcript is not fully normalised.** It still contains digits (about
  20% of utterances), guillemets «», periods and colons, and in Greek some Latin look-alike
  letters (Latin "H" instead of Greek "Η"). 17% of Greek and 31% of English normalised test
  references still contain punctuation.
- **Greek contains Latin-script words** in 9.2% of test utterances, mostly names and acronyms
  (Aerosmith, Sahel, SANParks). Whisper may transliterate these, which WER counts as errors.
- **Technically clean.** All audio is 16 kHz mono, durations match the metadata, no sentence
  appears in more than one split (relevant for fine-tuning: no train/test leakage by sentence).

### Discussion: utterances or time as the unit?

Utterances, counted as unique sentences. In FLEURS the two units give nearly the same amount of
data per language, and the utterance is the unit that gets rated and on which the metric–human
correlation is computed, so a time budget adds nothing. Time only becomes relevant for Swiss
German (STT4SG-350), whose utterances may differ in length: there the idea is to still sample by
utterance but restrict to a duration range similar to FLEURS (about 5–17 s) and report hours
alongside.

### Ideas for the design (not decided yet, to discuss with Linda)

1. Use the same sentences for EN/IT/DE/EL: take the sentence IDs common to all four languages and
   one recording per sentence. Content is then held constant, so differences in the WER–human
   correlation reflect the language and not the topic. Swiss German needs its own sample.
2. Rate the same 80 sentences in those four languages, stratified by WER averaged across languages.
3. Compute correlations at utterance level with bootstrap confidence intervals; keep corpus-level
   WER only for the baseline table.
4. Normalise reference and hypothesis with our own function, starting from the raw reference, and
   not from the FLEURS normalised column.
5. Treat digits and foreign-script words as error categories from the start.
6. Peak-normalise audio for the rating sheet, and balance gender when choosing one recording per
   sentence in English.

### Open points

- Add Linda as collaborator on the repo.
- Common Voice Greek (only needed for fine-tuning) not downloaded; needs a Mozilla login.
- Phase 0 decisions still to confirm with Linda and Sina: size of the evaluation set (500 is not
  available as distinct sentences), rated-set size, agreement method.
