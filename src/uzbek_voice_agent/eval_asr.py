"""Score NavAI Whisper-small on a prefix of a public Uzbek test split.

This command never trains and refuses any split other than ``test``.
The default slice is the first N rows of ``google/fleurs`` ``uz_uz`` ``test``
in dataset order. N is recorded in the JSON result. It is not the full
FLEURS test set and it is not NavAI's published macro WER.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from uzbek_voice_agent.asr import GENERATE_KWARGS, UzbekASR
from uzbek_voice_agent.metrics import score_utterances
from uzbek_voice_agent.normalize import normalization_backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WER/CER on a public Uzbek test slice")
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--config", default="uz_uz")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", default="eval/fleurs_uz_prefix.json")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _reference_text(row: dict[str, Any]) -> str:
    for key in ("transcription", "raw_transcription", "sentence", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("row has no transcription field")


def _audio_array(row: dict[str, Any]) -> tuple[Any, int]:
    audio = row["audio"]
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        return audio["array"], int(audio["sampling_rate"])
    raise TypeError("expected datasets Audio decoding with array and sampling_rate")


def load_test_slice(dataset: str, config: str, split: str, limit: int) -> list[dict[str, Any]]:
    if split != "test":
        raise SystemExit(
            f"refusing split {split!r}; eval reads the test split only and never trains"
        )
    if limit < 1:
        raise SystemExit("--limit must be at least 1")
    from datasets import Audio, load_dataset

    loaded = load_dataset(dataset, config, split=split, streaming=True)
    loaded = loaded.cast_column("audio", Audio(decode=True))
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(loaded):
        if index >= limit:
            break
        rows.append(row)
    if len(rows) < limit:
        raise SystemExit(f"dataset yielded {len(rows)} rows, fewer than --limit {limit}")
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_test_slice(args.dataset, args.config, args.split, args.limit)
    asr = UzbekASR(device=args.device)
    references: list[str] = []
    hypotheses: list[str] = []
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        reference = _reference_text(row)
        samples, sample_rate = _audio_array(row)
        hypothesis = asr.transcribe_array(samples, sample_rate)
        references.append(reference)
        hypotheses.append(hypothesis)
        items.append(
            {
                "index": index,
                "id": row.get("id"),
                "reference": reference,
                "hypothesis": hypothesis,
            }
        )
        print(f"[{index + 1}/{len(rows)}] ref={reference!r} hyp={hypothesis!r}", flush=True)

    rates = score_utterances(references, hypotheses)
    return {
        "model": asr.model_id,
        "device": asr.device,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "n": rates.n_utterances,
        "requested_n": args.limit,
        "selection": (
            f"first {args.limit} rows of {args.dataset} {args.config} {args.split} "
            "in Hugging Face streaming order"
        ),
        "trained_on_this_split": False,
        "decode": GENERATE_KWARGS,
        "normalizer": normalization_backend(),
        "wer": rates.wer,
        "cer": rates.cer,
        "wer_percent": round(rates.wer * 100, 2),
        "cer_percent": round(rates.cer * 100, 2),
        "n_ref_words": rates.n_ref_words,
        "n_ref_chars": rates.n_ref_chars,
        "skipped_empty_reference": rates.skipped_empty_reference,
        "items": items,
        "note": (
            "Subset metric only. Not comparable to NavAI's full-set FLEURS WER "
            "unless n covers the entire test split and the normalizer matches theirs."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.split != "test":
        raise SystemExit(
            f"refusing split {args.split!r}; eval reads the test split only and never trains"
        )
    report = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"n={report['n']} WER={report['wer_percent']}% CER={report['cer_percent']}% "
        f"normalizer={report['normalizer']} wrote {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
