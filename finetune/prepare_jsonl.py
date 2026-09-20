#!/usr/bin/env python3
"""Corpus metadata TSV -> AuK train/val JSONL (upstream trainer format).

Input TSV must have one column each for: audio path, speaker id, transcript.
Common Voice validated.tsv works as-is (client_id / path / sentence).

    python prepare_jsonl.py --meta validated.tsv --clips cv/corpus/clips \
        --out data/train.jsonl --val-out data/val.jsonl

Each sample pairs the target clip with a *different* clip of the same speaker
(voice-clone conditioning). Speakers with a single utterance, and --no-ref mode,
produce text-only samples instead.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

MIN_SEC, MAX_SEC = 0.3, 30.0
INSTR = 'Say the following with the same voice: "{text}"'

PATH_KEYS = ("path", "file", "audio", "audio_url", "utt", "filename")
SPK_KEYS = ("client_id", "speaker", "speaker_id", "spk", "speakerID")
TEXT_KEYS = ("sentence", "text", "transcript", "transcription")


def pick_col(header, keys, override, name):
    if override:
        if override not in header:
            sys.exit(f"--{name}-col {override!r} not in header {header}")
        return override
    for k in keys:
        if k in header:
            return k
    sys.exit(f"no {name} column in header {header}; use --{name}-col")


def probe_seconds(path):
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        pass
    if path.lower().endswith(".wav"):
        import wave
        try:
            with wave.open(path, "rb") as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            pass
    try:
        import torchaudio
        return torchaudio.info(path).num_frames / torchaudio.info(path).sample_rate
    except Exception:
        return -1.0


def sample_with_ref(text, ref_path, tgt_path, dur):
    return {
        "duration": round(dur, 2),
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": INSTR.format(text=text.replace('"', "'"))},
                {"type": "audio", "audio_url": ref_path},
            ]},
            {"role": "assistant", "content": [
                {"type": "audio", "audio_url": tgt_path},
            ]},
        ],
    }


def sample_no_ref(text, tgt_path, dur):
    return {
        "duration": round(dur, 2),
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": text},
            ]},
            {"role": "assistant", "content": [
                {"type": "audio", "audio_url": tgt_path},
            ]},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="metadata TSV (with header)")
    ap.add_argument("--clips", default="", help="dir prepended to relative audio paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--path-col", default=None)
    ap.add_argument("--spk-col", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--no-ref", action="store_true", help="text-only samples (no voice conditioning)")
    ap.add_argument("--val-by-speaker", action="store_true",
                    help="hold out whole speakers for val (no speaker leakage)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    with open(a.meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit("empty meta")
    hdr = list(rows[0].keys())
    c_path = pick_col(hdr, PATH_KEYS, a.path_col, "path")
    c_spk = pick_col(hdr, SPK_KEYS, a.spk_col, "spk")
    c_text = pick_col(hdr, TEXT_KEYS, a.text_col, "text")

    by_spk = defaultdict(list)
    skipped = defaultdict(int)
    for r in rows:
        p = os.path.join(a.clips, r[c_path]) if a.clips and not os.path.isabs(r[c_path]) else r[c_path]
        if not os.path.isfile(p):
            skipped["missing_file"] += 1
            continue
        dur = probe_seconds(p)
        if not (MIN_SEC <= dur <= MAX_SEC):
            skipped["duration"] += 1
            continue
        text = r[c_text].strip()
        if not text:
            skipped["empty_text"] += 1
            continue
        by_spk[r[c_spk]].append((p, text, dur))

    samples = []  # (speaker, sample)
    for spk, utts in by_spk.items():
        for p, text, dur in utts:
            others = [u for u in utts if u[0] != p]
            if a.no_ref or not others:
                s = sample_no_ref(text, p, dur)
            else:
                s = sample_with_ref(text, random.choice(others)[0], p, dur)
            samples.append((spk, s))

    random.shuffle(samples)
    if not a.val_out:
        val, train = [], [s for _, s in samples]
    elif a.val_by_speaker:
        spk_order = list(by_spk)
        random.shuffle(spk_order)
        val_spks, n_val = set(), 0
        for spk in spk_order:
            if n_val >= int(len(samples) * a.val_frac):
                break
            val_spks.add(spk)
            n_val += len(by_spk[spk])
        val = [s for spk, s in samples if spk in val_spks]
        train = [s for spk, s in samples if spk not in val_spks]
    else:
        n_val = int(len(samples) * a.val_frac)
        val = [s for _, s in samples[:n_val]]
        train = [s for _, s in samples[n_val:]]

    def dump(path, items):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    dump(a.out, train)
    if a.val_out:
        dump(a.val_out, val)

    hrs = sum(s["duration"] for _, s in samples) / 3600
    print(f"speakers={len(by_spk)} samples={len(samples)} ({len(train)} train / {len(val)} val) "
          f"hours={hrs:.1f} skipped={dict(skipped)}")


if __name__ == "__main__":
    main()
