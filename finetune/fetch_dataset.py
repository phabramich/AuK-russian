#!/usr/bin/env python3
"""Fetch public Russian speech corpora -> clips/ + meta.tsv for prepare_jsonl.py.

Profiles:
    trial  — Common Voice 17 ru validation+test only, capped at --max-hours
             (default 20). Enough to prove the pipeline end-to-end.
    sota   — Common Voice 17 ru (all splits) + Golos + Sova, no cap by default.
             Public-data recipe in the spirit of community F5-TTS ru runs
             (~400+ h after cleaning).

Or pick sources yourself:
    python fetch_dataset.py --sources cv-ru golos10 --out data/corpus

Sources are HF datasets; availability/columns vary between mirrors, so column
detection is fuzzy and failures print the actual column list. Common Voice via
fsicoli mirrors does NOT need a gated-repo token.

    pip install datasets soundfile
    python fetch_dataset.py --profile trial --out data/trial_corpus
    python prepare_jsonl.py --meta data/trial_corpus/meta.tsv \
        --clips data/trial_corpus --out data/train.jsonl --val-out data/val.jsonl
"""
import argparse
import csv
import io
import os
import re
import sys

TARGET_SR = 24000
MIN_SEC, MAX_SEC = 0.3, 30.0

# name -> (hf repo, config, splits). Splits may differ per mirror — edit if needed.
SOURCES = {
    "cv-ru":    ("fsicoli/common_voice_17_0", "ru", "train+validation+test"),
    "cv-ru-sm": ("fsicoli/common_voice_17_0", "ru", "validation+test"),
    "golos10":  ("bond005/sberdevices_golos_10h_crowd", None, "train"),
    "golos100": ("bond005/sberdevices_golos_100h_farfield", None, "train"),
    "sova":     ("bond005/sova_rudevices", None, "train"),
}
PROFILES = {
    "trial": ["cv-ru-sm"],
    "sota": ["cv-ru", "golos100", "golos10", "sova"],
}

SPK_CANDIDATES = ("client_id", "speaker_id", "speaker", "spk", "user_id")
TEXT_CANDIDATES = ("sentence", "transcription", "transcript", "text", "answer")

# --- tiny ru number->words (0..999999), enough for corpus normalization ---
_U = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь",
      "девять", "десять", "одиннадцать", "двенадцать", "тринадцать",
      "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
      "восемнадцать", "девятнадцать"]
_T = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
      "семьдесят", "восемьдесят", "девяносто"]
_H = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
      "семьсот", "восемьсот", "девятьсот"]


def _tri(n, fem=False):
    h, r = divmod(n, 100)
    t, u = divmod(r, 10)
    parts = []
    if h:
        parts.append(_H[h])
    if t == 1:
        parts.append(_U[10 + u])
    else:
        if t:
            parts.append(_T[t])
        if u:
            parts.append("одна" if fem and u == 1 else "две" if fem and u == 2 else _U[u])
    return parts


def ru_num(n):
    if n == 0:
        return "ноль"
    th, r = divmod(n, 1000)
    parts = []
    if th:
        parts += _tri(th, fem=True)
        u, t = th % 10, (th % 100) // 10
        parts.append("тысяч" if t == 1 or u == 0 or u > 4
                     else "тысяча" if u == 1 else "тысячи")
    parts += _tri(r)
    return " ".join(parts)


_SYM = {"%": "процентов", "№": "номер", "$": "долларов", "€": "евро",
        "&": "и", "+": "плюс"}
_BAD = re.compile(r"[^\w\s\.\,\!\?\:\;\-\'\"«»—…]+", re.UNICODE)
_DIGITS = re.compile(r"\d+")


def norm_text(t, digits=True):
    for k, v in _SYM.items():
        t = t.replace(k, f" {v} ")
    if digits:
        t = _DIGITS.sub(lambda m: ru_num(int(m.group())) if len(m.group()) <= 6 else m.group(), t)
    t = _BAD.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    return None


def audio_col(ds):
    for k, f in ds.features.items():
        if type(f).__name__ == "Audio":
            return k
    for k in ds.column_names:
        if k in ("audio", "audio_file", "wav", "file"):
            return k
    return None


def emit_wav(out_path, array, sr):
    import numpy as np
    import soundfile as sf
    x = np.asarray(array, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        n = int(round(len(x) * TARGET_SR / sr))
        if n < 1:
            return 0.0
        idx = np.linspace(0, len(x) - 1, n)
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    peak = np.abs(x).max() if len(x) else 0.0
    if peak > 0:
        x = x / peak * 0.95
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, x, TARGET_SR)
    return len(x) / TARGET_SR


def iter_source(name, repo, config, split, cache_dir):
    from datasets import load_dataset
    try:
        return load_dataset(repo, config, split=split, cache_dir=cache_dir,
                            trust_remote_code=True)
    except Exception as e:
        print(f"[{name}] load failed: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None)
    ap.add_argument("--sources", nargs="+", default=None,
                    help=f"any of: {sorted(SOURCES)}")
    ap.add_argument("--out", required=True, help="corpus dir (clips/ + meta.tsv)")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="cap total kept audio (0 = no cap; profile=trial defaults to 20)")
    ap.add_argument("--single-speaker", default=None,
                    help="'auto' = speaker with most data, or an id; keeps only that speaker")
    ap.add_argument("--norm-text", action="store_true", default=True)
    ap.add_argument("--no-norm-text", dest="norm_text", action="store_false")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--max-rows", type=int, default=0, help="debug: cap rows per source")
    a = ap.parse_args()

    sources = a.sources or PROFILES.get(a.profile or "sota")
    max_hours = a.max_hours or (20.0 if a.profile == "trial" else 0.0)
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)

    rows, seen = [], set()
    hours = 0.0
    for name in sources:
        if name not in SOURCES:
            sys.exit(f"unknown source {name}; choices: {sorted(SOURCES)}")
        repo, config, split = SOURCES[name]
        ds = iter_source(name, repo, config, split, a.hf_cache)
        if ds is None:
            continue
        cols = ds.column_names
        c_aud = audio_col(ds)
        c_txt = pick(cols, TEXT_CANDIDATES, "text")
        c_spk = pick(cols, SPK_CANDIDATES, "speaker")
        if not c_aud or not c_txt:
            print(f"[{name}] cannot find audio/text columns in {cols} — skipping; "
                  f"add the right names to TEXT_CANDIDATES/audio_col", file=sys.stderr)
            continue
        print(f"[{name}] {len(ds)} rows | audio={c_aud} text={c_txt} spk={c_spk or '(none)'}")

        kept = 0
        for i, r in enumerate(ds):
            if a.max_rows and i >= a.max_rows:
                break
            try:
                aud = r[c_aud]
                arr, sr = aud["array"], aud["sampling_rate"]
                dur = len(arr) / sr
                if not (MIN_SEC <= dur <= MAX_SEC):
                    continue
                text = r[c_txt] or ""
                text = norm_text(text) if a.norm_text else text.strip()
                if len(text) < 2:
                    continue
                spk = str(r[c_spk]) if c_spk and r.get(c_spk) else f"{name}_anon"
                key = (spk, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                rel = f"clips/{name}/{i:08d}.wav"
                dur = emit_wav(os.path.join(out_dir, rel), arr, sr)
                if dur <= 0:
                    continue
                rows.append((rel, f"{name}:{spk}", text, dur))
                hours += dur / 3600
                kept += 1
                if max_hours and hours >= max_hours:
                    break
            except Exception:
                continue
        print(f"[{name}] kept {kept} rows, total {hours:.1f} h")
        if max_hours and hours >= max_hours:
            print(f"reached --max-hours {max_hours}")
            break

    if a.single_speaker:
        from collections import Counter
        cnt = Counter(spk for _, spk, _, _ in rows)
        want = max(cnt, key=cnt.get) if a.single_speaker == "auto" and cnt else a.single_speaker
        rows = [r for r in rows if r[1] == want]
        print(f"single-speaker {want}: {len(rows)} rows "
              f"({sum(r[3] for r in rows)/3600:.1f} h)")

    meta = os.path.join(out_dir, "meta.tsv")
    with open(meta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "speaker", "sentence"])
        for rel, spk, text, _ in rows:
            w.writerow([rel, spk, text])
    print(f"wrote {meta} | {len(rows)} rows | {sum(r[3] for r in rows)/3600:.1f} h "
          f"| speakers={len(set(r[1] for r in rows))}")


if __name__ == "__main__":
    main()
