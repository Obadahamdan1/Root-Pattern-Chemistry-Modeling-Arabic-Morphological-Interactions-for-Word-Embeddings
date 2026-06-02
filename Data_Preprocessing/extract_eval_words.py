import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _strip_root(root: str) -> str:
    return root.replace(".", "").replace("-", "").strip()


def is_clean(tok: dict) -> bool:
    root_raw = tok.get("root", "") or ""
    pattern  = (tok.get("lexeme_pattern")
                or tok.get("surface_pattern")
                or tok.get("pattern", "")) or ""
    pos      = tok.get("pos", "") or ""
    root     = _strip_root(root_raw)

    if not root or not pattern or not pos:
        return False
    if root.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
        return False
    if pattern.upper() in ("NTWS", "FOREIGN", "NO_ANALYSIS"):
        return False
    if len(root) not in (3, 4):
        return False
    if "#" in root or "#" in root_raw or "#" in pattern:
        return False
    if pos in ("punc", "digit"):
        return False
    return True


def extract(data_path: str, output_path: str, max_sentences: int) -> None:
    seen_words  = set()
    total_sents = 0
    total_toks  = 0
    written     = 0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(data_path, encoding="utf-8") as fin, \
         open(out, "w", encoding="utf-8") as fout:

        for line in fin:
            if total_sents >= max_sentences:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            tokens = obj.get("tokens", [])
            if not tokens:
                continue

            total_sents += 1

            for tok in tokens:
                total_toks += 1
                if not is_clean(tok):
                    continue

                word = tok.get("word", tok.get("surface", "")) or ""
                if not word or word in seen_words:
                    continue

                seen_words.add(word)

                root_raw = tok.get("root", "") or ""
                record = {
                    "word":    word,
                    "root":    _strip_root(root_raw),   # store stripped
                    "pattern": (tok.get("lexeme_pattern")
                                or tok.get("surface_pattern")
                                or tok.get("pattern", "") or ""),
                    "prefix":  "".join(tok.get("prefixes", [])),
                    "suffix":  "".join(tok.get("suffixes", [])),
                    "pos":     tok.get("pos", "") or "",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    logger.info("Sentences scanned : %d", total_sents)
    logger.info("Tokens seen       : %d", total_toks)
    logger.info("Unique words written: %d", written)
    logger.info("Output -> %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",     required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--max_sentences", type=int, default=50_000)
    args = parser.parse_args()

    extract(args.data_path, args.output, args.max_sentences)
