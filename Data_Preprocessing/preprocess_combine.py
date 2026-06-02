import re
import unicodedata

WIKI_PATH   = "data/wiki_1m.txt"
MC4_PATH    = "data/cc100_500k.txt"
OUTPUT_PATH = "data/combined_2m.txt"
TARGET     = 2_000_000  # Stop at 2M total sentences

#Cleaning helpers 

# Patterns to reject a line entirely
REJECT_PATTERNS = [
    re.compile(r'[a-zA-Z]{4,}'),                  # English words (4+ letters)
    re.compile(r'\{\||\|\}|\|-|\|{2,}'),           # Wiki table markup
    re.compile(r'https?://|www\.'),                # URLs
    re.compile(r'\d{1,2}:\d{2}\s*(AM|PM|am|pm)'), # Timestamps
    re.compile(r'#\d+'),                            # Forum post numbers
    re.compile(r'\[color=|\[/color\]'),             # BBCode
    re.compile(r'<[a-zA-Z/]'),                      # HTML tags
    re.compile(r'={3,}|-{3,}|\*{3,}'),             # Repeated symbols
    re.compile(r'^\s*[\d\W]+\s*$'),                # Pure numbers/punctuation
]

def split_into_sentences(text):
    """Split a paragraph into sentences on Arabic/Latin punctuation."""
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.،؟!\n])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def is_mostly_arabic(text):
    """At least 70% of non-space characters must be Arabic."""
    chars = text.replace(' ', '')
    if not chars:
        return False
    arabic = sum(1 for c in chars if '\u0600' <= c <= '\u06FF')
    return arabic / len(chars) >= 0.70

def clean_line(text):
    text = text.strip()
    # Remove zero-width and control characters
    text = ''.join(c for c in text if not unicodedata.category(c).startswith('C'))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove repeated dots/ellipsis beyond 2
    text = re.sub(r'\.{3,}', '..', text)
    # Remove brackets content that looks like metadata: [text]
    text = re.sub(r'\[[^\]]{0,30}\]', '', text)
    # Remove leading punctuation/numbers
    text = re.sub(r'^[\d\s\W]+', '', text)
    return text.strip()

def should_reject(text):
    if len(text) < 20 or len(text) > 500:
        return True
    if not is_mostly_arabic(text):
        return True
    for pat in REJECT_PATTERNS:
        if pat.search(text):
            return True
    return False

#  Process files 

def process_file(path, is_paragraph=False):
    """Yield clean sentences from a file."""
    kept = 0
    dropped = 0
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # For mC4 (paragraph-level), split into sentences first
            if is_paragraph:
                sentences = split_into_sentences(raw_line)
            else:
                sentences = [raw_line]

            for sent in sentences:
                sent = clean_line(sent)
                if should_reject(sent):
                    dropped += 1
                    continue
                kept += 1
                yield sent

    print(f"  Kept: {kept:,} | Dropped: {dropped:,} | Ratio: {kept/(kept+dropped)*100:.1f}%")

#  Main 

print("="*60)
print("Preprocessing Arabic Corpus")
print("="*60)

total = 0
seen = set()  # Deduplication

with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:

    print(f"\n[1/2] Processing Wikipedia 1M: {WIKI_PATH}")
    for sent in process_file(WIKI_PATH, is_paragraph=False):
        if sent not in seen:
            seen.add(sent)
            out.write(sent + '\n')
            total += 1
            if total % 200000 == 0:
                print(f"  Written {total:,} sentences so far...")
        if total >= TARGET:
            break

    print(f"\n[2/2] Processing mC4 500K paragraphs (splitting into sentences): {MC4_PATH}")
    for sent in process_file(MC4_PATH, is_paragraph=True):
        if sent not in seen:
            seen.add(sent)
            out.write(sent + '\n')
            total += 1
            if total % 200000 == 0:
                print(f"  Written {total:,} sentences so far...")
        if total >= TARGET:
            print(f"  Reached {TARGET:,} target — stopping.")
            break

print(f"\n{'='*60}")
print(f"Done! Total clean sentences: {total:,}")
print(f"Output: {OUTPUT_PATH}")
print(f"{'='*60}")
