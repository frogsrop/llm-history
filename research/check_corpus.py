"""Check corpus candidates against plan criteria."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import Counter

CANDIDATES = {
    "A": [
        "мельник жил у мельницы и мельница кормила мельника всю жизнь",
        "старый мельник знал что мельница без воды мелет плохо и медленно",
        "вода крутила колесо а колесо крутило жернова и жернова мололи зерно",
    ],
    "B": [
        "кузнец ковал железо пока железо было горячим и кузница не остыла",
        "горячее железо легко гнётся а холодное железо ломается под молотом кузнеца",
        "молот кузнеца бил по железу и железо принимало форму нужную кузнецу",
    ],
    "C": [
        "охотник видел лису и лиса видела охотника но лес скрыл лису от охотника",
        "лиса хитрее волка потому что волк бежит прямо а лиса петляет по лесу",
        "в густом лесу охотник потерял след лисы которую он долго искал по лесу",
    ],
    "D": [
        "река несла воду в море а море не могло насытиться водой из реки",
        "рыбак ловил рыбу в реке но рыба уходила в глубину реки от рыбака",
        "глубокая река скрывала рыбу от рыбака хотя рыбак знал эту реку давно",
    ],
}

def analyze(name, sentences):
    tokens = " ".join(sentences).split()
    counts = Counter(tokens)
    vocab = set(tokens)
    most_common = counts.most_common(5)
    lengths = [len(s.split()) for s in sentences]
    max_freq = max(counts.values())

    print(f"\n=== Variant {name} ===")
    for i, s in enumerate(sentences, 1):
        print(f"  [{len(s.split())} words] {s}")
    print(f"  Vocab: {len(vocab)} unique words")
    print(f"  Top-5 words: {most_common}")
    print(f"  Max frequency: {max_freq}")
    ok = all(8 <= l <= 14 for l in lengths) and max_freq >= 3 and len(vocab) >= 10
    print(f"  Criteria: {'OK' if ok else 'FAIL'}")
    return ok

for name, sentences in CANDIDATES.items():
    analyze(name, sentences)
