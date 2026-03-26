"""
Benchmark: accuracy, latency, and ranking quality at scale.

Stores 100 diverse facts, then runs 15 queries with known correct answers.
Measures hit@1 (correct fact ranks first), hit@3, and per-operation latency.
"""

import sys
import os
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

BENCH_DB = "./chroma_benchmark"

# ── 100 facts across 10 domains ─────────────────────────────────────────────

FACTS = [
    # Programming languages (10)
    "Python was created by Guido van Rossum in 1991",
    "JavaScript was created by Brendan Eich in 10 days at Netscape",
    "Rust was originally designed by Graydon Hoare at Mozilla",
    "Go was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson",
    "TypeScript was developed by Microsoft and first released in 2012",
    "Java was created by James Gosling at Sun Microsystems",
    "C++ was designed by Bjarne Stroustrup as an extension of C",
    "Ruby was created by Yukihiro Matsumoto in Japan in 1995",
    "Swift was developed by Apple and first released in 2014",
    "Kotlin was developed by JetBrains and first appeared in 2011",

    # Geography (10)
    "The capital of France is Paris",
    "The capital of Japan is Tokyo",
    "The Amazon River is the largest river by water volume in the world",
    "Mount Everest is the tallest mountain above sea level at 8849 meters",
    "Australia is both a country and a continent",
    "The Sahara is the largest hot desert in the world",
    "Iceland sits on the Mid-Atlantic Ridge between tectonic plates",
    "The Nile is traditionally considered the longest river in the world",
    "The Mariana Trench is the deepest part of the ocean at about 11000 meters",
    "Antarctica is the driest continent on Earth",

    # Science (10)
    "DNA was first identified by Friedrich Miescher in 1869",
    "The speed of light in vacuum is approximately 299792458 meters per second",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure",
    "Einstein published his theory of general relativity in 1915",
    "The human body contains approximately 37 trillion cells",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen",
    "Mitochondria are often called the powerhouse of the cell",
    "The periodic table was first published by Dmitri Mendeleev in 1869",
    "Newton published Principia Mathematica in 1687",
    "CRISPR-Cas9 was adapted for genome editing by Doudna and Charpentier",

    # History (10)
    "The French Revolution began in 1789 with the storming of the Bastille",
    "World War II ended in 1945 with the surrender of Japan",
    "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed",
    "The printing press was invented by Johannes Gutenberg around 1440",
    "The Berlin Wall fell on November 9 1989",
    "The Industrial Revolution began in Britain in the late 18th century",
    "Alexander the Great conquered the Persian Empire by 331 BC",
    "The American Declaration of Independence was signed in 1776",
    "The Ottoman Empire was officially dissolved in 1922",
    "Magellan's expedition completed the first circumnavigation of Earth in 1522",

    # Technology (10)
    "The first iPhone was released by Apple in June 2007",
    "Tim Berners-Lee invented the World Wide Web in 1989",
    "The first email was sent by Ray Tomlinson in 1971",
    "Bitcoin was created by the pseudonymous Satoshi Nakamoto in 2009",
    "The transistor was invented at Bell Labs in 1947",
    "Linux was created by Linus Torvalds in 1991",
    "GPT-3 was released by OpenAI in June 2020",
    "The first computer mouse was invented by Douglas Engelbart in 1964",
    "ARPANET the precursor to the internet was established in 1969",
    "The USB standard was first released in 1996",

    # Music (10)
    "Beethoven composed his Ninth Symphony while almost completely deaf",
    "The Beatles released their final album Let It Be in 1970",
    "Mozart composed over 600 works during his 35 years of life",
    "Jazz originated in New Orleans in the early 20th century",
    "The electric guitar was first commercially produced in the 1930s",
    "Vinyl records were the dominant music format until the 1980s",
    "Hip hop originated in the Bronx New York in the 1970s",
    "Spotify was launched in Sweden in 2008",
    "The piano was invented by Bartolomeo Cristofori around 1700",
    "Auto-Tune was invented by Andy Hildebrand and first used in 1997",

    # Food (10)
    "Sushi originated in Southeast Asia as a method of preserving fish in fermented rice",
    "Chocolate was first consumed as a bitter drink by the Aztecs",
    "The tomato was originally thought to be poisonous in Europe",
    "Coffee was first cultivated in Ethiopia before spreading to the Arabian Peninsula",
    "Pizza Margherita was created in Naples to honor Queen Margherita in 1889",
    "Pasta has been a staple food in Italy since at least the 13th century",
    "The chili pepper originated in the Americas and was spread globally by Portuguese traders",
    "Fermentation is one of the oldest food preservation techniques dating back thousands of years",
    "Umami was identified as the fifth basic taste by Kikunae Ikeda in 1908",
    "The sandwich is named after John Montagu the 4th Earl of Sandwich",

    # Space (10)
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars",
    "Light from the Sun takes about 8 minutes to reach Earth",
    "Mars has the largest volcano in the solar system called Olympus Mons",
    "Saturn's rings are made mostly of ice particles and rocky debris",
    "The Apollo 11 mission landed the first humans on the Moon in 1969",
    "Pluto was reclassified as a dwarf planet in 2006",
    "A black hole is a region of spacetime where gravity is so strong nothing can escape",
    "The Hubble Space Telescope was launched in 1990",
    "Jupiter is the largest planet in our solar system",
    "The International Space Station orbits Earth at about 400 kilometers altitude",

    # Sports (10)
    "The modern Olympic Games were revived in Athens in 1896",
    "Basketball was invented by James Naismith in 1891",
    "The FIFA World Cup is held every four years",
    "Cricket originated in England and is now popular in South Asia and Australia",
    "The Tour de France is the most famous cycling race in the world",
    "Usain Bolt holds the world record for the 100 meters at 9.58 seconds",
    "Tennis was originally called lawn tennis and evolved from real tennis",
    "The first Super Bowl was played in 1967",
    "Baseball is often referred to as America's pastime",
    "Formula 1 racing began with the 1950 British Grand Prix",

    # Philosophy (10)
    "Socrates was sentenced to death by drinking hemlock in 399 BC",
    "Descartes famous statement Cogito ergo sum means I think therefore I am",
    "Confucius emphasized the importance of personal ethics and morality",
    "Nietzsche declared that God is dead in his work The Gay Science",
    "Plato founded the Academy in Athens one of the first institutions of higher learning",
    "Kant's Critique of Pure Reason was published in 1781",
    "Aristotle was a student of Plato and tutor of Alexander the Great",
    "Existentialism focuses on individual existence freedom and choice",
    "The Tao Te Ching attributed to Lao Tzu is a foundational text of Taoism",
    "Stoicism teaches that virtue is the highest good and one should accept what cannot be changed",
]

# ── 15 queries with known correct answers ────────────────────────────────────
# Each tuple is (query, substring that must appear in the top-1 result)

QUERIES = [
    ("Who created Python?", "Guido van Rossum"),
    ("What is the capital of Japan?", "Tokyo"),
    ("How fast does light travel?", "299792458"),
    ("When did World War 2 end?", "1945"),
    ("Who invented the World Wide Web?", "Tim Berners-Lee"),
    ("Where did jazz music come from?", "New Orleans"),
    ("What is umami?", "fifth basic taste"),
    ("What is the largest planet?", "Jupiter"),
    ("Who holds the 100 meter world record?", "Usain Bolt"),
    ("What does Cogito ergo sum mean?", "I think therefore I am"),
    ("When was the first iPhone released?", "2007"),
    ("Who designed Rust programming language?", "Graydon Hoare"),
    ("What is the deepest part of the ocean?", "Mariana Trench"),
    ("Who invented basketball?", "James Naismith"),
    ("When did the Berlin Wall fall?", "1989"),
]


def run():
    shutil.rmtree(BENCH_DB, ignore_errors=True)
    config.CHROMA_DB_PATH = BENCH_DB

    from models.semantic import SemanticMemory
    from stores.semantic_store import SemanticStore
    from retrieval.retriever import UnifiedRetriever

    store = SemanticStore()
    retriever = UnifiedRetriever(stores={"semantic": store})

    # ── store phase ──────────────────────────────────────────────────────
    print(f"Storing {len(FACTS)} facts...")
    store_times = []
    for i, fact in enumerate(FACTS):
        t0 = time.time()
        store.store(SemanticMemory(content=fact))
        elapsed = time.time() - t0
        store_times.append(elapsed)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(FACTS)} stored (avg {sum(store_times)/len(store_times)*1000:.0f}ms/fact)")

    avg_store = sum(store_times) / len(store_times) * 1000
    print(f"\nStore: {len(FACTS)} facts in {sum(store_times):.1f}s (avg {avg_store:.0f}ms/fact)\n")

    # ── query phase ──────────────────────────────────────────────────────
    print(f"Running {len(QUERIES)} queries against {len(FACTS)} facts...\n")
    hit_at_1 = 0
    hit_at_3 = 0
    query_times = []

    for query, expected_substr in QUERIES:
        t0 = time.time()
        results = retriever.query(query, top_k=5)
        elapsed = time.time() - t0
        query_times.append(elapsed)

        top1 = results[0].record.content if results else ""
        top3_contents = [r.record.content for r in results[:3]]
        found_at_1 = expected_substr.lower() in top1.lower()
        found_at_3 = any(expected_substr.lower() in c.lower() for c in top3_contents)

        if found_at_1:
            hit_at_1 += 1
        if found_at_3:
            hit_at_3 += 1

        status = "OK" if found_at_1 else ("top3" if found_at_3 else "MISS")
        print(f"  [{status:>4}] {query}")
        print(f"         → {top1[:80]}  (sim={results[0].raw_similarity:.4f}, {elapsed*1000:.0f}ms)")
        if not found_at_1:
            print(f"         expected: {expected_substr}")

    avg_query = sum(query_times) / len(query_times) * 1000

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"Facts stored:    {len(FACTS)}")
    print(f"Queries run:     {len(QUERIES)}")
    print(f"Hit@1:           {hit_at_1}/{len(QUERIES)} ({hit_at_1/len(QUERIES)*100:.0f}%)")
    print(f"Hit@3:           {hit_at_3}/{len(QUERIES)} ({hit_at_3/len(QUERIES)*100:.0f}%)")
    print(f"Avg store time:  {avg_store:.0f}ms")
    print(f"Avg query time:  {avg_query:.0f}ms")
    print(f"{'═' * 50}")

    shutil.rmtree(BENCH_DB, ignore_errors=True)


if __name__ == "__main__":
    run()
