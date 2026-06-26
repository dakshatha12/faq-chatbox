"""
setup_nltk.py
=============
Run once to download all required NLTK datasets.
  python setup_nltk.py
"""
import nltk

datasets = [
    ("tokenizers", "punkt"),
    ("tokenizers", "punkt_tab"),
    ("corpora",    "stopwords"),
    ("corpora",    "wordnet"),
    ("corpora",    "omw-1.4"),
]

print("Downloading NLTK datasets …")
for category, name in datasets:
    try:
        nltk.data.find(f"{category}/{name}")
        print(f"  ✓  {name}  (already present)")
    except LookupError:
        nltk.download(name)
        print(f"  ✓  {name}  downloaded")

print("\nAll datasets ready. You can now run the chatbot.")
