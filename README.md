# 🛍️ ShopBot — FAQ Chatbot with NLP

A complete FAQ chatbot built with **Python + NLTK + scikit-learn**, featuring:
- Full NLP preprocessing pipeline (tokenize → clean → lemmatize)
- TF-IDF cosine similarity for intent matching
- REST API via Flask
- Beautiful dark-themed chat UI (no extra libraries)
- Terminal CLI mode

---

## 📁 Project Structure

```
faq_chatbot/
│
├── data/
│   └── faqs.json           ← FAQ database (questions + answers)
│
├── src/
│   └── faq_engine.py       ← Core NLP engine (the brain)
│
├── static/
│   └── index.html          ← Chat UI (HTML/CSS/JS, single file)
│
├── app.py                  ← Flask web server + REST API
├── cli_chatbot.py          ← Terminal chatbot (no browser needed)
├── setup_nltk.py           ← One-time NLTK downloader
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1 — Clone / unzip the project
```bash
cd faq_chatbot
```

### 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Download NLTK datasets (run once)
```bash
python setup_nltk.py
```

---

## 🚀 Running the Project

### Option A — Web Chat UI (recommended)
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

### Option B — Terminal / CLI
```bash
python cli_chatbot.py
```

### Option C — Test the NLP engine directly
```bash
python src/faq_engine.py
```

---

## 🧪 Testing the API Manually

```bash
# Single question
curl -X POST http://127.0.0.1:5000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "how do I track my order"}'

# List all FAQs
curl http://127.0.0.1:5000/api/faqs

# List categories
curl http://127.0.0.1:5000/api/categories

# Health check
curl http://127.0.0.1:5000/api/health
```

---

## 🧠 How It Works

```
User query
    │
    ▼
Preprocessing (faq_engine.py)
  ├─ Lowercase
  ├─ Expand contractions  ("don't" → "do not")
  ├─ Remove punctuation
  ├─ Tokenize (NLTK word_tokenize)
  ├─ Remove stopwords (keeping question words: what, how, when…)
  └─ Lemmatize (NLTK WordNetLemmatizer)
    │
    ▼
TF-IDF Vectorization  (scikit-learn TfidfVectorizer)
  ├─ Built on FAQ questions at startup
  ├─ ngram_range=(1,2)  — unigrams + bigrams
  └─ sublinear_tf=True  — log smoothing
    │
    ▼
Cosine Similarity  (sklearn cosine_similarity)
    │
    ▼
Best match selected
  ├─ confidence ≥ 0.15  → return matched answer
  └─ confidence < 0.15  → "I'm not sure" fallback
    │
    ▼
JSON response → Chat UI / CLI
```

---

## 🗃️ Adding Your Own FAQs

Edit `data/faqs.json`. Each entry follows this structure:

```json
{
  "id": 21,
  "question": "Do you have a loyalty rewards program?",
  "answer": "Yes! Earn 1 point per ₹10 spent. Redeem at checkout. Sign up at My Account > Rewards.",
  "category": "Account"
}
```

The engine rebuilds automatically on next startup — no retraining needed.

---

## 🔧 Tuning the Engine

In `src/faq_engine.py`:

| Parameter | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.15` | Lower = more answers (but less precise) |
| `ngram_range` | `(1, 2)` | `(1,3)` adds trigrams for longer phrases |
| `sublinear_tf` | `True` | Log-scale TF smoothing |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `flask` | REST API + HTML serving |
| `nltk` | Tokenization, stopwords, lemmatization |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `numpy` | Array operations |

---

## 💡 Possible Extensions

- **Spell correction** — `pyspellchecker` for typo tolerance
- **Sentence embeddings** — replace TF-IDF with `sentence-transformers` (semantic search)
- **Multi-language support** — add Indic language FAQs + IndicNLP/AI4Bharat tokenizers
- **Intent classification** — fine-tune a small BERT model on your FAQ categories
- **Persistent chat** — store conversation history in SQLite
- **Voice input** — integrate `SpeechRecognition` for voice-to-text
