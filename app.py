"""
app.py
======
Flask REST API + HTML serving layer for the FAQ Chatbot.

Endpoints:
  GET  /                  → serves chat UI (index.html)
  POST /api/chat          → { "message": "..." } → { answer, question, category, confidence, matched }
  GET  /api/categories    → list of FAQ categories
  GET  /api/faqs          → all FAQ questions (for frontend hints)
  GET  /api/health        → health check
"""

from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import sys, os

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
from faq_engine import FAQEngine

app = Flask(__name__)

# ── initialise engine once at startup ─────────────────────────────────────────
FAQ_PATH = Path(__file__).parent / "data" / "faqs.json"
engine = FAQEngine(FAQ_PATH)

# ── read HTML template ─────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent / "static" / "index.html"


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400
    result = engine.get_answer(message)
    return jsonify(result)


@app.route("/api/categories")
def categories():
    return jsonify(engine.get_categories())


@app.route("/api/faqs")
def faqs():
    cat = request.args.get("category")
    if cat:
        return jsonify(engine.get_faqs_by_category(cat))
    return jsonify([{"question": q} for q in engine.all_questions()])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "faq_count": len(engine.faqs)})


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  FAQ Chatbot running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
