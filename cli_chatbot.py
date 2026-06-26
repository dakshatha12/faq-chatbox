"""
cli_chatbot.py
==============
Terminal-based FAQ chatbot (no browser needed).
Run:  python cli_chatbot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from faq_engine import FAQEngine


BANNER = """
╔══════════════════════════════════════════════╗
║         🛍️  ShopBot FAQ Assistant  🛍️         ║
║  Type your question and press Enter.         ║
║  Commands:  'help'  'quit'  'categories'     ║
╚══════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
  help         – show this message
  categories   – list FAQ categories
  list <cat>   – list all FAQs in a category  (e.g. 'list Returns')
  quit / exit  – exit the chatbot

Otherwise, just type your question naturally!
"""


def run_cli():
    engine = FAQEngine(Path(__file__).parent / "data" / "faqs.json")
    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "bye"):
            print("Bot: 👋  Goodbye! Have a great day!")
            break

        if cmd == "help":
            print(HELP_TEXT)
            continue

        if cmd == "categories":
            cats = engine.get_categories()
            print(f"Bot: Categories available: {', '.join(cats)}\n")
            continue

        if cmd.startswith("list "):
            cat = user_input[5:].strip()
            faqs = engine.get_faqs_by_category(cat)
            if faqs:
                print(f"Bot: FAQs in '{cat}':")
                for i, f in enumerate(faqs, 1):
                    print(f"  {i}. {f['question']}")
            else:
                print(f"Bot: No FAQs found for category '{cat}'.")
            print()
            continue

        # ── Normal question ──────────────────────────────────
        result = engine.get_answer(user_input)

        if result["matched"]:
            conf_pct = int(result["confidence"] * 100)
            bar_len = int(result["confidence"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"\n🤖  Bot  [{result['category']}]  [{bar}] {conf_pct}%")
            print(f"   Matched: \"{result['question']}\"")
            print(f"\n   {result['answer']}\n")
        else:
            print(f"\n🤖  Bot: {result['answer']}\n")


if __name__ == "__main__":
    run_cli()
