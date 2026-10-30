
from pathlib import Path
import argparse
import sys

# Configure UTF-8 output for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Locate the project and reusable source modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retentionai_service import RetentionAIAssistant


def answer_question(assistant, question):
    """Display an answer and its retrieved evidence."""

    result = assistant.ask(question)

    print("\n" + "=" * 65)
    print("RETENTIONAI BUSINESS INTELLIGENCE")
    print("=" * 65)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nResponse source:")
    print(result["response_source"])

    retrieved = result["retrieved_documents"]

    print("\nRetrieved knowledge:")
    if retrieved.empty:
        print("No relevant knowledge documents found.")
    else:
        for _, row in retrieved.iterrows():
            print(
                f"- {row['topic']} "
                f"(similarity: {row['similarity_score']:.4f})"
            )

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Local RetentionAI business assistant."
    )

    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Ask one business question and exit."
    )

    args = parser.parse_args()

    try:
        assistant = RetentionAIAssistant()
    except Exception as error:
        print("Could not initialize RetentionAI.")
        print(f"{type(error).__name__}: {error}")
        raise SystemExit(1)

    print("RetentionAI is ready.")
    print("Mode: Local structured data + RAG")
    print("Gemini API calls: Disabled")
    print("Type 'exit' or 'quit' to stop.")

    # One-shot mode for scripts and automated tests.
    if args.question:
        answer_question(assistant, args.question)
        return

    # Interactive terminal mode.
    while True:
        try:
            question = input("\nAsk RetentionAI: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nRetentionAI session ended.")
            break

        if question.lower() in {"exit", "quit"}:
            print("RetentionAI session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer_question(assistant, question)
        except Exception as error:
            print(
                f"Could not answer the question: "
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()
