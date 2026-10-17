"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


# TODO: return the subset of chunks relevant to `question` (simple keyword overlap is fine).
def YOUR_RETRIEVER(question: str, chunks: list[str]) -> list[str]:
    stopwords = {"what", "whats", "how", "long", "have", "your", "from", "with", "that", "this"}
    q_words = {
        w.strip(".,?!:;\"'").lower()
        for w in question.split()
        if len(w.strip(".,?!:;\"'")) >= 4 and w.strip(".,?!:;\"'").lower() not in stopwords
    }
    scored = []
    for chunk in chunks:
        c_words = {
            w.strip(".,?!:;\"'").lower()
            for w in chunk.split()
            if len(w.strip(".,?!:;\"'")) >= 4
        }
        score = sum(
            1
            for qw in q_words
            if any(qw == cw or qw.startswith(cw[:4]) or cw.startswith(qw[:4]) for cw in c_words)
        )
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:3]]


# TODO: Fill this in! Force the model to answer only from the provided context.
YOUR_SYSTEM_PROMPT = """You are a customer support assistant. Answer the customer's question using ONLY the provided Context.
- Do not use outside knowledge or guess.
- Answer using the wording of the relevant Context line, including any specific details such as numbers and timeframes.
- If the answer is not explicitly stated in the Context, reply: "I don't know based on the provided help-center documents." """

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
