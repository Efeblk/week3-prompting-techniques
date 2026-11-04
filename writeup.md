# Week 3 — Prompting Techniques Writeup

---

## 1. K-Shot Prompting (`k_shot_prompting.py`)

### Final Prompt
```python
YOUR_SYSTEM_PROMPT = """You are a support ticket classifier for an online store.
Classify the customer's message into exactly one of these lowercase labels:
billing, shipping, returns, technical, other

Rules:
- Output ONLY the single lowercase label word.
- Do NOT include any punctuation, quotes, explanation, or extra words.

Examples:
Customer: Why is there an extra $15 fee on my credit card statement?
billing

Customer: My tracking link has been stuck on label created for five days.
shipping

Customer: Can I exchange this jacket for a smaller size or get my money back?
returns

Customer: I keep getting a 500 error whenever I click the checkout button.
technical

Customer: Do you have physical gift cards available in stores?
other"""
```

### Sample Input / Output (from `outputs/k_shot_prompting.txt`)
- **Input:** `"My card was charged twice for order 4412."`
- **Output:**
  ```text
  [PASS] 'My card was charged twice for order 4412.' -> got 'billing', expected 'billing'
  ```

### Prompt Iterations & What Failed in Earlier Attempts
- **Iterations taken:** 2 iterations (after an unedited baseline run).
- **What failed:**
  - **Baseline (empty prompt — 0/4 passed):** `llama3.1:8b` acted as a conversational customer service agent and asked multi-bullet follow-up questions (e.g., `"i'm so sorry to hear that your card was charged twice for order 4412. i'd be happy to help you resolve the issue..."`).
  - **Iteration 1 (zero-shot instruction without examples or strict formatting rule — 0/4 passed):** Using `"Classify the customer support ticket into one of these categories: billing, shipping, returns, technical, other."`, the model chose the right category every time in meaning, but failed all 4 equality checks (`got == expected`) because it wrapped the label in conversational text, quotes, periods, or Markdown bolding:
    - `'i would classify this customer support ticket as: billing.'`
    - `'i would classify this customer support ticket as "shipping".'`
    - `'i would classify this customer support ticket as: **returns**.'`
    - `'i would classify this customer support ticket as "technical". the issue described is related to the functionality of the app...'`
  - **Iteration 2 (5-shot prompt with one example per label + explicit single-word rule — 4/4 passed):** Providing concrete `Customer: ...` -> `<label>` examples alongside the no-punctuation rule constrained the model to emit only the bare lowercase label.

**What I learned:** Just saying "classify" made the model chat and fail (0/4), while 5 examples forced a clean one-word answer (4/4).

---

## 2. Chain-of-Thought (`chain_of_thought.py`)

### Final Prompt
```python
YOUR_SYSTEM_PROMPT = """You are a support-desk assistant deciding whether an order is eligible for a refund.
Follow these steps using numbered step-by-step reasoning:
1. Check if the item was purchased within 30 days.
2. Check if the item is unused / in original condition (unless not mentioned as violated).
3. Check if the item is a final-sale item (final-sale items are never refundable).

An order is eligible ONLY if all applicable refund policy conditions are satisfied.
After your numbered steps, your VERY LAST line must be literally and only:
DECISION: YES
or
DECISION: NO
Do not add any punctuation, markdown, or extra text on the final line."""
```

### Sample Input / Output (from `outputs/chain_of_thought.txt`)
- **Input:** `"Bought 5 days ago but it was a final-sale item."`
- **Output:**
  ```text
  [PASS] 'Bought 5 days ago but it was a final-sale item.' -> NO, expected NO
  ```

### Prompt Iterations & What Failed in Earlier Attempts
- **Iterations taken:** 2 iterations (after an unedited baseline run).
- **What failed:**
  - **Baseline & Iteration 1 (generic step-by-step instruction without exact final-line specification — 0/3 passed):** Using `"Reason step by step through the refund policy to decide if the order is eligible for a refund, then state your decision."`, all three test cases failed with `?`:
    - `[FAIL] 'Bought 10 days ago, unused, not final sale.' -> ?, expected YES`
    - `[FAIL] 'Bought 45 days ago, unused.' -> ?, expected NO`
    - `[FAIL] 'Bought 5 days ago but it was a final-sale item.' -> ?, expected NO`
    The grader in `decide()` only inspects `text.splitlines()[-1].upper()` for `"DECISION: YES"` or `"DECISION: NO"`. Without an explicit output contract for the final line, the model concluded with natural-language summary sentences instead of the exact tag.
  - **Iteration 2 (numbered 3-step policy check + strict final-line format — 3/3 passed):** Instructing the model to walk through each of the 3 policy conditions by number and end with literally `DECISION: YES` or `DECISION: NO` on the final line passed 3/3 across repeated runs.

**What I learned:** "Think step by step" rambled and scored 0/3 until I required numbered steps ending with `DECISION: YES/NO` (3/3).

---

## 3. Tool Calling (`tool_calling.py`)

### Final Prompt & Tool Schema
```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status of a customer order by its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The exact order ID string from the customer's message (for example, '4412'), without '#' or extra words.",
                    },
                },
                "required": ["order_id"],
            },
        },
    }
]

YOUR_SYSTEM_PROMPT = (
    "You are a support-desk assistant. Whenever a customer asks about an order, "
    "you MUST call the `get_order_status` tool with the exact `order_id` string "
    "extracted from the user's message (digits only, e.g. '4412') to look up the order before answering."
)
```

### Sample Input / Output (from `outputs/tool_calling.txt`)
- **Input:** `"Hi, can you check on order 4412 for me?"`
- **Output:**
  ```text
  [PASS] tool called with order_id='4412'
  tool result: shipped, arriving Friday
  ```

### Prompt Iterations & What Failed in Earlier Attempts
- **Iterations taken:** 1 iteration (after inspecting the baseline run).
- **What failed / observations from earlier attempts:**
  - **Baseline (empty tool descriptions and empty system prompt — 1/1 passed):** Even with `description: ""` and `YOUR_SYSTEM_PROMPT = ""`, `llama3.1:8b` actually emitted a tool call `get_order_status(order_id='4412')` because Ollama passes `TOOLS` natively and the function/parameter identifiers (`get_order_status`, `order_id`) were self-descriptive enough for the simple prompt `"Hi, can you check on order 4412 for me?"`.
  - **Iteration 1 (explicit schema descriptions + system prompt — 1/1 passed):** Relying on empty descriptions is fragile because the model could either answer conversationally without invoking the tool or pass `order_id="#4412"` / `"order 4412"`, which would fail the strict check `order_id == "4412"`. Adding explicit descriptions to the schema and instructing the model in `YOUR_SYSTEM_PROMPT` to always call `get_order_status` with the digits-only `order_id` guaranteed reliable tool invocation.

**What I learned:** Obvious function names worked even with empty descriptions, though adding descriptions made it dependable.

---

## 4. Retrieval-Augmented Generation (`rag.py`)

### Final Retriever & Prompt
```python
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


YOUR_SYSTEM_PROMPT = """You are a customer support assistant. Answer the customer's question using ONLY the provided Context.
- Do not use outside knowledge or guess.
- Answer using the wording of the relevant Context line, including any specific details such as numbers and timeframes.
- If the answer is not explicitly stated in the Context, reply: "I don't know based on the provided help-center documents." """
```

### Sample Input / Output (from `outputs/rag.txt`)
- **Input:** `"What's your CEO's home address?"` (and `"Do you ship internationally?"`)
- **Output:**
  ```text
  [PASS] 'Do you ship internationally?' -> 'yes, we offer international shipping to most countries.'
  [PASS] "What's your CEO's home address?" -> "i don't know based on the provided help-center documents."
  ```

### Prompt Iterations & What Failed in Earlier Attempts
- **Iterations taken:** 3 iterations (after an unedited baseline run).
- **What failed:**
  - **Baseline (empty retriever `return []` and empty system prompt — 2/3 passed):** With an empty `Context:`, the model hallucinated generic store answers that happened to include the substrings `"30 days"` and `"yes, we ship internationally"`, falsely passing the first two checks without grounding! However, it failed Case 3 (`"What's your CEO's home address?"`) by outputting `'i can’t provide that information. is there anything else i can help you with?'` — using a contraction (`can’t`) that did not match any expected refusal substring in `["don't know", "do not know", "not sure", "cannot"]`.
  - **Iteration 1 (strict grounding prompt + naive exact-word overlap `q_words & c_words` — 2/3 passed):** Case 1 and Case 3 passed, but **Case 2 failed**:
    - `[FAIL] 'Do you ship internationally?' -> "i don't know based on the provided help-center documents."`
    Inspecting `data/help_center.txt` revealed why: the question `"Do you ship internationally?"` contains `"ship"` and `"internationally"`, whereas the relevant chunk reads `"Shipping: We offer international shipping to most countries; delivery takes 7-21 business days."` (`"shipping"` and `"international"`). Exact word intersection missed both keywords (`0` overlap) and instead matched the word `"you"` against the Returns and Billing chunks! Because the system prompt strictly forbade using outside knowledge, the model saw only Returns and Billing chunks and dutifully replied `"i don't know based on the provided help-center documents."`
  - **Iteration 2 (stopword removal + 4-character prefix/stem matching + explicit refusal phrasing — 3/3 passed):** Filtering out short/stopword tokens and matching words that share a 4-character prefix (`"ship"` <-> `"shipping"`, `"internationally"` <-> `"international"`) retrieved the shipping policy chunk and passed all 3/3 cases reliably.
  - **Iteration 3 (removing a test-specific hint — 3/3 passed):** The Iteration 2 prompt said to include details "such as timeframes or international shipping availability", which names the content of a test case and could overfit the prompt to the grader. I replaced it with a generic rule ("answer using the wording of the relevant Context line, including specific details such as numbers and timeframes") and it still passed 3/3 on repeated runs.

**What I learned:** With no context, the model guessed and got lucky; retrieval only worked once search matched "ship" to "shipping".

---

## Closing Reflection

LLMs are great at structured tasks once they see a clear pattern to copy. They struggle when prompts are open-ended or lack real data—without strict output formats and solid search logic, they either ramble or make things up.
