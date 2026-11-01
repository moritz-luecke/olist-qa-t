"""
============================================================
Quick test: verify OPENAI_API_KEY works
============================================================

Costs less than $0.0001 (one tiny embedding + one tiny chat call).
Run BEFORE the first real benchmark run.

Usage:
    export OPENAI_API_KEY=sk-...
    python test_openai_key.py

Or with a .env file in the same directory:
    OPENAI_API_KEY=sk-...
============================================================
"""
from __future__ import annotations

import os
import sys
import time


# ----------------------------------------------------------------
# 1. Optional .env loading (no python-dotenv dependency)
# ----------------------------------------------------------------
def load_dotenv_if_present() -> None:
    """Minimal .env loader — KEY=VALUE lines, no quotes, no interpolation."""
    for candidate in (".env", "../.env"):
        if os.path.isfile(candidate):
            with open(candidate) as fp:
                for line in fp:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return


# ----------------------------------------------------------------
# 2. Run the checks
# ----------------------------------------------------------------
def main() -> int:
    load_dotenv_if_present()

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("✗ OPENAI_API_KEY is not set.")
        print("  Either: export OPENAI_API_KEY=sk-...")
        print("  Or:     create a .env file with OPENAI_API_KEY=sk-...")
        return 1
    masked = f"{key[:7]}...{key[-4:]}" if len(key) > 14 else "[short]"
    print(f"✓ OPENAI_API_KEY found ({masked})")

    try:
        from openai import OpenAI
    except ImportError:
        print("✗ openai package not installed. Run: pip install openai")
        return 1
    print(f"✓ openai package importable")

    client = OpenAI()

    # ------------------------------------------------------------
    # Check 1: Embeddings
    # ------------------------------------------------------------
    print("\n--- Test 1: text-embedding-3-small ---")
    t0 = time.time()
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input="hello world",
        )
    except Exception as e:
        print(f"✗ Embedding call failed: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    vec = resp.data[0].embedding
    print(f"✓ Embedding returned in {dt:.2f}s")
    print(f"  dim={len(vec)}, first 3 values: {vec[:3]}")
    print(f"  prompt_tokens={resp.usage.prompt_tokens}")

    # ------------------------------------------------------------
    # Check 2: Chat completion (the model we'll use for generation)
    # ------------------------------------------------------------
    print("\n--- Test 2: gpt-4o-mini-2024-07-18 (generation model) ---")
    t0 = time.time()
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            top_p=1,
            seed=42,
            max_tokens=20,
            messages=[
                {"role": "system", "content": "You answer in one word."},
                {"role": "user", "content": "What is the capital of Brazil?"},
            ],
        )
    except Exception as e:
        print(f"✗ Chat call failed: {type(e).__name__}: {e}")
        return 1
    dt = time.time() - t0
    answer = chat.choices[0].message.content.strip()
    print(f"✓ Chat returned in {dt:.2f}s")
    print(f"  answer: {answer!r}")
    print(f"  usage: prompt={chat.usage.prompt_tokens}, "
          f"completion={chat.usage.completion_tokens}")

    # ------------------------------------------------------------
    # Check 3: Judge model (only needed for Phase 5)
    # ------------------------------------------------------------
    print("\n--- Test 3: gpt-4o-2024-08-06 (judge model) ---")
    t0 = time.time()
    try:
        judge = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=20,
            messages=[
                {"role": "system", "content": "Reply with one JSON object."},
                {"role": "user", "content": 'Output: {"ok": true}'},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"✗ Judge call failed: {type(e).__name__}: {e}")
        print("  (If the error is 'model not found', your account may need")
        print("   to be upgraded for gpt-4o; gpt-4o-mini still works.)")
        return 1
    dt = time.time() - t0
    print(f"✓ Judge returned in {dt:.2f}s")
    print(f"  answer: {judge.choices[0].message.content!r}")

    # ------------------------------------------------------------
    # Cost summary
    # ------------------------------------------------------------
    total_cost = (
        resp.usage.prompt_tokens * 0.02 / 1_000_000              # embedding
        + chat.usage.prompt_tokens * 0.15 / 1_000_000            # gpt-4o-mini in
        + chat.usage.completion_tokens * 0.60 / 1_000_000        # gpt-4o-mini out
        + judge.usage.prompt_tokens * 2.50 / 1_000_000           # gpt-4o in
        + judge.usage.completion_tokens * 10.00 / 1_000_000      # gpt-4o out
    )
    print(f"\n=== All three API checks passed ===")
    print(f"Total spend on this test: ${total_cost:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
