"""Non-destructive DeepSeek connectivity and model check.

The script never prints the API key or response content. It is intentionally
separate from the evaluation runner so a missing key or unavailable model is
detected before a long benchmark run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true", help="list model IDs")
    parser.add_argument("--chat", action="store_true", help="send one minimal chat")
    args = parser.parse_args()
    load_dotenv(override=True)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("OPENAI_API_KEY is empty; copy .env.deepseek.example to .env and set it.")
        return 2
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(base_url=base_url, api_key=key, timeout=60.0, max_retries=0)
    if args.models:
        started = time.monotonic()
        models = client.models.list()
        ids = [m.id for m in models.data]
        print(f"models_ok count={len(ids)} elapsed_s={time.monotonic()-started:.2f}")
        print("model_ids=" + ",".join(ids))
    if args.chat:
        model = os.getenv("BEMULA_AGENT_MODEL", "deepseek-flash")
        started = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            max_tokens=16,
            extra_body={"thinking": {"type": os.getenv("BEMULA_THINKING", "disabled")}},
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"chat_ok model={model} elapsed_s={time.monotonic()-started:.2f} chars={len(content)}")
    if not args.models and not args.chat:
        parser.error("choose --models and/or --chat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
