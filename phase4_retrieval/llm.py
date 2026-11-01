"""
============================================================
Phase 4 — Chat completion wrapper (cached)
============================================================

Wraps OpenAI's gpt-4o-mini (or any chat model) with the same
disk-cache pattern as embeddings.py. The cache key includes
model, temperature, top_p, seed — so re-runs with identical
parameters are free, but a parameter change forces a re-call.

Dry-run mode returns a placeholder string instead of an API
call. Use this for smoke-testing the pipeline.

Cost reference (May 2026):
  gpt-4o-mini  $0.15 / 1M input, $0.60 / 1M output
  gpt-4o       $2.50 / 1M input, $10.00 / 1M output  (judge)
============================================================
"""
from __future__ import annotations

import os
import time
from base64 import b64encode
from pathlib import Path

from cache import DiskCache
from config import (
    GENERATION_MODEL, GEN_TEMPERATURE, GEN_TOP_P, GEN_SEED, GEN_MAX_TOKENS,
)


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


class ChatLLM:
    def __init__(
        self,
        cache_root: Path,
        model: str = GENERATION_MODEL,
        temperature: float = GEN_TEMPERATURE,
        top_p: float = GEN_TOP_P,
        seed: int = GEN_SEED,
        max_tokens: int = GEN_MAX_TOKENS,
        dry_run: bool = True,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict | None = None,
        provider: str = "openai",
        extra_body: dict | None = None,
    ):
        self.cache = DiskCache(cache_root, "chat")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        # Backend selection. Default (base_url=None) => OpenAI cloud, exactly
        # as before. A base_url + Basic-auth header => an OpenAI-compliant
        # gateway (e.g. the Fraunhofer GenAI Gateway). The client is built
        # per-instance so generation (gateway) and extraction/embeddings/judge
        # (OpenAI cloud) can coexist in one process.
        self.base_url = base_url
        self.api_key = api_key
        self.default_headers = default_headers
        self.provider = provider
        # Extra request params passed through to the backend (e.g. vLLM's
        # {"chat_template_kwargs": {"enable_thinking": False}} to switch a
        # reasoning model into plain-instruct mode for a fair comparison).
        self.extra_body = extra_body
        self._client = None

    # ------------------------------------------------------------
    def _client_(self):
        """Lazily build a per-instance OpenAI client for this backend."""
        if self._client is None:
            from openai import OpenAI
            kwargs: dict = {
                "api_key": self.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.default_headers:
                kwargs["default_headers"] = self.default_headers
            self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------
    @classmethod
    def genai_gateway(
        cls,
        cache_root: Path,
        model: str,
        *,
        base_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        dry_run: bool = True,
        **kw,
    ) -> "ChatLLM":
        """Build a ChatLLM pointed at the Fraunhofer GenAI Gateway
        (OpenAI-compliant /api/v2, Basic-auth header).

        Credentials/host are read from the environment when not passed:
          GENAI_BASE_URL          (default https://genai.iais.fraunhofer.de/api/v2)
          GENAI_USER, GENAI_PASSWORD
          GENAI_API_KEY           -> X-Api-Key          (if model requires it)
          GENAI_SUBSCRIPTION_KEY  -> X-Subscription-Key  (if model requires it)
        In dry-run mode missing credentials are tolerated (no call is made).
        """
        base_url = base_url or os.getenv(
            "GENAI_BASE_URL", "https://genai.iais.fraunhofer.de/api/v2")
        username = username or os.getenv("GENAI_USER")
        password = password or os.getenv("GENAI_PASSWORD")
        if not username or not password:
            if dry_run:
                username, password = "dry", "dry"
            else:
                raise RuntimeError(
                    "GenAI Gateway needs GENAI_USER and GENAI_PASSWORD "
                    "(or username=/password= args) for a real run.")
        token = b64encode(f"{username}:{password}".encode()).decode()
        headers = {"Authorization": f"Basic {token}"}
        if os.getenv("GENAI_API_KEY"):
            headers["X-Api-Key"] = os.environ["GENAI_API_KEY"]
        if os.getenv("GENAI_SUBSCRIPTION_KEY"):
            headers["X-Subscription-Key"] = os.environ["GENAI_SUBSCRIPTION_KEY"]
        return cls(
            cache_root, model=model, base_url=base_url, api_key="genai",
            default_headers=headers, provider="genai", dry_run=dry_run, **kw,
        )

    # ------------------------------------------------------------
    def chat(
        self,
        system: str,
        user: str,
        response_format: dict | None = None,
    ) -> dict:
        """
        Send a system+user message pair, return:
          {"text": <model response>,
           "tokens": {"prompt": int, "completion": int},
           "cached": bool}
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "system": system,
            "user": user,
            "response_format": response_format,
        }
        # Namespace the cache key per backend ONLY for non-default backends,
        # so existing OpenAI (gpt-4o-mini) cache entries stay valid.
        if self.base_url:
            payload["base_url"] = self.base_url
            payload["provider"] = self.provider
        if self.extra_body:
            payload["extra_body"] = self.extra_body
        cached = self.cache.get(payload)
        if cached is not None:
            return {"text": cached["output"],
                    "tokens": cached["tokens"],
                    "cached": True}

        if self.dry_run:
            # Deterministic placeholder so downstream evaluation can run
            placeholder = "[DRY RUN — would call LLM here]"
            self.cache.put(payload, output=placeholder,
                           tokens={"prompt_tokens": 0,
                                   "completion_tokens": 0, "dry_run": True})
            return {"text": placeholder, "tokens": {"prompt": 0, "completion": 0},
                    "cached": False}

        # Real call
        client = self._client_()
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            # Don't let one bad question (e.g. a context-length overflow on a
            # small-context model) kill a 7-system x 90-question run. Record an
            # abstention, log it, and DO NOT cache it (so a re-run retries).
            print(f"    [llm] generation error ({type(e).__name__}): "
                  f"{str(e)[:160]} -> abstaining", file=__import__("sys").stderr)
            return {"text": "INSUFFICIENT EVIDENCE (generation error)",
                    "tokens": {"prompt": 0, "completion": 0},
                    "cached": False, "error": type(e).__name__}
        text = response.choices[0].message.content or ""
        # Some OpenAI-compliant gateways omit usage; guard against that.
        usage = getattr(response, "usage", None)
        tokens = {"prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
                  "completion": getattr(usage, "completion_tokens", 0) if usage else 0}
        self.cache.put(payload, output=text, tokens=tokens)
        time.sleep(0.05)
        return {"text": text, "tokens": tokens, "cached": False}
