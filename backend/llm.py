"""
llm.py — Async LLM client using Groq (llama-3.3-70b-versatile by default).

Set GROQ_API_KEY in .env (local) or Streamlit Secrets (deployment).
Override the model via GROQ_MODEL env var.
"""
from typing import List, Dict
from openai import AsyncOpenAI
from backend.config import GROQ_API_KEY, GROQ_MODEL

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient:
    def __init__(self):
        if GROQ_API_KEY:
            self._client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=_GROQ_BASE_URL)
            self._model = GROQ_MODEL
            print(f"[LLM] Backend: Groq ({self._model})")
        else:
            self._client = None
            self._model = ""
            print("[LLM] GROQ_API_KEY not set — LLM disabled.")

    async def invoke(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        if not self._client:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env or Streamlit Secrets.")
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


llm_client = LLMClient()
