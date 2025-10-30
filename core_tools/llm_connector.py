# core_tools/llm_connector.py
"""
LLM wrapper: uses OpenAI Python SDK (OpenAI.Client) to generate text.
"""

import os
from openai import OpenAI
from typing import List

from config.settings import OPENAI_API_KEY, LLM_MODEL

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment (config/settings.py)")

_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_text(prompt: str, system: str = None, max_tokens: int = 450, temperature: float = 0.3) -> str:
    system_msg = system or "You are a helpful expert data analyst who writes concise, reader-friendly reports."
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # API returns choices with message
    return resp.choices[0].message.content.strip()

def generate_short_bullets(prompt: str, **kwargs) -> str:
    return generate_text(prompt, **kwargs)
