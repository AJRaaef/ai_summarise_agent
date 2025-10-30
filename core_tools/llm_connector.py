# core_tools/llm_connector.py
"""
LLM wrapper: uses OpenAI Python SDK (OpenAI.Client) to generate text.
"""

import streamlit as st
from openai import OpenAI
from typing import List

# -----------------------------
# 🔹 Load OpenAI key from Streamlit secrets
# -----------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    raise RuntimeError(
        "OPENAI_API_KEY not set in Streamlit secrets. "
        "Go to app settings → Secrets and add it in TOML format."
    )

# Optional: Model selection, default to GPT-3.5
LLM_MODEL = st.secrets.get("LLM_MODEL", "gpt-3.5-turbo")

_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_text(
    prompt: str,
    system: str = None,
    max_tokens: int = 450,
    temperature: float = 0.3
) -> str:
    """Generate text using OpenAI chat model."""
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
    return resp.choices[0].message.content.strip()

def generate_short_bullets(prompt: str, **kwargs) -> str:
    """Generate a short bullet-style summary using the LLM."""
    return generate_text(prompt, **kwargs)
