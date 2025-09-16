"""Service for handling OMOP concept chat requests."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from agents import ResponseAgent
from utilities import PROMPTS, config, format_prompt


@lru_cache(maxsize=1)
def _get_agent() -> ResponseAgent:
    """Return a cached :class:`ResponseAgent` for OMOP concept chat."""

    model = config.get("OMOP", "CONCEPT_CHAT_MODEL", fallback="gemini-1.5-pro")
    return ResponseAgent(model)


async def run(question: str) -> str:
    """Generate an OMOP concept chat response for ``question``.

    Args:
        question: The user question about OMOP vocabulary concepts.

    Returns:
        The language model response formatted according to the prompt template.
    """

    prompt_template = PROMPTS["omop_concept_chat"]
    formatted_prompt = format_prompt(prompt_template, question=question)
    agent = _get_agent()
    return await asyncio.to_thread(agent.generate_llm_response, formatted_prompt)

