"""Provides the base class for all Agents."""

from abc import ABC
from .providers import (
    LLMProvider,
    VertexAIProvider,
    LocalProvider,
)


class Agent(ABC):
    """The core class for all Agents.

    Args:
        model_id: Identifier of the underlying model to load.
        provider: Backend provider to use. Defaults to ``"vertexai"``.
    """

    agentType: str = "Agent"

    def __init__(self, model_id: str, provider: str = "vertexai"):
        self.model_id = model_id
        if provider == "vertexai":
            self.provider: LLMProvider = VertexAIProvider(model_id)
        elif provider in {"local", "huggingface"}:
            self.provider = LocalProvider(model_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # Helper methods proxying to the underlying provider
    # ------------------------------------------------------------------
    def generate_llm_response(self, prompt: str, **kwargs) -> str:
        """Generate a response for a given prompt via the provider."""
        return self.provider.generate(prompt, **kwargs)

    def start_chat(self, **kwargs):
        """Return a chat session from the provider."""
        return self.provider.start_chat(**kwargs)

    # ------------------------------------------------------------------
    # Utility method reused by a few agents
    # ------------------------------------------------------------------
    def rewrite_question(self, question, session_history):
        formatted_history = ""
        concat_questions = ""
        for i, _row in enumerate(session_history, start=1):
            user_question = _row["user_question"]
            formatted_history += f"User Question - Turn :: {i} : {user_question}\n"
            concat_questions += f"{user_question} "

        context_prompt = f"""
            Your main objective is to rewrite and refine the question based on the previous questions that has been asked.

            Refine the given question using the provided questions history to produce a standalone question with full context. The refined question should be self-contained, requiring no additional context for answering it.

            Make sure all the information is included in the re-written question. You just need to respond with the re-written question.

            Below is the previous questions history:

            {formatted_history}

            Question to rewrite:

            {question}
        """
        re_written_qe = str(self.generate_llm_response(context_prompt))

        print("*" * 25 + "Re-written question:: " + "*" * 25 + "\n" + str(re_written_qe))

        return str(concat_questions), str(re_written_qe)
