"""Abstraction layer for different LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class ChatSession(ABC):
    """Minimal chat session interface returned by providers."""

    @abstractmethod
    def send_message(self, prompt: str, **kwargs) -> str:  # pragma: no cover - interface
        ...


class LLMProvider(ABC):
    """Base interface for language model providers."""

    model_id: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - interface
        ...

    @abstractmethod
    def start_chat(self, history: Optional[List] = None, context: Optional[str] = None, **kwargs) -> ChatSession:  # pragma: no cover - interface
        ...


# ---------------------------------------------------------------------------
# Vertex AI implementation
# ---------------------------------------------------------------------------
from typing import Any


class _VertexAIChatSession(ChatSession):
    def __init__(self, session: Any):
        self._session = session

    def send_message(self, prompt: str, **kwargs) -> str:
        response = self._session.send_message(prompt, **kwargs)
        if hasattr(response, "text"):
            return str(response.text)
        return str(response)


class VertexAIProvider(LLMProvider):
    def __init__(self, model_id: str):
        import vertexai
        from google.cloud.aiplatform import telemetry
        from vertexai.language_models import (
            TextGenerationModel,
            CodeGenerationModel,
            CodeChatModel,
        )
        from vertexai.generative_models import (
            GenerativeModel,
            HarmCategory,
            HarmBlockThreshold,
        )
        from utilities import PROJECT_ID, PG_REGION

        vertexai.init(project=PROJECT_ID, location=PG_REGION)

        self.model_id = model_id
        self.safety_settings = None
        if model_id == "code-bison-32k":
            with telemetry.tool_context_manager("opendataqna"):
                self.model = CodeGenerationModel.from_pretrained("code-bison-32k")
        elif model_id == "text-bison-32k":
            with telemetry.tool_context_manager("opendataqna"):
                self.model = TextGenerationModel.from_pretrained("text-bison-32k")
        elif model_id == "codechat-bison-32k":
            with telemetry.tool_context_manager("opendataqna"):
                self.model = CodeChatModel.from_pretrained("codechat-bison-32k")
        elif model_id == "gemini-1.0-pro":
            with telemetry.tool_context_manager("opendataqna"):
                self.model = GenerativeModel("gemini-1.0-pro-001")
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        elif model_id == "gemini-1.5-flash":
            with telemetry.tool_context_manager("opendataqna"):
                self.model = GenerativeModel("gemini-1.5-flash-preview-0514")
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        elif model_id == "gemini-1.5-pro":
            with telemetry.tool_context_manager("opendataqna"):
                self.model = GenerativeModel("gemini-1.5-pro-001")
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        else:
            raise ValueError("Please specify a compatible model.")

    def generate(self, prompt: str, **kwargs) -> str:
        if hasattr(self.model, "generate_content"):
            context_query = self.model.generate_content(
                prompt, safety_settings=self.safety_settings, stream=False
            )
            return str(context_query.candidates[0].text).replace("```sql", "").replace("```", "").rstrip("\n")
        context_query = self.model.predict(prompt, **kwargs)
        return str(context_query.candidates[0])

    def start_chat(self, history: Optional[List] = None, context: Optional[str] = None, **kwargs) -> ChatSession:
        from vertexai.generative_models import GenerationConfig, Content, Part
        from google.cloud.aiplatform import telemetry

        chat_history = []
        if history:
            for entry in history:
                user_message = Content(parts=[Part.from_text(entry["user_question"])], role="user")
                bot_message = Content(parts=[Part.from_text(entry["bot_response"])], role="assistant")
                chat_history.extend([user_message, bot_message])

        if "gemini" in self.model_id:
            config = GenerationConfig(**kwargs) if kwargs else None
            with telemetry.tool_context_manager("opendataqna-buildsql-v2"):
                session = self.model.start_chat(history=chat_history, response_validation=False, generation_config=config)
                if context:
                    session.send_message(context)
        elif self.model_id == "codechat-bison-32k":
            with telemetry.tool_context_manager("opendataqna-buildsql-v2"):
                session = self.model.start_chat(context=context)
        else:
            raise ValueError("Invalid model for chat")
        return _VertexAIChatSession(session)


# ---------------------------------------------------------------------------
# Local transformers / OpenAI compatible provider
# ---------------------------------------------------------------------------

class _LocalChatSession(ChatSession):
    def __init__(self, provider: "LocalProvider", history: Optional[List] = None):
        self.provider = provider
        self.history = history or []

    def send_message(self, prompt: str, **kwargs) -> str:
        # Concatenate previous turns naively
        full_prompt = "\n".join(self.history + [prompt])
        response = self.provider.generate(full_prompt, **kwargs)
        self.history.extend([prompt, response])
        return response


class LocalProvider(LLMProvider):
    def __init__(self, model_id: str = "gpt2"):
        self.model_id = model_id
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline("text-generation", model=model_id)
        except Exception:
            # If transformers or the model is not available, fall back to echo behaviour
            self._pipe = None

    def generate(self, prompt: str, **kwargs) -> str:
        if self._pipe is None:
            # Fallback behaviour for environments without transformers
            return ""  # Return empty string to keep pipeline functional
        max_tokens = kwargs.get("max_output_tokens", 256)
        result = self._pipe(prompt, max_new_tokens=max_tokens)
        text = result[0]["generated_text"]
        return text[len(prompt) :]

    def start_chat(self, history: Optional[List] = None, context: Optional[str] = None, **kwargs) -> ChatSession:
        session_history = history or []
        if context:
            session_history.append(context)
        return _LocalChatSession(self, session_history)
