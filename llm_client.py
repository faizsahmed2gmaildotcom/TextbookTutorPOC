import openai
import logging
from settings import settings
from typing import Any
from schemas import TOCStructure
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class __ModelSettings:
    model_name: str
    temperature: float
    max_tokens: int
    system_prompt_override: str | None = None
    response_format: Any | None = None


_model_settings = {
    "none": __ModelSettings(
        model_name="qwen3-coder-30b-a3b-instruct",
        temperature=0.2,
        max_tokens=200000
    ),
    "parse_toc": __ModelSettings(
        model_name="qwen3-coder-30b-a3b-instruct",
        temperature=0.2,
        max_tokens=200000,
        response_format=TOCStructure
    )
}


class LLMClient:
    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        self.client = openai.AsyncOpenAI(
            base_url=settings.LITELLM_BASE_URL,
            api_key=settings.LITELLM_API_KEY,
            timeout=settings.LITELLM_TIMEOUT_SECONDS,
            max_retries=settings.LITELLM_MAX_RETRIES
        )

    async def get_completion(
            self,
            messages: list[dict[str, Any]],
            task_type: str
    ) -> str:
        """
        Get a completion from the LLM

        Args:
            messages: List of message dictionaries
            task_type: Type of task (for model selection)

        Returns:
            Generated text response
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized")

        try:
            model_settings = _model_settings[task_type]

            params = {
                "messages": messages,
                "model": model_settings.model_name,
                "temperature": model_settings.temperature,
                "max_tokens": model_settings.max_tokens,
                "response_format": model_settings.response_format
            }

            if model_settings.system_prompt_override:
                system_message = {"role": "system", "content": model_settings.system_prompt_override}
                params["messages"] = [system_message] + messages

            if not model_settings.response_format:
                response = await self.client.chat.completions.create(**params)
            else:
                response = await self.client.chat.completions.parse(**params)
            return response.choices[0].message.content


        except Exception as e:
            logger.error(f"Error getting LLM completion: {e}")
            raise


client = LLMClient()
