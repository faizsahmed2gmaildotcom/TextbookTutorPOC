# Rename this file to "settings.py" and fill in the API details below

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Textbook Tutor"
    DEBUG: bool = False

    # LiteLLM
    LITELLM_BASE_URL: str = ""
    LITELLM_API_KEY: str = ""
    LITELLM_TIMEOUT_SECONDS: int = 300
    LITELLM_MAX_RETRIES: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
