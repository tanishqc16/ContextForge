from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
