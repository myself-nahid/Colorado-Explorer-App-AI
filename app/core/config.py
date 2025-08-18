from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    GOOGLE_MAPS_API_KEY: str
    TAVILY_API_KEY: str
    SERPER_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()