from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    NEWSAPI_KEY: str
    MAX_RETRIES: int = 3
    BATCH_SIZE: int = 3
    DATE_RANGE_DAYS: int = 30
    MODEL_NAME: str = "gpt-4o-mini"
    REQUEST_TIMEOUT: int = 30
    WORKFLOW_TIMEOUT: int = 300
    
    class Config:
        env_file = ".env"

