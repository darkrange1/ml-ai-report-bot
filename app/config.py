from pydantic import BaseModel
import os

class Settings(BaseModel):
    app_env: str = os.getenv("APP_ENV", "dev")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()

settings = Settings()
