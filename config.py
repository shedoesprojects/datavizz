# simple settings container (no extra dependency)
import os
from dataclasses import dataclass

@dataclass
class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEBUG: bool = os.getenv("DATAWIZ_DEBUG", "0") == "1"

settings = Settings()
