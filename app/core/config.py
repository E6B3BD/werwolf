"""项目配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    """应用运行配置。"""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8008"))
    agent_decision_timeout_seconds: float = float(os.getenv("AGENT_DECISION_TIMEOUT_SECONDS", "18"))
    auto_ai_live: bool = os.getenv("AUTO_AI_LIVE", "false").lower() in {"1", "true", "yes", "on"}

    @property
    def openai_enabled(self) -> bool:
        """是否已配置 OpenAI API Key。"""
        return bool(self.openai_api_key.strip())


settings = Settings()
