"""FastAPI Web 入口。"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger = logging.getLogger("werwolf")

app = FastAPI(title="Werwolf", version="0.1.0")
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.on_event("startup")
async def startup_log():
    """启动时打印模型配置自检信息。"""
    api_key_masked = "未配置"
    if settings.openai_api_key.strip():
        key = settings.openai_api_key.strip()
        api_key_masked = f"{key[:8]}...{key[-4:]}"

    logger.warning(
        "Werwolf startup | openai_enabled=%s | model=%s | base_url=%s | tracing_disabled=%s | api_key=%s",
        settings.openai_enabled,
        settings.openai_model,
        settings.openai_base_url,
        True,
        api_key_masked,
    )


@app.get("/health")
async def health():
    """健康检查。"""
    return {
        "ok": True,
        "openai_enabled": settings.openai_enabled,
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url,
        "tracing_disabled": True,
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """避免浏览器反复请求 favicon 产生 404。"""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页。"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "openai_enabled": settings.openai_enabled,
            "default_port": settings.port,
        },
    )
