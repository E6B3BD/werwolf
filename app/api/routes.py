"""API 路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.engine.game import GameManager
from app.engine.models import HumanNightAction, SheriffAction


router = APIRouter()
game_manager = GameManager()


class CreateGameRequest(BaseModel):
    """创建游戏请求。"""

    player_count: int = Field(ge=6, le=12)


class NightRequest(BaseModel):
    """夜晚动作请求。"""

    action_type: str
    target_id: int | None = None
    chat_content: str = ""


class SpeechRequest(BaseModel):
    """发言请求。"""

    content: str = ""


class VoteRequest(BaseModel):
    """投票请求。"""

    target_id: int


class SheriffRequest(BaseModel):
    """警长相关请求。"""

    run_for_sheriff: bool = False
    vote_target_id: int | None = None
    speech: str = ""
    speech_order_direction: str | None = None
    badge_target_id: int | None = None
    tear_badge: bool = False


def get_game_or_404(game_id: str):
    """统一获取游戏实例。"""
    try:
        return game_manager.get_game(game_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="游戏不存在") from exc


@router.post("/games")
async def create_game(payload: CreateGameRequest):
    """创建游戏。"""
    game = game_manager.create_game(payload.player_count)
    return game.to_snapshot()


@router.get("/games/{game_id}")
async def get_game(game_id: str):
    """获取游戏快照。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    await game.advance_ready_ai_step_if_needed()
    return game.to_snapshot()


@router.post("/games/{game_id}/night")
async def resolve_night(game_id: str, payload: NightRequest):
    """执行夜晚。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    action = HumanNightAction(
        action_type=payload.action_type,
        target_id=payload.target_id,
        chat_content=payload.chat_content,
    )
    await game.resolve_night(action)
    return game.to_snapshot()


@router.post("/games/{game_id}/wolf-chat")
async def resolve_wolf_chat(game_id: str, payload: NightRequest):
    """执行狼人协商阶段。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    action = HumanNightAction(
        action_type=payload.action_type,
        target_id=payload.target_id,
        chat_content=payload.chat_content,
    )
    await game.resolve_wolf_chat(action)
    return game.to_snapshot()


@router.post("/games/{game_id}/speech")
async def resolve_speech(game_id: str, payload: SpeechRequest):
    """执行逐位发言。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    await game.resolve_day_speeches(payload.content)
    return game.to_snapshot()


@router.post("/games/{game_id}/last-words")
async def resolve_last_words(game_id: str, payload: SpeechRequest):
    """执行遗言。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    await game.resolve_last_words(payload.content)
    return game.to_snapshot()


@router.post("/games/{game_id}/self-destruct")
async def resolve_self_destruct(game_id: str):
    """执行狼人自爆。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    await game.resolve_self_destruct()
    return game.to_snapshot()


@router.post("/games/{game_id}/sheriff")
async def resolve_sheriff(game_id: str, payload: SheriffRequest):
    """执行警长竞选与警徽相关动作。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    action = SheriffAction(
        run_for_sheriff=payload.run_for_sheriff,
        vote_target_id=payload.vote_target_id,
        speech=payload.speech,
        speech_order_direction=payload.speech_order_direction,  # 兼容前端透传
        badge_target_id=payload.badge_target_id,
        tear_badge=payload.tear_badge,
    )
    await game.resolve_sheriff_election(action)
    return game.to_snapshot()


@router.post("/games/{game_id}/speech-order")
async def choose_speech_order(game_id: str, payload: SheriffRequest):
    """警长选择发言方向。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    await game.choose_speech_order(payload.speech_order_direction)
    return game.to_snapshot()


@router.post("/games/{game_id}/badge")
async def resolve_badge(game_id: str, payload: SheriffRequest):
    """死亡警长处理警徽。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    action = SheriffAction(
        badge_target_id=payload.badge_target_id,
        tear_badge=payload.tear_badge,
    )
    await game.resolve_badge_transfer(action)
    return game.to_snapshot()


@router.post("/games/{game_id}/vote")
async def resolve_vote(game_id: str, payload: VoteRequest):
    """执行白天投票。"""
    game = get_game_or_404(game_id)
    await game.advance_timeout_if_needed()
    await game.resolve_votes(payload.target_id)
    return game.to_snapshot()
