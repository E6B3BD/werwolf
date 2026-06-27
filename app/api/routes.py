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

    player_count: int = Field(default=12, ge=12, le=12)


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


def ensure_sheriff_enabled(game) -> None:
    """默认主规则不启用警长/警徽，扩展入口必须显式开关。"""
    if not game.rule_profile.sheriff_enabled:
        raise HTTPException(status_code=400, detail="当前默认规则未启用警长/警徽扩展")


async def snapshot_after_action(game, *, max_steps: int = 6):
    """真人提交动作后推进有限 AI 步骤，避免一次 POST 跨过多个可感知阶段。"""
    if max_steps > 0:
        await game.advance_ready_ai_steps(max_steps=max_steps, ignore_delay=True)
    return game.to_snapshot()


async def snapshot_after_bounded_action(game, *, before_phase: str):
    """按入口限制自动推进范围，保留夜聊/夜晚/遗言等关键反馈。"""
    if before_phase == "wolf_chat":
        return await snapshot_after_action(game, max_steps=0)
    if before_phase == "night":
        return await snapshot_after_action(game, max_steps=1)
    return await snapshot_after_action(game)


async def snapshot_after_poll(game):
    """普通轮询在无人类可操作时也要把纯 AI 段推进过去，避免前端看起来卡死。"""
    await game.advance_timeout_if_needed()
    before_phase = game.phase
    if game.pending_human_action is None and not game._human_has_required_night_action():
        max_steps = (
            1
            if before_phase.value in {
                "wolf_chat",
                "night",
                "last_words",
                "hunter_shot",
                "day_speech",
                "exile_pk_speech",
                "sheriff_speech",
                "sheriff_pk_speech",
            }
            else 6
        )
        await game.advance_ready_ai_steps(max_steps=max_steps, ignore_delay=True)
    else:
        await game.advance_ready_ai_steps(max_steps=1)
    return game.to_snapshot()


@router.post("/games")
async def create_game(payload: CreateGameRequest):
    """创建游戏。"""
    game = game_manager.create_game(payload.player_count)
    async with game.operation_lock:
        return await snapshot_after_action(game, max_steps=0)


@router.get("/games/{game_id}")
async def get_game(game_id: str):
    """获取游戏快照。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        return await snapshot_after_poll(game)


@router.post("/games/{game_id}/night")
async def resolve_night(game_id: str, payload: NightRequest):
    """执行夜晚。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        before_phase = game.phase.value
        action = HumanNightAction(
            action_type=payload.action_type,
            target_id=payload.target_id,
            chat_content=payload.chat_content,
        )
        await game.resolve_night(action)
        return await snapshot_after_bounded_action(game, before_phase=before_phase)


@router.post("/games/{game_id}/wolf-chat")
async def resolve_wolf_chat(game_id: str, payload: NightRequest):
    """执行狼人协商阶段。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        before_phase = game.phase.value
        action = HumanNightAction(
            action_type=payload.action_type,
            target_id=payload.target_id,
            chat_content=payload.chat_content,
        )
        await game.resolve_wolf_chat(action)
        return await snapshot_after_bounded_action(game, before_phase=before_phase)


@router.post("/games/{game_id}/speech")
async def resolve_speech(game_id: str, payload: SpeechRequest):
    """执行逐位发言。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        await game.resolve_day_speeches(payload.content)
        # 真人发言后的首个返回只交付最新桌面快照，避免同一个请求里继续推进 AI，
        # 否则后置位会在还没看到你的发言刷新前就被自动说完。
        return await snapshot_after_action(game, max_steps=0)


@router.post("/games/{game_id}/last-words")
async def resolve_last_words(game_id: str, payload: SpeechRequest):
    """执行遗言。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        await game.resolve_last_words(payload.content)
        return await snapshot_after_action(game, max_steps=0)


@router.post("/games/{game_id}/self-destruct")
async def resolve_self_destruct(game_id: str):
    """执行狼人自爆。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        await game.resolve_self_destruct()
        return await snapshot_after_action(game)


@router.post("/games/{game_id}/sheriff")
async def resolve_sheriff(game_id: str, payload: SheriffRequest):
    """执行警长竞选与警徽相关动作。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        ensure_sheriff_enabled(game)
        action = SheriffAction(
            run_for_sheriff=payload.run_for_sheriff,
            vote_target_id=payload.vote_target_id,
            speech=payload.speech,
            speech_order_direction=payload.speech_order_direction,  # 兼容前端透传
            badge_target_id=payload.badge_target_id,
            tear_badge=payload.tear_badge,
        )
        await game.resolve_sheriff_election(action)
        return await snapshot_after_action(game)


@router.post("/games/{game_id}/speech-order")
async def choose_speech_order(game_id: str, payload: SheriffRequest):
    """警长选择发言方向。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        ensure_sheriff_enabled(game)
        await game.choose_speech_order(payload.speech_order_direction)
        return await snapshot_after_action(game)


@router.post("/games/{game_id}/badge")
async def resolve_badge(game_id: str, payload: SheriffRequest):
    """死亡警长处理警徽。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        ensure_sheriff_enabled(game)
        action = SheriffAction(
            badge_target_id=payload.badge_target_id,
            tear_badge=payload.tear_badge,
        )
        await game.resolve_badge_transfer(action)
        return await snapshot_after_action(game)


@router.post("/games/{game_id}/vote")
async def resolve_vote(game_id: str, payload: VoteRequest):
    """执行白天投票。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        await game.resolve_votes(payload.target_id)
        return await snapshot_after_action(game)


@router.post("/games/{game_id}/hunter-shot")
async def resolve_hunter_shot(game_id: str, payload: VoteRequest):
    """执行猎人开枪。"""
    game = get_game_or_404(game_id)
    async with game.operation_lock:
        await game.resolve_hunter_shot(payload.target_id)
        return await snapshot_after_action(game)
