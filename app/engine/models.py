"""狼人杀核心数据模型。"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Camp(str, Enum):
    """阵营。"""

    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    THIRD_PARTY = "third_party"


class Phase(str, Enum):
    """游戏大阶段。"""

    SETUP = "setup"
    WOLF_CHAT = "wolf_chat"
    NIGHT = "night"
    SHERIFF_ELECTION = "sheriff_election"
    SHERIFF_SPEECH = "sheriff_speech"
    SHERIFF_VOTE = "sheriff_vote"
    SHERIFF_PK_SPEECH = "sheriff_pk_speech"
    SHERIFF_PK_VOTE = "sheriff_pk_vote"
    LAST_WORDS = "last_words"
    BADGE_TRANSFER = "badge_transfer"
    DAY_SPEECH = "day_speech"
    DAY_VOTE = "day_vote"
    GAME_OVER = "game_over"


class RoleName(str, Enum):
    """当前版本支持的角色。"""

    WEREWOLF = "狼人"
    SEER = "预言家"
    WITCH = "女巫"
    HUNTER = "猎人"
    GUARD = "守卫"
    IDIOT = "白痴"
    VILLAGER = "平民"


ROLE_CAMP = {
    RoleName.WEREWOLF: Camp.WEREWOLF,
    RoleName.SEER: Camp.VILLAGER,
    RoleName.WITCH: Camp.VILLAGER,
    RoleName.HUNTER: Camp.VILLAGER,
    RoleName.GUARD: Camp.VILLAGER,
    RoleName.IDIOT: Camp.VILLAGER,
    RoleName.VILLAGER: Camp.VILLAGER,
}


ROLE_CONFIGS: dict[int, list[RoleName]] = {
    6: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.VILLAGER,
    ],
    7: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.GUARD,
        RoleName.VILLAGER,
    ],
    8: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.GUARD,
        RoleName.VILLAGER,
    ],
    9: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.GUARD,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
    ],
    10: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.GUARD,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
    ],
    11: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.GUARD,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
    ],
    12: [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.IDIOT,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
    ],
}


class PlayerState(BaseModel):
    """玩家状态。"""

    id: int
    name: str
    role: RoleName
    camp: Camp
    is_human: bool = False
    alive: bool = True
    can_vote: bool = True
    is_sheriff: bool = False
    idiot_revealed: bool = False
    last_speech: str = ""
    private_note: str = ""
    persona_style: str = ""
    strategy_style: str = ""


class SpeechRecord(BaseModel):
    """发言记录。"""

    day: int
    player_id: int
    player_name: str
    content: str
    speech_type: Literal["campaign", "pk_campaign", "day", "last_words"] = "day"


class VoteRecord(BaseModel):
    """投票记录。"""

    day: int
    voter_id: int
    voter_name: str
    target_id: int
    target_name: str
    vote_type: Literal["sheriff", "exile"] = "exile"
    vote_round: str = ""


class WolfChatRecord(BaseModel):
    """狼人夜晚聊天记录。"""

    day: int
    player_id: int
    player_name: str
    content: str
    proposed_target_id: Optional[int] = None


class NightSummary(BaseModel):
    """夜晚摘要。"""

    day: int
    wolf_target_id: Optional[int] = None
    guard_target_id: Optional[int] = None
    seer_target_id: Optional[int] = None
    seer_result: Optional[str] = None
    witch_saved: bool = False
    witch_poison_target_id: Optional[int] = None
    deaths: list[int] = Field(default_factory=list)


class GameEvent(BaseModel):
    """前端展示事件。"""

    phase: str
    message: str


class AgentDecision(BaseModel):
    """AI 玩家结构化决策。"""

    action: Literal["speak", "vote", "night_action"]
    target_id: Optional[int] = None
    content: str = ""
    reason: str = ""


class HumanNightAction(BaseModel):
    """真人夜间动作。"""

    action_type: Literal["guard", "inspect", "save", "poison", "skip", "wolf_kill"]
    target_id: Optional[int] = None
    chat_content: str = ""


class SheriffAction(BaseModel):
    """警长相关动作。"""

    run_for_sheriff: bool = False
    vote_target_id: int | None = None
    speech: str = ""
    speech_order_direction: Literal["left", "right"] | None = None
    badge_target_id: int | None = None
    tear_badge: bool = False


class GameSnapshot(BaseModel):
    """发给前端的公开快照。"""

    game_id: str
    phase: Phase
    day: int
    human_player_id: int
    human_role: RoleName
    human_alive: bool
    winner: Optional[str]
    human_private_message: str = ""
    current_hint: str = ""
    human_allowed_night_actions: list[str] = Field(default_factory=list)
    human_target_candidates: list[int] = Field(default_factory=list)
    sheriff_id: Optional[int] = None
    sheriff_candidates: list[int] = Field(default_factory=list)
    human_is_wolf: bool = False
    wolf_teammate_ids: list[int] = Field(default_factory=list)
    wolf_chat_records: list[WolfChatRecord] = Field(default_factory=list)
    players: list[PlayerState]
    speeches: list[SpeechRecord]
    votes: list[VoteRecord]
    night_summaries: list[NightSummary]
    events: list[GameEvent]
    pending_human_action: Optional[str]
    current_speaker_id: Optional[int] = None
    speech_order: list[int] = Field(default_factory=list)
    can_self_destruct: bool = False
    available_speech_directions: list[str] = Field(default_factory=list)
    timer_label: str = ""
    time_limit_seconds: int = 0
    remaining_seconds: int = 0
    deadline_ts: float | None = None
