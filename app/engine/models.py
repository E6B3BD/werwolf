"""狼人杀核心数据模型。"""
from __future__ import annotations

from enum import Enum
import uuid
from typing import Any, Literal, Optional

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
    EXILE_PK_SPEECH = "exile_pk_speech"
    EXILE_PK_VOTE = "exile_pk_vote"
    LAST_WORDS = "last_words"
    HUNTER_SHOT = "hunter_shot"
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


class RuleProfile(BaseModel):
    """当前对局规则配置。"""

    name: str = "12人预女猎白竞技规则"
    player_count: int = 12
    role_pool: list[RoleName] = Field(default_factory=lambda: ROLE_CONFIGS[12][:])
    sheriff_enabled: bool = False
    guard_enabled: bool = False
    wolf_chat_rounds: int = 2


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


class SeatRef(BaseModel):
    """供 Agent 使用的座位引用，避免 0-based id 和几号位混淆。"""

    player_id: int
    seat_no: int
    name: str
    alive: bool
    is_sheriff: bool = False
    role: Optional[RoleName] = None
    camp: Optional[Camp] = None


class PrivateObservation(BaseModel):
    """单个玩家私有信息条目。"""

    day: int
    phase: str
    content: str
    night_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    """单个 Agent 的长期可见记忆。"""

    player_id: int
    private_observations: list[PrivateObservation] = Field(default_factory=list)
    public_observations: list[PrivateObservation] = Field(default_factory=list)
    camp_observations: list[PrivateObservation] = Field(default_factory=list)
    last_seen_message_id: int = -1
    last_seen_event_seq: int = -1

    @property
    def observations(self) -> list[PrivateObservation]:
        """兼容旧测试/调用；新代码必须显式选择 private/public/camp。"""
        return [*self.private_observations, *self.camp_observations, *self.public_observations]


class CampSharedMemory(BaseModel):
    """阵营共享记忆。当前主要用于狼队夜聊。"""

    camp: Camp
    records: list[dict[str, Any]] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)


class PlayerAgentState(BaseModel):
    """单个玩家 Agent 的持续状态。"""

    player_id: int
    seat_no: int
    role: RoleName
    camp: Camp
    persona_style: str = ""
    strategy_style: str = ""
    private_summary: str = ""
    public_summary: str = ""
    current_focus: str = ""
    suspected_player_ids: list[int] = Field(default_factory=list)
    trusted_player_ids: list[int] = Field(default_factory=list)
    role_claim: str = ""
    last_internal_plan: str = ""
    last_public_position: str = ""
    memory_version: int = 0


class LegalAction(BaseModel):
    """规则引擎给 Agent 的合法动作约束。"""

    action_type: str
    target_ids: list[int] = Field(default_factory=list)
    target_seats: list[int] = Field(default_factory=list)
    required: bool = False
    note: str = ""


class PublicSpeechEvidence(BaseModel):
    """公开发言证据，供 Agent 引用具体话术。"""

    day: int
    speaker_id: int
    speaker_seat_no: int
    speech_type: str
    content: str
    mentioned_seat_nos: list[int] = Field(default_factory=list)
    stance_keywords: list[str] = Field(default_factory=list)


class VoteEvidence(BaseModel):
    """公开票型证据。"""

    day: int
    voter_id: int
    voter_seat_no: int
    target_id: int
    target_seat_no: int
    vote_type: str
    vote_round: str = ""


class PublicClaimEvidence(BaseModel):
    """公开身份宣称证据。"""

    day: int
    speaker_id: int
    speaker_seat_no: int
    claimed_role: RoleName
    source_text: str
    inspected_target_id: Optional[int] = None
    inspected_target_seat_no: Optional[int] = None
    inspected_result: Optional[str] = None


class SeerInspectionFact(BaseModel):
    """预言家私有查验事实，规则事实不依赖中文渲染文本。"""

    seer_id: int
    target_id: int
    target_seat_no: int
    result: Literal["狼人", "好人"]
    day: int
    night_id: int


class WitchNightInfo(BaseModel):
    """女巫夜间私有事实视图。"""

    witch_id: int
    day: int
    night_id: int
    wolf_target_id: Optional[int] = None
    wolf_target_seat_no: Optional[int] = None
    save_available: bool = False
    poison_available: bool = False
    can_save_target: bool = False
    can_self_save: bool = False


class WitchActionFact(BaseModel):
    """女巫行动事实，和前端/Agent中文提示解耦。"""

    witch_id: int
    day: int
    night_id: int
    wolf_target_id: Optional[int] = None
    saved_target_id: Optional[int] = None
    poison_target_id: Optional[int] = None
    save_available_after: bool
    poison_available_after: bool


class DeathFact(BaseModel):
    """单次死亡事实。猎人能否开枪必须跟随死亡原因，而不是全局状态。"""

    player_id: int
    seat_no: int
    cause: Literal["wolf_kill", "witch_poison", "exile", "hunter_shot", "self_destruct"]
    day: int
    night_id: Optional[int] = None
    source_player_id: Optional[int] = None
    can_hunter_shoot: bool = False


class IdiotRevealFact(BaseModel):
    """白痴翻牌事实。"""

    player_id: int
    seat_no: int
    day: int
    prevented_death: bool = True


class TableMessage(BaseModel):
    """统一桌面消息日志，供 Agent 视角过滤和复盘。"""

    message_id: int
    day: int
    night_id: int
    phase: str
    message_type: Literal["talk", "whisper", "vote", "night_action", "system", "last_words"]
    visibility: Literal["public", "wolf", "private", "audit"] = "public"
    speaker_id: Optional[int] = None
    speaker_seat_no: Optional[int] = None
    speaker_name: str = ""
    speaker_is_sheriff: bool = False
    round_id: Optional[int] = None
    turn_index: Optional[int] = None
    action: str = ""
    content: str = ""
    target_id: Optional[int] = None
    target_seat_no: Optional[int] = None
    target_role: Optional[RoleName] = None
    visible_to_player_ids: list[int] = Field(default_factory=list)
    created_at: float = 0.0


class AgentVisibleContext(BaseModel):
    """Agent 决策前可见的结构化上下文。"""

    self_player: SeatRef
    day: int
    night_id: int
    phase: str
    public_players: list[SeatRef]
    status_map: dict[int, str] = Field(default_factory=dict)
    known_role_map: dict[int, RoleName] = Field(default_factory=dict)
    talk_quota: dict[int, int] = Field(default_factory=dict)
    whisper_quota: dict[int, int] = Field(default_factory=dict)
    visible_messages: list[TableMessage] = Field(default_factory=list)
    new_visible_messages: list[TableMessage] = Field(default_factory=list)
    new_visible_events: list["GameEvent"] = Field(default_factory=list)
    private_observations: list[PrivateObservation] = Field(default_factory=list)
    recent_public_speeches: list[PublicSpeechEvidence] = Field(default_factory=list)
    recent_votes: list[VoteEvidence] = Field(default_factory=list)
    public_claims: list[PublicClaimEvidence] = Field(default_factory=list)
    seer_inspections: list[SeerInspectionFact] = Field(default_factory=list)
    witch_night_info: WitchNightInfo | None = None
    death_facts: list[DeathFact] = Field(default_factory=list)
    idiot_reveals: list[IdiotRevealFact] = Field(default_factory=list)
    legal_actions: list[LegalAction] = Field(default_factory=list)
    wolf_teammates: list[SeatRef] = Field(default_factory=list)
    wolf_chat_records: list["WolfChatRecord"] = Field(default_factory=list)
    wolf_history_summaries: list[str] = Field(default_factory=list)
    private_summary: str = ""
    public_summary: str = ""
    current_focus: str = ""


class DecisionAudit(BaseModel):
    """记录 Agent 决策校验结果，便于排查离谱行为。"""

    day: int
    phase: str
    player_id: int
    action: str
    requested_target_id: Optional[int] = None
    final_target_id: Optional[int] = None
    legal_target_ids: list[int] = Field(default_factory=list)
    corrected: bool = False
    reason: str = ""


class SpeechRecord(BaseModel):
    """发言记录。"""

    day: int
    player_id: int
    player_name: str
    content: str
    speech_type: Literal["campaign", "pk_campaign", "exile_pk", "day", "last_words"] = "day"


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
    night_id: int
    round_id: int = 1
    turn_index: int = 0
    player_id: int
    speaker_seat_no: int = 0
    player_name: str
    speaker_is_sheriff: bool = False
    content: str
    proposed_target_id: Optional[int] = None
    proposed_target_seat_no: Optional[int] = None
    stance_to_previous: str = ""
    reply_to_message_id: Optional[int] = None
    is_valid_target: bool = True
    created_at: float = 0.0


class WolfNightPlan(BaseModel):
    """当前狼队夜晚共识状态。"""

    day: int
    night_id: int
    current_target_id: Optional[int] = None
    supporters: list[int] = Field(default_factory=list)
    opponents: list[int] = Field(default_factory=list)
    final_confirmer_id: Optional[int] = None
    locked: bool = False
    finalized: bool = False
    final_source: str = ""


class NightSummary(BaseModel):
    """夜晚摘要。"""

    day: int
    night_id: int | None = None
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
    visibility: Literal["public", "wolf", "private", "audit"] = "public"
    day: int | None = None
    night_id: int | None = None
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    seq: int = 0
    created_at: float = 0.0
    occurrence_key: str = ""
    visible_to_player_ids: list[int] = Field(default_factory=list)


class VisibleTimelineItem(BaseModel):
    """按玩家视角过滤后的统一可见时间线。"""

    item_id: str
    kind: Literal["event", "message", "speech", "vote", "night_summary", "wolf_chat"]
    day: int | None = None
    night_id: int | None = None
    phase: str = ""
    visibility: Literal["public", "wolf", "private", "audit"] = "public"
    order: float = 0.0
    speaker_id: Optional[int] = None
    speaker_seat_no: Optional[int] = None
    speaker_name: str = ""
    speaker_is_sheriff: bool = False
    message_type: str = ""
    action: str = ""
    content: str = ""
    target_id: Optional[int] = None
    target_seat_no: Optional[int] = None
    occurrence_key: str = ""


class AgentDecision(BaseModel):
    """AI 玩家结构化决策。"""

    action: Literal[
        "speak",
        "vote",
        "night_action",
        "wolf_chat",
        "wolf_confirm",
        "sheriff",
        "hunter_shot",
        "badge_transfer",
    ]
    target_id: Optional[int] = None
    content: str = ""
    reason: str = ""
    action_type: str = ""


class HumanNightAction(BaseModel):
    """真人夜间动作。"""

    action_type: Literal["guard", "inspect", "save", "poison", "skip", "wolf_kill", "wolf_chat", "wolf_confirm"]
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
    snapshot_seq: int = 0
    phase: Phase
    day: int
    night_id: int
    sheriff_enabled: bool = False
    guard_enabled: bool = False
    human_player_id: int
    human_role: RoleName
    human_alive: bool
    winner: Optional[str]
    human_private_message: str = ""
    current_hint: str = ""
    human_private_context: str = ""
    human_allowed_night_actions: list[str] = Field(default_factory=list)
    human_target_candidates: list[int] = Field(default_factory=list)
    sheriff_id: Optional[int] = None
    sheriff_candidates: list[int] = Field(default_factory=list)
    human_is_wolf: bool = False
    wolf_teammate_ids: list[int] = Field(default_factory=list)
    wolf_chat_records: list[WolfChatRecord] = Field(default_factory=list)
    wolf_history_summaries: list[str] = Field(default_factory=list)
    wolf_night_plan: WolfNightPlan | None = None
    players: list[PlayerState]
    speeches: list[SpeechRecord]
    votes: list[VoteRecord]
    night_summaries: list[NightSummary]
    events: list[GameEvent]
    visible_timeline: list[VisibleTimelineItem] = Field(default_factory=list)
    pending_human_action: Optional[str]
    current_speaker_id: Optional[int] = None
    speech_order: list[int] = Field(default_factory=list)
    can_self_destruct: bool = False
    available_speech_directions: list[str] = Field(default_factory=list)
    timer_label: str = ""
    time_limit_seconds: int = 0
    remaining_seconds: int = 0
    deadline_ts: float | None = None
