"""狼人杀规则引擎。"""
from __future__ import annotations

import random
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field

from app.agents.runtime import AIContext, OpenAIAgentRuntime
from app.engine.models import (
    Camp,
    GameEvent,
    GameSnapshot,
    HumanNightAction,
    NightSummary,
    Phase,
    PlayerState,
    ROLE_CAMP,
    ROLE_CONFIGS,
    RoleName,
    SheriffAction,
    SpeechRecord,
    VoteRecord,
    WolfChatRecord,
)


def build_default_names(count: int) -> list[str]:
    """生成默认玩家名。"""
    return [f"玩家{i + 1}" for i in range(count)]


PERSONA_STYLES = [
    "铁腕控场型",
    "冷刀拆解型",
    "毒舌挑刺型",
    "深水潜伏型",
    "情绪爆燃型",
    "圆滑拿捏型",
    "票型偏执型",
    "伪善安抚型",
    "赌徒冲锋型",
    "记仇回踩型",
    "高傲优越型",
    "谨慎求稳型",
    "反骨抬杠型",
]

ROLE_STRATEGY_STYLES = {
    RoleName.WEREWOLF: ["控场悍跳流", "深水倒钩流", "冲票做局流"],
    RoleName.SEER: ["强预带队流", "稳预控场流", "藏锋反打流"],
    RoleName.WITCH: ["轮次收益流", "藏毒等待流", "强博弈反制流"],
    RoleName.HUNTER: ["隐忍带枪流", "压场威慑流", "残局定胜流"],
    RoleName.IDIOT: ["装民抗推流", "翻牌反打流", "稳视角补位流"],
    RoleName.VILLAGER: ["逻辑平民流", "票型平民流", "听感搏杀流"],
}


@dataclass(slots=True)
class WitchState:
    """女巫资源状态。"""

    save_available: bool = True
    poison_available: bool = True


@dataclass(slots=True)
class WerwolfGame:
    """单局游戏状态机。"""

    player_count: int
    human_player_id: int
    game_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    day: int = 1
    phase: Phase = Phase.SETUP
    winner: str | None = None
    players: list[PlayerState] = field(default_factory=list)
    speeches: list[SpeechRecord] = field(default_factory=list)
    votes: list[VoteRecord] = field(default_factory=list)
    night_summaries: list[NightSummary] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    pending_human_action: str | None = None
    witch_state: WitchState = field(default_factory=WitchState)
    guard_last_target_id: int | None = None
    last_human_seer_result: str = ""
    wolf_chat_records: list[WolfChatRecord] = field(default_factory=list)
    wolf_consensus_target_id: int | None = None
    sheriff_id: int | None = None
    sheriff_candidate_ids: list[int] = field(default_factory=list)
    sheriff_vote_tally: dict[int, float] = field(default_factory=dict)
    sheriff_pk_candidate_ids: list[int] = field(default_factory=list)
    exile_pk_candidate_ids: list[int] = field(default_factory=list)
    speech_order: list[int] = field(default_factory=list)
    speech_cursor: int = 0
    last_words_queue: list[int] = field(default_factory=list)
    current_exile_target_id: int | None = None
    last_night_deaths: list[int] = field(default_factory=list)
    first_day_death_announcement_pending: bool = False
    hunter_poisoned: bool = False
    timer_label: str = ""
    time_limit_seconds: int = 0
    deadline_ts: float | None = None
    timer_signature: str = ""
    auto_step_ready_ts: float = 0.0
    runtime: OpenAIAgentRuntime = field(default_factory=OpenAIAgentRuntime)

    @classmethod
    def create(cls, player_count: int) -> "WerwolfGame":
        """创建新对局。"""
        if player_count not in ROLE_CONFIGS:
            raise ValueError("当前版本支持 6-12 人局。")

        human_player_id = random.randint(0, player_count - 1)
        roles = ROLE_CONFIGS[player_count][:]
        random.shuffle(roles)
        personas = random.sample(PERSONA_STYLES, k=min(player_count, len(PERSONA_STYLES)))
        while len(personas) < player_count:
            personas.append(random.choice(PERSONA_STYLES))
        players = []
        for idx, role in enumerate(roles):
            players.append(
                PlayerState(
                    id=idx,
                    name=build_default_names(player_count)[idx],
                    role=role,
                    camp=ROLE_CAMP[role],
                    is_human=idx == human_player_id,
                    persona_style=personas[idx],
                    strategy_style=random.choice(ROLE_STRATEGY_STYLES[role]),
                )
            )

        game = cls(player_count=player_count, human_player_id=human_player_id, players=players)
        game.events.append(GameEvent(phase="setup", message="游戏创建完成，已随机分配角色，当前进入首夜狼人协商阶段。"))
        game.phase = Phase.WOLF_CHAT
        game._prepare_wolf_chat_order()
        game._refresh_timer_state()
        return game

    @property
    def human_player(self) -> PlayerState:
        """获取真人玩家。"""
        return self.players[self.human_player_id]

    def alive_players(self) -> list[PlayerState]:
        """所有存活玩家。"""
        return [player for player in self.players if player.alive]

    def alive_wolves(self) -> list[PlayerState]:
        """所有存活狼人。"""
        return [player for player in self.alive_players() if player.camp == Camp.WEREWOLF]

    def alive_villagers(self) -> list[PlayerState]:
        """所有存活好人。"""
        return [player for player in self.alive_players() if player.camp == Camp.VILLAGER]

    def public_state_text(self) -> str:
        """为 AI 生成公开局面文本。"""
        alive_desc = []
        hide_deaths = self._should_hide_first_day_deaths()
        for player in self.players:
            status = "存活" if hide_deaths else ("存活" if player.alive else "死亡")
            sheriff = " 警长" if player.is_sheriff else ""
            alive_desc.append(f"{player.id}: {player.name} - {status}{sheriff}")

        last_speeches = [f"{record.player_name}: {record.content}" for record in self.speeches[-10:]]
        recent_votes = [
            f"第{vote.day}天{vote.voter_name} -> {vote.target_name}（{vote.vote_type}）"
            for vote in self.votes[-10:]
        ]
        recent_events = [f"{event.phase}: {event.message}" for event in self.events[-8:]]
        return "\n".join(
            [
                f"当前天数：第 {self.day} 天",
                f"阶段：{self.phase.value}",
                "玩家状态：",
                *alive_desc,
                "最近发言：",
                *(last_speeches or ["暂无发言记录"]),
                "最近票型：",
                *(recent_votes or ["暂无公开投票记录"]),
                "最近系统播报：",
                *(recent_events or ["暂无系统播报"]),
            ]
        )

    def _player_private_context(self, player: PlayerState) -> str:
        """构建玩家私有信息。"""
        notes: list[str] = []
        if player.private_note:
            notes.append(player.private_note)
        if player.is_human and self.last_human_seer_result:
            notes.append(self.last_human_seer_result)
        return "\n".join(notes) if notes else "暂无私有信息。"

    def _arm_auto_step_delay(self, seconds: float = 2.2) -> None:
        """给下一次自动推进设置短暂停顿，避免节奏过快。"""
        self.auto_step_ready_ts = time.time() + seconds

    def _announce_last_night_deaths_if_needed(self) -> None:
        """在需要时公布上一夜死讯。"""
        if self.day == 1 and not self.first_day_death_announcement_pending:
            return
        if self.last_night_deaths:
            death_names = "、".join(self.players[player_id].name for player_id in self.last_night_deaths)
            self.events.append(GameEvent(phase="night", message=f"昨夜死亡：{death_names}"))
        else:
            self.events.append(GameEvent(phase="night", message="昨夜是平安夜。"))
        self.first_day_death_announcement_pending = False

    def _should_hide_first_day_deaths(self) -> bool:
        """首日警长流程期间，对外隐藏首夜死亡状态。"""
        return self.first_day_death_announcement_pending and self.phase in {
            Phase.SHERIFF_ELECTION,
            Phase.SHERIFF_SPEECH,
            Phase.SHERIFF_VOTE,
            Phase.SHERIFF_PK_SPEECH,
            Phase.SHERIFF_PK_VOTE,
        }

    def _build_snapshot_players(self) -> list[PlayerState]:
        """构建给前端的公开玩家视图。"""
        if self.phase == Phase.GAME_OVER:
            return [player.model_copy(deep=True) for player in self.players]

        hide_deaths = self._should_hide_first_day_deaths()
        snapshot_players: list[PlayerState] = []
        for player in self.players:
            snapshot_player = player.model_copy(deep=True)
            if hide_deaths:
                snapshot_player.alive = True
            snapshot_players.append(snapshot_player)
        return snapshot_players

    def _build_snapshot_night_summaries(self) -> list[NightSummary]:
        """构建给前端的公开夜晚摘要。"""
        if self.phase == Phase.GAME_OVER:
            return [NightSummary.model_validate(deepcopy(summary.model_dump())) for summary in self.night_summaries]

        hide_deaths = self._should_hide_first_day_deaths()
        public_summaries: list[NightSummary] = []
        for summary in self.night_summaries:
            public_summaries.append(
                NightSummary(
                    day=summary.day,
                    wolf_target_id=None,
                    guard_target_id=None,
                    seer_target_id=None,
                    seer_result=None,
                    witch_saved=False,
                    witch_poison_target_id=None,
                    deaths=[] if hide_deaths and summary.day == 1 else list(summary.deaths),
                )
            )
        return public_summaries

    def to_snapshot(self) -> GameSnapshot:
        """对前端返回游戏快照。"""
        self._refresh_timer_state()
        human_target_candidates = self._get_human_target_candidates()
        snapshot_players = self._build_snapshot_players()
        wolf_teammate_ids = []
        if self.human_player.camp == Camp.WEREWOLF:
            wolf_teammate_ids = [
                player.id
                for player in self.players
                if player.id != self.human_player_id and player.camp == Camp.WEREWOLF
            ]
        return GameSnapshot(
            game_id=self.game_id,
            phase=self.phase,
            day=self.day,
            human_player_id=self.human_player_id,
            human_role=self.human_player.role,
            human_alive=self.human_player.alive,
            human_is_wolf=self.human_player.camp == Camp.WEREWOLF,
            wolf_teammate_ids=wolf_teammate_ids,
            winner=self.winner,
            human_private_message=self.last_human_seer_result,
            current_hint=self._build_current_hint(),
            human_allowed_night_actions=self._get_human_allowed_night_actions(),
            human_target_candidates=human_target_candidates,
            sheriff_id=self.sheriff_id,
            sheriff_candidates=self._preview_sheriff_candidates(),
            wolf_chat_records=self.wolf_chat_records,
            players=snapshot_players,
            speeches=self.speeches,
            votes=self.votes,
            night_summaries=self._build_snapshot_night_summaries(),
            events=self.events[-30:],
            pending_human_action=self.pending_human_action,
            current_speaker_id=self.current_speaker_id,
            speech_order=self.speech_order,
            can_self_destruct=self._can_human_self_destruct(),
            available_speech_directions=self._available_speech_directions(),
            timer_label=self.timer_label,
            time_limit_seconds=self.time_limit_seconds,
            remaining_seconds=self._remaining_seconds(),
            deadline_ts=self.deadline_ts,
        )

    @property
    def current_speaker_id(self) -> int | None:
        """当前轮到发言的人。"""
        if 0 <= self.speech_cursor < len(self.speech_order):
            return self.speech_order[self.speech_cursor]
        return None

    def _get_human_allowed_night_actions(self) -> list[str]:
        """返回真人夜晚可执行动作。"""
        role = self.human_player.role
        if not self.human_player.alive:
            return []
        if self.phase == Phase.WOLF_CHAT and role == RoleName.WEREWOLF:
            return ["wolf_kill"]
        if self.phase != Phase.NIGHT:
            return []
        if role == RoleName.SEER:
            return ["inspect", "skip"]
        if role == RoleName.GUARD:
            return ["guard", "skip"]
        if role == RoleName.WITCH:
            actions = ["skip"]
            if self.witch_state.save_available and self._witch_can_save_target(self.human_player_id, self.wolf_consensus_target_id):
                actions.append("save")
            if self.witch_state.poison_available:
                actions.append("poison")
            return actions
        return ["skip"]

    def _get_human_target_candidates(self) -> list[int]:
        """返回真人当前可选目标。"""
        if not self.human_player.alive:
            return []

        if self.phase == Phase.DAY_VOTE:
            return [player.id for player in self.alive_players() if player.id != self.human_player_id]

        if self.phase == Phase.WOLF_CHAT:
            if self.human_player.role == RoleName.WEREWOLF:
                return [player.id for player in self.alive_players() if player.camp != Camp.WEREWOLF]
            return []

        if self.phase == Phase.NIGHT:
            role = self.human_player.role
            if role == RoleName.GUARD:
                return [player.id for player in self.alive_players() if player.id != self.guard_last_target_id]
            if role == RoleName.SEER:
                return [player.id for player in self.alive_players() if player.id != self.human_player_id]
            if role == RoleName.WITCH:
                candidates = [player.id for player in self.alive_players() if player.id != self.human_player_id]
                if self.wolf_consensus_target_id is not None and self.wolf_consensus_target_id not in candidates:
                    candidates.append(self.wolf_consensus_target_id)
                return candidates
            return []

        if self.phase == Phase.SHERIFF_VOTE:
            return [player_id for player_id in self.sheriff_candidate_ids]
        if self.phase == Phase.SHERIFF_PK_VOTE:
            return [player_id for player_id in self.sheriff_pk_candidate_ids]

        if self.phase == Phase.BADGE_TRANSFER:
            return [
                player.id
                for player in self.alive_players()
                if player.id != self.human_player_id and player.camp == Camp.VILLAGER
            ]

        return [player.id for player in self.alive_players() if player.id != self.human_player_id]

    def _build_current_hint(self) -> str:
        """构建当前阶段提示语。"""
        if self.phase == Phase.GAME_OVER:
            return f"本局已结束，获胜方：{self.winner or '未知'}。"
        if not self.human_player.alive and self.phase not in {Phase.LAST_WORDS, Phase.BADGE_TRANSFER}:
            return "你已经出局，可以继续观战并查看后续日志。"
        if self.phase == Phase.WOLF_CHAT:
            if self.human_player.role == RoleName.WEREWOLF:
                if self.current_speaker_id == self.human_player_id:
                    return "狼人协商阶段：轮到你发言并给出刀口建议。"
                if self.current_speaker_id is not None:
                    return f"狼人协商阶段：当前轮到 {self.players[self.current_speaker_id].name} 和狼队交流。"
                return "狼人协商阶段：等待狼队完成这一轮协商。"
            return "狼人协商阶段：你不是狼人，等待狼队完成夜间讨论。"
        if self.phase == Phase.NIGHT:
            role = self.human_player.role
            if role == RoleName.SEER:
                return "夜晚阶段：你是预言家，请选择一名玩家查验。"
            if role == RoleName.GUARD:
                return "夜晚阶段：你是守卫，请选择一名玩家守护。"
            if role == RoleName.WITCH:
                return "夜晚阶段：你是女巫，可以选择救人、毒人或跳过。"
            return "夜晚阶段：你的角色今晚没有主动技能，系统会自动按跳过处理。"
        if self.phase == Phase.SHERIFF_ELECTION:
            return "上警报名阶段：决定你是否竞选警长。"
        if self.phase == Phase.SHERIFF_SPEECH:
            if self.current_speaker_id == self.human_player_id:
                return "警上发言轮到你了，请发表竞选发言。"
            return f"警上发言阶段：当前轮到 {self.players[self.current_speaker_id].name} 发言。"
        if self.phase == Phase.SHERIFF_PK_SPEECH:
            if self.current_speaker_id == self.human_player_id:
                return "警长 PK 发言轮到你了，请做最后陈述。"
            return f"警长 PK 发言阶段：当前轮到 {self.players[self.current_speaker_id].name} 发言。"
        if self.phase == Phase.SHERIFF_VOTE:
            return "警长投票阶段：未上警玩家对警上玩家投票。"
        if self.phase == Phase.SHERIFF_PK_VOTE:
            return "警长 PK 投票阶段：警下玩家重新投票。"
        if self.phase == Phase.DAY_SPEECH:
            if self.sheriff_id == self.human_player_id and not self.speech_order:
                return "你是警长，请先选择白天发言顺序方向。"
            if self.current_speaker_id == self.human_player_id:
                return "白天逐位发言轮到你，请输入你的发言。"
            if self.current_speaker_id is not None:
                return f"白天逐位发言中，当前轮到 {self.players[self.current_speaker_id].name}。"
            return "白天发言阶段。"
        if self.phase == Phase.DAY_VOTE:
            return "白天投票阶段：请选择一名你要放逐的玩家。"
        if self.phase == Phase.LAST_WORDS:
            if self.current_exile_target_id == self.human_player_id:
                return "你已出局，请发表遗言。"
            if self.current_exile_target_id is not None:
                return f"{self.players[self.current_exile_target_id].name} 正在发表遗言。"
            return "遗言阶段。"
        if self.phase == Phase.BADGE_TRANSFER:
            if self.current_exile_target_id == self.human_player_id:
                return "你是死亡警长，请选择移交警徽或撕毁警徽。"
            return "警徽移交阶段。"
        return "准备开始。"

    def _preview_sheriff_candidates(self) -> list[int]:
        """返回当前可预览的警长候选人。"""
        if self.sheriff_candidate_ids:
            return self.sheriff_candidate_ids
        return []

    def _available_speech_directions(self) -> list[str]:
        """返回警长可选发言方向。"""
        if (
            self.phase == Phase.DAY_SPEECH
            and not self.speech_order
            and self.sheriff_id == self.human_player_id
            and self.human_player.alive
        ):
            return ["left", "right"]
        return []

    def _remaining_seconds(self) -> int:
        """返回剩余秒数。"""
        if self.deadline_ts is None or self.time_limit_seconds <= 0:
            return 0
        return max(0, int(self.deadline_ts - time.time()))

    def _timer_config(self) -> tuple[str, str, int] | None:
        """根据当前阶段给出限时配置。"""
        if self.phase == Phase.WOLF_CHAT:
            return (f"wolf_chat_{self.day}", "狼人夜聊限时", 45)
        if self.phase == Phase.NIGHT:
            return (f"night_{self.day}", "夜间操作限时", 45)
        if self.phase == Phase.SHERIFF_ELECTION:
            return (f"sheriff_election_{self.day}", "上警报名限时", 45)
        if self.phase == Phase.SHERIFF_SPEECH and self.current_speaker_id is not None:
            speaker = self.players[self.current_speaker_id]
            return (f"sheriff_speech_{self.day}_{self.speech_cursor}", f"{speaker.name} 警上发言限时", 75)
        if self.phase == Phase.SHERIFF_PK_SPEECH and self.current_speaker_id is not None:
            speaker = self.players[self.current_speaker_id]
            return (f"sheriff_pk_speech_{self.day}_{self.speech_cursor}", f"{speaker.name} PK发言限时", 60)
        if self.phase in {Phase.SHERIFF_VOTE, Phase.SHERIFF_PK_VOTE}:
            return (f"{self.phase.value}_{self.day}", "投票限时", 45)
        if self.phase == Phase.DAY_SPEECH and self.sheriff_id == self.human_player_id and not self.speech_order:
            return (f"speech_order_{self.day}", "发言顺序选择限时", 30)
        if self.phase == Phase.DAY_SPEECH and self.current_speaker_id is not None:
            speaker = self.players[self.current_speaker_id]
            return (f"day_speech_{self.day}_{self.speech_cursor}", f"{speaker.name} 白天发言限时", 120)
        if self.phase == Phase.DAY_VOTE:
            return (f"day_vote_{self.day}", "投票限时", 45)
        if self.phase == Phase.LAST_WORDS and self.current_exile_target_id is not None:
            speaker = self.players[self.current_exile_target_id]
            return (f"last_words_{self.day}_{self.current_exile_target_id}", f"{speaker.name} 遗言限时", 90)
        if self.phase == Phase.BADGE_TRANSFER and self.current_exile_target_id is not None:
            speaker = self.players[self.current_exile_target_id]
            return (f"badge_transfer_{self.day}_{self.current_exile_target_id}", f"{speaker.name} 警徽处理限时", 45)
        return None

    def _refresh_timer_state(self) -> None:
        """刷新当前阶段倒计时。"""
        config = self._timer_config()
        if config is None:
            self.timer_label = ""
            self.time_limit_seconds = 0
            self.deadline_ts = None
            self.timer_signature = ""
            return
        signature, label, seconds = config
        if signature != self.timer_signature:
            self.timer_signature = signature
            self.timer_label = label
            self.time_limit_seconds = seconds
            self.deadline_ts = time.time() + seconds

    async def advance_timeout_if_needed(self) -> None:
        """若超时则按默认动作自动推进。"""
        self._refresh_timer_state()
        if self.deadline_ts is not None and self._remaining_seconds() <= 0:
            await self._apply_timeout_default_action()
            self._refresh_timer_state()

    async def advance_ready_ai_step_if_needed(self) -> None:
        """如果当前轮到 AI，可立即推进一步，而不必等倒计时耗尽。"""
        if self.phase == Phase.GAME_OVER:
            return
        if time.time() < self.auto_step_ready_ts:
            return
        if self.phase == Phase.SHERIFF_ELECTION and (not self.human_player.alive or self.pending_human_action is None):
            await self.resolve_sheriff_election(SheriffAction(run_for_sheriff=False))
            self._arm_auto_step_delay()
            return
        if self.phase == Phase.SHERIFF_VOTE and (not self.human_player.alive or self.pending_human_action is None):
            candidate_ids = self._get_human_target_candidates()
            await self.resolve_sheriff_election(
                SheriffAction(vote_target_id=candidate_ids[0] if candidate_ids else None)
            )
            self._arm_auto_step_delay()
            return
        if self.phase == Phase.SHERIFF_PK_VOTE and (not self.human_player.alive or self.pending_human_action is None):
            candidate_ids = self._get_human_target_candidates()
            await self.resolve_sheriff_election(
                SheriffAction(vote_target_id=candidate_ids[0] if candidate_ids else None)
            )
            self._arm_auto_step_delay()
            return
        if self.phase == Phase.NIGHT and self.pending_human_action is None:
            await self.resolve_night(HumanNightAction(action_type="skip", target_id=None))
            self._arm_auto_step_delay()
            return
        if self.phase == Phase.SHERIFF_SPEECH and self.current_speaker_id is not None:
            if self.current_speaker_id != self.human_player_id:
                await self._advance_sheriff_speech("")
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.SHERIFF_PK_SPEECH and self.current_speaker_id is not None:
            if self.current_speaker_id != self.human_player_id:
                await self._advance_sheriff_pk_speech("")
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.DAY_SPEECH and self.current_speaker_id is not None:
            if self.current_speaker_id != self.human_player_id:
                await self.resolve_day_speeches("")
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.LAST_WORDS and self.current_exile_target_id is not None:
            if self.current_exile_target_id != self.human_player_id:
                await self.resolve_last_words("")
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.BADGE_TRANSFER and self.current_exile_target_id is not None:
            if self.current_exile_target_id != self.human_player_id:
                await self.resolve_badge_transfer(None)
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.DAY_VOTE and (not self.human_player.alive or self.pending_human_action is None or not self.human_player.can_vote):
            candidates = self._get_human_target_candidates()
            if candidates:
                await self.resolve_votes(candidates[0])
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.WOLF_CHAT and self.current_speaker_id is not None:
            if self.current_speaker_id != self.human_player_id:
                await self.resolve_wolf_chat(None)
                self._arm_auto_step_delay()

    async def _apply_timeout_default_action(self) -> None:
        """对超时阶段执行默认动作。"""
        if self.phase == Phase.WOLF_CHAT:
            candidates = self._get_human_target_candidates()
            action = None
            if self.human_player.alive and self.human_player.role == RoleName.WEREWOLF:
                action = HumanNightAction(
                    action_type="wolf_kill",
                    target_id=candidates[0] if candidates else None,
                    chat_content="先过。",
                )
            await self.resolve_wolf_chat(action)
            return
        if self.phase == Phase.NIGHT:
            action = None
            if self.human_player.alive:
                action = HumanNightAction(action_type="skip", target_id=None)
            await self.resolve_night(action)
            return
        if self.phase == Phase.SHERIFF_ELECTION:
            await self.resolve_sheriff_election(SheriffAction(run_for_sheriff=False))
            return
        if self.phase in {Phase.SHERIFF_SPEECH, Phase.SHERIFF_PK_SPEECH}:
            await self.resolve_sheriff_election(SheriffAction(speech="过。"))
            return
        if self.phase in {Phase.SHERIFF_VOTE, Phase.SHERIFF_PK_VOTE}:
            candidates = self._get_human_target_candidates()
            await self.resolve_sheriff_election(SheriffAction(vote_target_id=candidates[0] if candidates else None))
            return
        if self.phase == Phase.DAY_SPEECH and self.sheriff_id == self.human_player_id and not self.speech_order:
            await self.choose_speech_order("right")
            return
        if self.phase == Phase.DAY_SPEECH and self.current_speaker_id is not None:
            await self.resolve_day_speeches("过。")
            return
        if self.phase == Phase.DAY_VOTE:
            candidates = self._get_human_target_candidates()
            if candidates:
                await self.resolve_votes(candidates[0])
            return
        if self.phase == Phase.LAST_WORDS and self.current_exile_target_id is not None:
            await self.resolve_last_words("过。")
            return
        if self.phase == Phase.BADGE_TRANSFER and self.current_exile_target_id is not None:
            await self.resolve_badge_transfer(SheriffAction(tear_badge=True))
            return

    def _can_human_self_destruct(self) -> bool:
        """真人是否可以自爆。"""
        return (
            self.human_player.alive
            and self.human_player.camp == Camp.WEREWOLF
            and self.phase in {Phase.DAY_SPEECH, Phase.SHERIFF_SPEECH}
        )

    def _prepare_wolf_chat_order(self) -> None:
        """准备本轮狼人夜聊顺序。"""
        wolves = [player.id for player in self.alive_wolves()]
        self.speech_order = wolves
        self.speech_cursor = 0
        self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None

    async def resolve_wolf_chat(self, human_action: HumanNightAction | None = None) -> None:
        """执行狼人协商阶段。"""
        if self.phase not in (Phase.WOLF_CHAT, Phase.SETUP):
            return

        alive = self.alive_players()
        wolves = [player for player in alive if player.camp == Camp.WEREWOLF]
        non_wolves = [player for player in alive if player.camp != Camp.WEREWOLF]
        candidate_ids = [player.id for player in non_wolves]

        if not wolves:
            self.phase = Phase.NIGHT
            return

        if not self.speech_order:
            self._prepare_wolf_chat_order()
            if not self.wolf_chat_records:
                self.events.append(GameEvent(phase="wolf_chat", message="狼人开始逐人夜聊协商刀口。"))

        speaker_id = self.current_speaker_id
        if speaker_id is None:
            proposals = [record.proposed_target_id for record in self.wolf_chat_records if record.proposed_target_id is not None]
            if proposals:
                self.wolf_consensus_target_id = max(set(proposals), key=proposals.count)
                self.events.append(GameEvent(phase="wolf_chat", message="狼人夜谈结束，狼队已完成今晚刀口协商。"))
            else:
                self.wolf_consensus_target_id = None
                self.events.append(GameEvent(phase="wolf_chat", message="狼人夜谈结束，但未形成有效刀人目标。"))
            self.phase = Phase.NIGHT
            self.pending_human_action = None
            self.speech_order = []
            self.speech_cursor = 0
            self._arm_auto_step_delay(1.4)
            return

        speaker = self.players[speaker_id]
        if speaker.is_human:
            if human_action and human_action.action_type == "wolf_kill" and human_action.target_id in candidate_ids:
                proposed_target_id = human_action.target_id
            else:
                proposed_target_id = candidate_ids[0] if candidate_ids else None
            content = (human_action.chat_content or "我先听一下你们的想法，当前刀口我偏向高价值神职。").strip()
        else:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=speaker.id,
                    role=speaker.role,
                    day=self.day,
                    phase="wolf_chat",
                    visible_state=f"{self._build_wolf_visible_state()}\n私有信息：\n{self._player_private_context(speaker)}",
                    allowed_target_ids=candidate_ids,
                    prompt=(
                        "你在和狼人队友逐人协商刀人。"
                        "必须先接前面狼队已经说过的话，明确你是在认同、补充还是修正某个刀口。"
                        "不要空喊'先刀信息位'，要说清楚这个人为什么像神、像带队位、像明天会压住狼坑的人。"
                        "如果你改刀口，必须说明前面方案哪里不够优。"
                        "发言不要像总结报告，要像狼队内部真实商量。"
                    ),
                    persona_style=speaker.persona_style,
                    strategy_style=speaker.strategy_style,
                )
            )
            proposed_target_id = decision.target_id if decision.target_id in candidate_ids else (candidate_ids[0] if candidate_ids else None)
            content = (decision.content or "前面队友的刀口我能接，我补一条理由：先处理信息位。").strip()

        self.wolf_chat_records.append(
            WolfChatRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                proposed_target_id=proposed_target_id,
            )
        )
        self.speech_cursor += 1
        self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None

    def _build_wolf_visible_state(self) -> str:
        """给狼人看的夜聊局面。"""
        wolf_names = [f"{player.id}: {player.name}" for player in self.alive_wolves()]
        public = self.public_state_text()
        wolf_chat_lines = [
            f"{record.player_name}: {record.content}（建议刀 {record.proposed_target_id if record.proposed_target_id is not None else '无'}）"
            for record in self.wolf_chat_records[-8:]
        ]
        return "\n".join([public, "狼人队友：", *(wolf_names or ["无"]), "本轮狼队夜聊：", *(wolf_chat_lines or ["暂无夜聊记录"])])

    async def resolve_night(self, human_action: HumanNightAction | None = None) -> None:
        """执行夜晚结算。"""
        if self.phase not in (Phase.NIGHT, Phase.SETUP):
            return

        if not self.human_player.alive:
            human_action = None
        elif self.human_player.role not in (RoleName.SEER, RoleName.GUARD, RoleName.WITCH):
            human_action = HumanNightAction(action_type="skip", target_id=None)

        alive = self.alive_players()
        wolf_target_id = self.wolf_consensus_target_id
        guard_target_id = None
        seer_target_id = None
        seer_result = None
        witch_saved = False
        witch_poison_target_id = None
        witch_used_action = False

        non_wolves = [player for player in alive if player.role != RoleName.WEREWOLF]
        if wolf_target_id is None and non_wolves:
            wolf_target_id = non_wolves[0].id

        for player in alive:
            if player.role == RoleName.GUARD:
                candidate_ids = [candidate.id for candidate in alive if candidate.id != self.guard_last_target_id]
                if player.is_human:
                    self.pending_human_action = "night"
                    if human_action and human_action.action_type == "guard" and human_action.target_id in candidate_ids:
                        guard_target_id = human_action.target_id
                else:
                    decision = await self.runtime.decide(
                        AIContext(
                    player_id=player.id,
                    role=player.role,
                    day=self.day,
                    phase="night_action",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                    allowed_target_ids=candidate_ids,
                    prompt="请选择今晚要守护的玩家，不能连续守护同一人。",
                            persona_style=player.persona_style,
                            strategy_style=player.strategy_style,
                        )
                    )
                    if decision.target_id in candidate_ids:
                        guard_target_id = decision.target_id
                if guard_target_id is not None:
                    self.guard_last_target_id = guard_target_id

        for player in alive:
            if player.role == RoleName.SEER:
                candidate_ids = [candidate.id for candidate in alive if candidate.id != player.id]
                if player.is_human:
                    self.pending_human_action = "night"
                    if human_action and human_action.action_type == "inspect" and human_action.target_id in candidate_ids:
                        seer_target_id = human_action.target_id
                else:
                    decision = await self.runtime.decide(
                        AIContext(
                    player_id=player.id,
                    role=player.role,
                    day=self.day,
                    phase="night_action",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                    allowed_target_ids=candidate_ids,
                    prompt="请选择今晚要查验的玩家。你是预言家，必须认真利用自己的历史验人结果，不要把已经验过的铁信息当不存在。",
                    persona_style=player.persona_style,
                    strategy_style=player.strategy_style,
                )
                    )
                    if decision.target_id in candidate_ids:
                        seer_target_id = decision.target_id
                if seer_target_id is not None:
                    target = self.players[seer_target_id]
                    seer_result = "狼人" if target.camp == Camp.WEREWOLF else "好人"
                    player.private_note = f"你是预言家。历史查验结果：第{self.day}夜查验 {target.name} -> {seer_result}。"
                    if player.is_human:
                        self.last_human_seer_result = f"{target.name} 的查验结果是：{seer_result}"

        for player in alive:
            if player.role == RoleName.WITCH:
                allowed_ids = [candidate.id for candidate in alive if candidate.id != player.id]
                if player.is_human:
                    self.pending_human_action = "night"
                    if human_action and human_action.action_type == "save":
                        if (
                            not witch_used_action
                            and self.witch_state.save_available
                            and self._witch_can_save_target(player.id, wolf_target_id)
                            and human_action.target_id == wolf_target_id
                        ):
                            witch_saved = wolf_target_id is not None
                            witch_used_action = witch_saved
                    if human_action and human_action.action_type == "poison":
                        if (
                            not witch_used_action
                            and self.witch_state.poison_available
                            and human_action.target_id in allowed_ids
                        ):
                            witch_poison_target_id = human_action.target_id
                            witch_used_action = True
                else:
                    decision = await self.runtime.decide(
                        AIContext(
                    player_id=player.id,
                    role=player.role,
                    day=self.day,
                    phase="night_action",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                    allowed_target_ids=allowed_ids,
                    prompt=(
                        f"今晚狼人目标是 {wolf_target_id}。"
                        "你是女巫。请认真判断是否要开解药或毒药。"
                        "如果今晚刀中的是高价值信息位，通常优先考虑救；如果场上已经形成高概率狼坑，可考虑毒。"
                        "若想救人请选择该目标；若想毒人请选择一名高风险玩家；若不行动返回空 target_id。"
                    ),
                    persona_style=player.persona_style,
                    strategy_style=player.strategy_style,
                        )
                    )
                    if (
                        self.witch_state.save_available
                        and self._witch_can_save_target(player.id, wolf_target_id)
                        and decision.target_id == wolf_target_id
                        and not witch_used_action
                    ):
                        witch_saved = True
                        witch_used_action = True
                    elif (
                        self.witch_state.poison_available
                        and decision.target_id in allowed_ids
                        and not witch_used_action
                    ):
                        witch_poison_target_id = decision.target_id
                        witch_used_action = True

                if witch_saved:
                    self.witch_state.save_available = False
                    player.private_note = f"你是女巫。你在第{self.day}夜使用了解药，救下了 {self.players[wolf_target_id].name if wolf_target_id is not None else '未知目标'}。"
                if witch_poison_target_id is not None:
                    self.witch_state.poison_available = False
                    player.private_note = f"你是女巫。你在第{self.day}夜使用了毒药，毒杀了 {self.players[witch_poison_target_id].name}。"

        deaths: set[int] = set()
        if wolf_target_id is not None and wolf_target_id != guard_target_id and not witch_saved:
            deaths.add(wolf_target_id)
        if witch_poison_target_id is not None:
            deaths.add(witch_poison_target_id)
            if self.players[witch_poison_target_id].role == RoleName.HUNTER:
                self.hunter_poisoned = True

        for player_id in deaths:
            self.players[player_id].alive = False

        summary = NightSummary(
            day=self.day,
            wolf_target_id=wolf_target_id,
            guard_target_id=guard_target_id,
            seer_target_id=seer_target_id,
            seer_result=seer_result,
            witch_saved=witch_saved,
            witch_poison_target_id=witch_poison_target_id,
            deaths=sorted(deaths),
        )
        self.night_summaries.append(summary)

        self.pending_human_action = None
        self.wolf_consensus_target_id = None
        self.last_night_deaths = sorted(deaths)
        self._check_winner()
        if self.phase == Phase.GAME_OVER:
            return
        if self.day == 1 and self.sheriff_id is None:
            self.first_day_death_announcement_pending = True
            self.phase = Phase.SHERIFF_ELECTION
            self.pending_human_action = "sheriff_election" if self.human_player.alive else None
        else:
            self._announce_last_night_deaths_if_needed()
            self._prepare_day_speech_order()

    async def resolve_sheriff_election(self, human_action: SheriffAction | None = None) -> None:
        """执行警长完整流程。"""
        if self.phase == Phase.SHERIFF_ELECTION:
            await self._collect_sheriff_candidates(human_action)
            return
        if self.phase == Phase.SHERIFF_SPEECH:
            await self._advance_sheriff_speech(human_action.speech if human_action else "")
            return
        if self.phase == Phase.SHERIFF_PK_SPEECH:
            await self._advance_sheriff_pk_speech(human_action.speech if human_action else "")
            return
        if self.phase == Phase.SHERIFF_VOTE:
            await self._resolve_sheriff_vote(human_action)
            return
        if self.phase == Phase.SHERIFF_PK_VOTE:
            await self._resolve_sheriff_pk_vote(human_action)

    async def _collect_sheriff_candidates(self, human_action: SheriffAction | None) -> None:
        """收集上警名单。"""
        self.votes = [vote for vote in self.votes if vote.vote_round not in {"sheriff_vote", "sheriff_pk_vote"}]
        self.sheriff_candidate_ids = []
        alive = self.alive_players()

        for player in alive:
            if player.is_human:
                if human_action and human_action.run_for_sheriff:
                    self.sheriff_candidate_ids.append(player.id)
            else:
                should_run = player.role in {RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.WEREWOLF}
                if should_run and random.random() > 0.28:
                    self.sheriff_candidate_ids.append(player.id)

        if not self.sheriff_candidate_ids:
            self.events.append(GameEvent(phase="sheriff", message="本局无人上警，跳过警长竞选。"))
            self._announce_last_night_deaths_if_needed()
            self._prepare_day_speech_order()
            return

        self.events.append(
            GameEvent(
                phase="sheriff",
                message="上警名单：" + "、".join(self.players[player_id].name for player_id in self.sheriff_candidate_ids),
            )
        )
        self.speech_order = self.sheriff_candidate_ids[:]
        self.speech_cursor = 0
        self.phase = Phase.SHERIFF_SPEECH
        self.pending_human_action = "sheriff_speech" if self.current_speaker_id == self.human_player_id else None

    async def _advance_sheriff_speech(self, human_speech: str) -> None:
        """推进警上逐位发言。"""
        if self.phase != Phase.SHERIFF_SPEECH:
            return
        speaker_id = self.current_speaker_id
        if speaker_id is None:
            self.phase = Phase.SHERIFF_VOTE
            return

        speaker = self.players[speaker_id]
        if speaker.is_human:
            content = human_speech.strip() or "我愿意担任警长，带队发言和归票。"
        else:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=speaker.id,
                    role=speaker.role,
                    day=self.day,
                    phase="campaign_speech",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(speaker)}",
                    allowed_target_ids=[],
                    prompt=(
                        "你正在竞选警长。"
                        "这不是喊口号，必须像真人警上发言：要么给站边倾向，要么点一个你最在意的位置，要么说明你拿警徽后的第一处理顺序。"
                        "如果你像预言家视角，可以自然地体现警徽流意识，但别背模板。"
                        "不要只说'我带队''我控场''给我警徽'这种空话。"
                    ),
                    persona_style=speaker.persona_style,
                    strategy_style=speaker.strategy_style,
                )
            )
            content = decision.content.strip() or "我上警是为了整理逻辑，带大家找狼。"

        speaker.last_speech = content
        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="campaign",
            )
        )
        self.speech_cursor += 1

        if self.current_speaker_id is None:
            self.events.append(GameEvent(phase="sheriff", message="警上发言结束，开始投票选警长。"))
            self.phase = Phase.SHERIFF_VOTE
            can_human_vote = self.human_player.alive and self.human_player_id not in self.sheriff_candidate_ids
            self.pending_human_action = "sheriff_vote" if can_human_vote else None
            return

        self.pending_human_action = "sheriff_speech" if self.current_speaker_id == self.human_player_id else None

    async def _advance_sheriff_pk_speech(self, human_speech: str) -> None:
        """推进警长 PK 发言。"""
        if self.phase != Phase.SHERIFF_PK_SPEECH:
            return
        speaker_id = self.current_speaker_id
        if speaker_id is None:
            self.phase = Phase.SHERIFF_PK_VOTE
            can_human_vote = self.human_player.alive and self.human_player_id not in self.sheriff_pk_candidate_ids
            self.pending_human_action = "sheriff_vote" if can_human_vote else None
            return

        speaker = self.players[speaker_id]
        if speaker.is_human:
            content = human_speech.strip() or "我补充一下警徽流和站边逻辑，请大家把票投给我。"
        else:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=speaker.id,
                    role=speaker.role,
                    day=self.day,
                    phase="pk_campaign_speech",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(speaker)}",
                    allowed_target_ids=[],
                    prompt=(
                        "你正在进行警长 PK 发言。"
                        "必须正面回应别人为什么不该把警徽给对手，而不是重复自己上一轮的话。"
                        "优先抓对手一个具体矛盾、改口点或警徽流漏洞狠狠干。"
                    ),
                    persona_style=speaker.persona_style,
                    strategy_style=speaker.strategy_style,
                )
            )
            content = decision.content.strip() or "请大家回看我前面的逻辑和站边，我更适合拿警徽。"

        speaker.last_speech = content
        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="pk_campaign",
            )
        )
        self.speech_cursor += 1

        if self.current_speaker_id is None:
            self.events.append(GameEvent(phase="sheriff", message="警长 PK 发言结束，进入重新投票。"))
            self.phase = Phase.SHERIFF_PK_VOTE
            can_human_vote = self.human_player.alive and self.human_player_id not in self.sheriff_pk_candidate_ids
            self.pending_human_action = "sheriff_vote" if can_human_vote else None
            return

        self.pending_human_action = "sheriff_pk_speech" if self.current_speaker_id == self.human_player_id else None

    async def _resolve_sheriff_vote(self, human_action: SheriffAction | None) -> None:
        """结算警长投票。"""
        alive = self.alive_players()
        non_candidates = [player for player in alive if player.id not in self.sheriff_candidate_ids]
        if not non_candidates:
            self.sheriff_id = self.sheriff_candidate_ids[0]
        else:
            tally: dict[int, float] = {candidate_id: 0.0 for candidate_id in self.sheriff_candidate_ids}
            for player in non_candidates:
                if player.is_human:
                    target_id = human_action.vote_target_id if human_action and human_action.vote_target_id in self.sheriff_candidate_ids else self.sheriff_candidate_ids[0]
                else:
                    decision = await self.runtime.decide(
                        AIContext(
                            player_id=player.id,
                            role=player.role,
                            day=self.day,
                            phase="sheriff_vote",
                            visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                            allowed_target_ids=self.sheriff_candidate_ids,
                            prompt="请在警上玩家中选择一位你认为更适合做警长的人。",
                            persona_style=player.persona_style,
                            strategy_style=player.strategy_style,
                        )
                    )
                    target_id = decision.target_id if decision.target_id in self.sheriff_candidate_ids else self.sheriff_candidate_ids[0]

                tally[target_id] += 1.0
                self.votes.append(
                    VoteRecord(
                        day=self.day,
                        voter_id=player.id,
                        voter_name=player.name,
                        target_id=target_id,
                        target_name=self.players[target_id].name,
                        vote_type="sheriff",
                        vote_round="sheriff_vote",
                    )
                )

            self.sheriff_vote_tally = tally
            max_vote = max(tally.values())
            top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
            if len(top_ids) > 1:
                self.sheriff_pk_candidate_ids = top_ids
                self.speech_order = top_ids[:]
                self.speech_cursor = 0
                self.phase = Phase.SHERIFF_PK_SPEECH
                self.pending_human_action = "sheriff_pk_speech" if self.current_speaker_id == self.human_player_id else None
                self.events.append(GameEvent(phase="sheriff", message="警长投票平票，进入 PK 发言环节。"))
                return
            self.sheriff_id = top_ids[0]

        await self._finalize_sheriff_after_election()

    async def _resolve_sheriff_pk_vote(self, human_action: SheriffAction | None) -> None:
        """结算警长 PK 投票。"""
        alive = self.alive_players()
        non_candidates = [player for player in alive if player.id not in self.sheriff_pk_candidate_ids]
        tally: dict[int, float] = {candidate_id: 0.0 for candidate_id in self.sheriff_pk_candidate_ids}

        for player in non_candidates:
            if player.is_human:
                target_id = human_action.vote_target_id if human_action and human_action.vote_target_id in self.sheriff_pk_candidate_ids else self.sheriff_pk_candidate_ids[0]
            else:
                decision = await self.runtime.decide(
                    AIContext(
                        player_id=player.id,
                        role=player.role,
                        day=self.day,
                        phase="sheriff_pk_vote",
                        visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                        allowed_target_ids=self.sheriff_pk_candidate_ids,
                        prompt="警长 PK 投票，请在候选人中选一人。",
                        persona_style=player.persona_style,
                        strategy_style=player.strategy_style,
                    )
                )
                target_id = decision.target_id if decision.target_id in self.sheriff_pk_candidate_ids else self.sheriff_pk_candidate_ids[0]
            tally[target_id] += 1.0
            self.votes.append(
                VoteRecord(
                    day=self.day,
                    voter_id=player.id,
                    voter_name=player.name,
                    target_id=target_id,
                    target_name=self.players[target_id].name,
                    vote_type="sheriff",
                    vote_round="sheriff_pk_vote",
                )
            )

        max_vote = max(tally.values()) if tally else 0.0
        top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
        if len(top_ids) > 1:
            self.events.append(GameEvent(phase="sheriff", message="警长 PK 再次平票，本局警徽流失。"))
            self.sheriff_id = None
            self.sheriff_candidate_ids = []
            self.sheriff_pk_candidate_ids = []
            self.speech_order = []
            self.speech_cursor = 0
            self.pending_human_action = None
            self._prepare_day_speech_order()
            return

        self.sheriff_id = top_ids[0]
        await self._finalize_sheriff_after_election()

    async def _finalize_sheriff_after_election(self) -> None:
        """警长竞选收口。"""
        self.players[self.sheriff_id].is_sheriff = True
        self.events.append(GameEvent(phase="sheriff", message=f"{self.players[self.sheriff_id].name} 当选警长。"))
        self.sheriff_candidate_ids = []
        self.sheriff_pk_candidate_ids = []
        self.speech_order = []
        self.speech_cursor = 0
        self.pending_human_action = None
        self._announce_last_night_deaths_if_needed()
        if self.sheriff_id == self.human_player_id and self.human_player.alive:
            self.phase = Phase.DAY_SPEECH
            self.pending_human_action = "choose_speech_order"
            return

        direction = "right"
        sheriff = self.players[self.sheriff_id]
        if sheriff.alive:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=sheriff.id,
                    role=sheriff.role,
                    day=self.day,
                    phase="choose_speech_order",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(sheriff)}",
                    allowed_target_ids=[],
                    prompt="你是警长，请在 left 和 right 中选择一个白天发言方向，并在 content 中只输出 left 或 right。",
                    persona_style=sheriff.persona_style,
                    strategy_style=sheriff.strategy_style,
                )
            )
            direction = decision.content.strip().lower()
            if direction not in {"left", "right"}:
                direction = "right"

        self._prepare_day_speech_order(direction)

    def _prepare_day_speech_order(self, direction: str | None = None) -> None:
        """准备逐位发言顺序。"""
        alive_ids = [player.id for player in self.alive_players()]
        if not alive_ids:
            return

        if self.sheriff_id is not None and self.players[self.sheriff_id].alive:
            sheriff_index = alive_ids.index(self.sheriff_id)
            if direction is None:
                direction = "right"
            if direction == "left":
                order = alive_ids[sheriff_index::-1] + alive_ids[:sheriff_index:-1]
            else:
                order = alive_ids[sheriff_index + 1 :] + alive_ids[: sheriff_index + 1]
        else:
            order = alive_ids[:]

        self.speech_order = order
        self.speech_cursor = 0
        self.phase = Phase.DAY_SPEECH
        self.pending_human_action = "day_speech" if self.current_speaker_id == self.human_player_id else None
        self._arm_auto_step_delay(1.2)

    async def choose_speech_order(self, direction: str | None) -> None:
        """允许警长选择发言方向。"""
        if self.phase != Phase.DAY_SPEECH or self.speech_order or self.sheriff_id != self.human_player_id:
            return
        if direction not in {"left", "right"}:
            direction = "right"
        self._prepare_day_speech_order(direction)

    async def resolve_day_speeches(self, human_speech: str) -> None:
        """执行白天逐位发言。"""
        if self.phase != Phase.DAY_SPEECH:
            return

        speaker_id = self.current_speaker_id
        if speaker_id is None:
            self.phase = Phase.DAY_VOTE
            return

        speaker = self.players[speaker_id]
        if speaker.is_human:
            content = human_speech.strip() or "我先听听大家怎么盘。"
        else:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=speaker.id,
                    role=speaker.role,
                    day=self.day,
                    phase="day_speech",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(speaker)}",
                    allowed_target_ids=[],
                    prompt=(
                        "请进行白天发言。"
                        "默认只接前面最 relevant 的 1 到 2 个点继续往下打，不要每轮从零总结全场。"
                        "必须点透一个核心矛盾：比如某人的改口、站边不闭环、票型补刀、先说满后回收。"
                        "不要只说'像狼''不干净''带节奏'，后面必须跟具体事实。"
                    ),
                    persona_style=speaker.persona_style,
                    strategy_style=speaker.strategy_style,
                )
            )
            content = decision.content.strip() or "我先保留一点身份信息，重点看前后位逻辑。"

        speaker.last_speech = content
        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="day",
            )
        )
        self.speech_cursor += 1

        if self.current_speaker_id is None:
            self.events.append(GameEvent(phase="speech", message=f"第 {self.day} 天白天发言结束。"))
            self.phase = Phase.DAY_VOTE
            self.pending_human_action = "day_vote"
            return

        self.pending_human_action = "day_speech" if self.current_speaker_id == self.human_player_id else None

    async def resolve_votes(self, human_target_id: int) -> None:
        """执行白天投票。"""
        if self.phase == Phase.SHERIFF_PK_SPEECH and self.exile_pk_candidate_ids:
            await self._advance_exile_pk_speech("过。" if self.current_speaker_id != self.human_player_id else "")
            return
        if self.phase == Phase.SHERIFF_PK_VOTE and self.exile_pk_candidate_ids:
            await self._resolve_exile_pk_vote(human_target_id)
            return
        if self.phase != Phase.DAY_VOTE:
            return

        self.votes = [
            vote
            for vote in self.votes
            if not (vote.vote_type == "exile" and vote.day == self.day)
        ]
        alive = self.alive_players()
        alive_ids = [player.id for player in alive]
        tally: dict[int, float] = {}

        for player in alive:
            if not player.can_vote:
                continue
            candidates = [candidate_id for candidate_id in alive_ids if candidate_id != player.id]
            if not candidates:
                continue
            if player.is_human:
                target_id = human_target_id if human_target_id in candidates else candidates[0]
            else:
                decision = await self.runtime.decide(
                    AIContext(
                        player_id=player.id,
                        role=player.role,
                        day=self.day,
                        phase="day_vote",
                        visible_state=self.public_state_text(),
                        allowed_target_ids=candidates,
                        prompt="请选择今天放逐投票的目标。",
                        persona_style=player.persona_style,
                        strategy_style=player.strategy_style,
                    )
                )
                target_id = decision.target_id if decision.target_id in candidates else candidates[0]

            vote_weight = 1.5 if player.is_sheriff else 1.0
            tally[target_id] = tally.get(target_id, 0.0) + vote_weight
            self.votes.append(
                VoteRecord(
                    day=self.day,
                    voter_id=player.id,
                    voter_name=player.name,
                    target_id=target_id,
                    target_name=self.players[target_id].name,
                    vote_type="exile",
                    vote_round=f"day_{self.day}_exile",
                )
            )

        if not tally:
            self.events.append(GameEvent(phase="vote", message="本轮无人能够投票，直接进入黑夜。"))
            self._advance_to_next_day()
            return

        max_vote = max(tally.values())
        top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
        if len(top_ids) > 1:
            self.exile_pk_candidate_ids = top_ids
            self.speech_order = top_ids[:]
            self.speech_cursor = 0
            self.phase = Phase.SHERIFF_PK_SPEECH
            self.pending_human_action = "sheriff_pk_speech" if self.current_speaker_id == self.human_player_id else None
            self.events.append(
                GameEvent(
                    phase="vote",
                    message="本轮放逐投票平票，进入 PK 辩护发言与重新投票。",
                )
            )
            return

        await self._execute_exile_outcome(top_ids[0])

    async def _advance_exile_pk_speech(self, human_speech: str) -> None:
        """推进放逐 PK 辩护发言。"""
        speaker_id = self.current_speaker_id
        if speaker_id is None:
            self.phase = Phase.SHERIFF_PK_VOTE
            self.pending_human_action = "day_vote" if self.human_player.alive and self.human_player.can_vote else None
            return

        speaker = self.players[speaker_id]
        if speaker.is_human:
            content = human_speech.strip() or "我补充一下，刚才那轮不该把我直接打死。"
        else:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=speaker.id,
                    role=speaker.role,
                    day=self.day,
                    phase="pk_campaign_speech",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(speaker)}",
                    allowed_target_ids=[],
                    prompt=(
                        "你正在进行白天放逐 PK 辩护发言。"
                        "必须解释为什么自己不该被今天推出去，并点对手一个最硬的逻辑漏洞。"
                        "不要复读上一轮，发言要更像被顶到刀口上的真人反击。"
                    ),
                    persona_style=speaker.persona_style,
                    strategy_style=speaker.strategy_style,
                )
            )
            content = decision.content.strip() or "我这轮被顶到 PK 不是因为我像狼，而是有人在借势做公共坑。"

        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="pk_campaign",
            )
        )
        self.speech_cursor += 1
        self.pending_human_action = "day_speech" if self.current_speaker_id == self.human_player_id else None

    async def _resolve_exile_pk_vote(self, human_target_id: int) -> None:
        """结算放逐 PK 重新投票。"""
        self.votes = [
            vote
            for vote in self.votes
            if not (vote.vote_type == "exile" and vote.vote_round == f"day_{self.day}_pk_exile")
        ]
        alive = self.alive_players()
        tally: dict[int, float] = {candidate_id: 0.0 for candidate_id in self.exile_pk_candidate_ids}

        for player in alive:
            if not player.can_vote or player.id in self.exile_pk_candidate_ids:
                continue
            if player.is_human:
                target_id = human_target_id if human_target_id in self.exile_pk_candidate_ids else self.exile_pk_candidate_ids[0]
            else:
                decision = await self.runtime.decide(
                    AIContext(
                        player_id=player.id,
                        role=player.role,
                        day=self.day,
                        phase="day_vote",
                        visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                        allowed_target_ids=self.exile_pk_candidate_ids,
                        prompt="现在是放逐 PK 重新投票，只能在两名 PK 玩家里选一人出局。",
                        persona_style=player.persona_style,
                        strategy_style=player.strategy_style,
                    )
                )
                target_id = decision.target_id if decision.target_id in self.exile_pk_candidate_ids else self.exile_pk_candidate_ids[0]

            vote_weight = 1.5 if player.is_sheriff else 1.0
            tally[target_id] = tally.get(target_id, 0.0) + vote_weight
            self.votes.append(
                VoteRecord(
                    day=self.day,
                    voter_id=player.id,
                    voter_name=player.name,
                    target_id=target_id,
                    target_name=self.players[target_id].name,
                    vote_type="exile",
                    vote_round=f"day_{self.day}_pk_exile",
                )
            )

        max_vote = max(tally.values()) if tally else 0.0
        top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
        self.exile_pk_candidate_ids = []
        self.speech_order = []
        self.speech_cursor = 0

        if len(top_ids) != 1:
            self.events.append(GameEvent(phase="vote", message="放逐 PK 再次平票，无人出局，直接进入黑夜。"))
            self._advance_to_next_day()
            return

        await self._execute_exile_outcome(top_ids[0])

    async def _execute_exile_outcome(self, out_id: int) -> None:
        """执行放逐出局后的统一结算。"""
        out_player = self.players[out_id]
        self.current_exile_target_id = out_id

        if out_player.role == RoleName.IDIOT and not out_player.idiot_revealed:
            out_player.idiot_revealed = True
            out_player.can_vote = False
            self.events.append(GameEvent(phase="vote", message=f"{out_player.name} 翻牌白痴，免于出局，但失去投票权。"))
            self._advance_to_next_day()
            return

        out_player.alive = False
        self.events.append(GameEvent(phase="vote", message=f"{out_player.name} 被公投出局。"))
        if self.day == 1:
            self.last_words_queue = [out_id]
            self.phase = Phase.LAST_WORDS
            self.current_exile_target_id = self.last_words_queue.pop(0)
            self.pending_human_action = "last_words" if self.current_exile_target_id == self.human_player_id else None
            return

        if out_player.role == RoleName.HUNTER:
            await self._resolve_hunter_shot(out_player.id)
            if self.phase == Phase.GAME_OVER:
                return

        if out_player.is_sheriff:
            self.phase = Phase.BADGE_TRANSFER
            self.current_exile_target_id = out_id
            self.pending_human_action = "badge_transfer" if out_player.is_human else None
            return

        self._check_winner()
        if self.phase != Phase.GAME_OVER:
            self._advance_to_next_day()

    async def resolve_last_words(self, content: str) -> None:
        """结算遗言。"""
        if self.phase != Phase.LAST_WORDS or self.current_exile_target_id is None:
            return

        player = self.players[self.current_exile_target_id]
        if any(
            record.day == self.day
            and record.player_id == player.id
            and record.speech_type == "last_words"
            for record in self.speeches
        ):
            return
        if player.is_human:
            speech = content.strip() or "我遗言结束，祝大家好运。"
        else:
            decision = await self.runtime.decide(
                AIContext(
                    player_id=player.id,
                    role=player.role,
                    day=self.day,
                    phase="last_words",
                    visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(player)}",
                    allowed_target_ids=[],
                    prompt=(
                        "你已出局，请发表简短遗言。"
                        "遗言只留最关键的信息：你最认的一张、最想点的一张、或你为什么会被推出去。"
                        "不要像赛后总结，不要长篇大论。"
                    ),
                    persona_style=player.persona_style,
                    strategy_style=player.strategy_style,
                )
            )
            speech = decision.content.strip() or "我遗言结束，大家自己分辨站边。"

        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=player.id,
                player_name=player.name,
                content=speech,
                speech_type="last_words",
            )
        )

        if player.role == RoleName.HUNTER:
            await self._resolve_hunter_shot(player.id)
            if self.phase == Phase.GAME_OVER:
                return

        if self.last_words_queue:
            self.current_exile_target_id = self.last_words_queue.pop(0)
            self.pending_human_action = "last_words" if self.current_exile_target_id == self.human_player_id else None
            return

        if player.is_sheriff:
            self.phase = Phase.BADGE_TRANSFER
            self.pending_human_action = "badge_transfer" if player.is_human else None
            return

        self._check_winner()
        if self.phase != Phase.GAME_OVER:
            self._advance_to_next_day()

    async def resolve_badge_transfer(self, action: SheriffAction | None = None) -> None:
        """结算警徽移交。"""
        if self.phase != Phase.BADGE_TRANSFER or self.current_exile_target_id is None:
            return

        dead_sheriff = self.players[self.current_exile_target_id]
        dead_sheriff.is_sheriff = False
        self.sheriff_id = None
        alive_good_ids = [
            player.id for player in self.alive_players() if player.camp == Camp.VILLAGER and player.id != dead_sheriff.id
        ]

        if dead_sheriff.is_human:
            if action and action.tear_badge:
                self.events.append(GameEvent(phase="badge", message=f"{dead_sheriff.name} 选择撕毁警徽。"))
            else:
                target_id = action.badge_target_id if action and action.badge_target_id in alive_good_ids else None
                if target_id is not None:
                    self.sheriff_id = target_id
                    self.players[target_id].is_sheriff = True
                    self.events.append(GameEvent(phase="badge", message=f"{dead_sheriff.name} 将警徽移交给 {self.players[target_id].name}。"))
                else:
                    self.events.append(GameEvent(phase="badge", message=f"{dead_sheriff.name} 未移交警徽，警徽作废。"))
        else:
            if alive_good_ids:
                decision = await self.runtime.decide(
                    AIContext(
                        player_id=dead_sheriff.id,
                        role=dead_sheriff.role,
                        day=self.day,
                        phase="badge_transfer",
                        visible_state=f"{self.public_state_text()}\n私有信息：\n{self._player_private_context(dead_sheriff)}",
                        allowed_target_ids=alive_good_ids,
                        prompt="你是死亡警长。请在存活好人里选择一名你最认的人移交警徽；若都不认可，可返回空目标视为撕徽。",
                        persona_style=dead_sheriff.persona_style,
                        strategy_style=dead_sheriff.strategy_style,
                    )
                )
                target_id = decision.target_id if decision.target_id in alive_good_ids else None
            else:
                target_id = None
            if target_id is not None:
                self.sheriff_id = target_id
                self.players[target_id].is_sheriff = True
                self.events.append(GameEvent(phase="badge", message=f"{dead_sheriff.name} 将警徽移交给 {self.players[target_id].name}。"))
            else:
                self.events.append(GameEvent(phase="badge", message=f"{dead_sheriff.name} 撕毁了警徽。"))

        self._check_winner()
        if self.phase != Phase.GAME_OVER:
            self._advance_to_next_day()

    async def resolve_self_destruct(self) -> None:
        """狼人白天自爆。"""
        if not self._can_human_self_destruct():
            return

        player = self.human_player
        player.alive = False
        if self.phase == Phase.SHERIFF_SPEECH:
            self.sheriff_id = None
            self.sheriff_candidate_ids = []
            self.sheriff_pk_candidate_ids = []
            self.events.append(GameEvent(phase="explode", message=f"{player.name} 在警上发言阶段自爆，警徽流失，直接进入黑夜。"))
        else:
            self.events.append(GameEvent(phase="explode", message=f"{player.name} 选择自爆，白天流程立即结束，直接进入黑夜。"))
        if player.is_sheriff:
            self.current_exile_target_id = player.id
            self.phase = Phase.BADGE_TRANSFER
            self.pending_human_action = "badge_transfer"
            return
        self._check_winner()
        if self.phase != Phase.GAME_OVER:
            self._advance_to_next_day()

    def _advance_to_next_day(self) -> None:
        """推进到下一晚。"""
        self.current_exile_target_id = None
        self.last_words_queue = []
        self.pending_human_action = None
        self.speech_order = []
        self.speech_cursor = 0
        self.last_night_deaths = []
        self.day += 1
        self.wolf_chat_records = []
        self.phase = Phase.WOLF_CHAT
        self._prepare_wolf_chat_order()

    def _witch_can_save_target(self, witch_id: int, wolf_target_id: int | None) -> bool:
        """竞技标准预女猎白：女巫全程不可自救。"""
        if wolf_target_id is None:
            return False
        return wolf_target_id != witch_id

    def _check_winner(self) -> None:
        """检查胜负。"""
        wolves = len(self.alive_wolves())
        alive_goods = self.alive_villagers()
        gods = [player for player in alive_goods if player.role != RoleName.VILLAGER]
        villagers = [player for player in alive_goods if player.role == RoleName.VILLAGER]

        if wolves == 0:
            self.phase = Phase.GAME_OVER
            self.winner = "好人阵营"
            self.events.append(GameEvent(phase="result", message="游戏结束：好人阵营获胜。"))
        elif not gods or not villagers:
            self.phase = Phase.GAME_OVER
            self.winner = "狼人阵营"
            self.events.append(GameEvent(phase="result", message="游戏结束：狼人阵营获胜。"))

    async def _resolve_hunter_shot(self, hunter_id: int) -> None:
        """猎人死亡后开枪。"""
        hunter = self.players[hunter_id]
        if self.hunter_poisoned:
            self.events.append(GameEvent(phase="hunter", message=f"{hunter.name} 被女巫毒死，无法开枪。"))
            return
        candidates = [player.id for player in self.alive_players() if player.id != hunter.id]
        if not candidates:
            return
        target_id = candidates[0]
        target = self.players[target_id]
        target.alive = False
        self.events.append(GameEvent(phase="hunter", message=f"{hunter.name} 死亡开枪，带走了 {target.name}。"))
        self._check_winner()


class GameManager:
    """管理多局游戏。"""

    def __init__(self) -> None:
        self._games: dict[str, WerwolfGame] = {}

    def create_game(self, player_count: int) -> WerwolfGame:
        """创建并保存新对局。"""
        game = WerwolfGame.create(player_count)
        self._games[game.game_id] = game
        return game

    def get_game(self, game_id: str) -> WerwolfGame:
        """获取对局。"""
        return self._games[game_id]
