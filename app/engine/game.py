"""狼人杀规则引擎。"""
from __future__ import annotations

import random
import re
import time
import asyncio
from contextlib import contextmanager
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from collections import Counter

from app.agents.action_space import ActionOption, ActionSpace
from app.agents.action_space import ActionKind
from app.agents.pipeline import DecisionPipeline, DecisionRequest
from app.agents.prompts import PERSONA_POOL
from app.agents.runtime import AIContext, OpenAIAgentRuntime
from app.core.config import settings
from app.engine.models import (
    Camp,
    AgentMemory,
    AgentVisibleContext,
    CampSharedMemory,
    DeathFact,
    DecisionAudit,
    GameEvent,
    GameSnapshot,
    HumanNightAction,
    IdiotRevealFact,
    LegalAction,
    NightSummary,
    Phase,
    PlayerAgentState,
    PlayerState,
    PrivateObservation,
    PublicClaimEvidence,
    PublicSpeechEvidence,
    ROLE_CAMP,
    ROLE_CONFIGS,
    RuleProfile,
    RoleName,
    SeatRef,
    SeerInspectionFact,
    SheriffAction,
    SpeechRecord,
    TableMessage,
    VisibleTimelineItem,
    VoteEvidence,
    VoteRecord,
    WitchActionFact,
    WitchNightInfo,
    WolfChatRecord,
    WolfNightPlan,
)


def build_default_names(count: int) -> list[str]:
    """生成默认玩家名。"""
    return [f"玩家{i + 1}" for i in range(count)]


PERSONA_STYLES = list(PERSONA_POOL)

ROLE_STRATEGY_STYLES = {
    RoleName.WEREWOLF: ["控场悍跳流", "深水倒钩流", "冲票做局流"],
    RoleName.SEER: ["强预带队流", "稳预控场流", "藏锋反打流"],
    RoleName.WITCH: ["轮次收益流", "藏毒等待流", "强博弈反制流"],
    RoleName.HUNTER: ["隐忍带枪流", "压场威慑流", "残局定胜流"],
    RoleName.GUARD: ["稳守信息流", "轮次博弈流", "藏身份守护流"],
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
    night_id: int = 1
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
    wolf_night_plan: WolfNightPlan | None = None
    wolf_chat_prepared_night_id: int | None = None
    wolf_chat_round: int = 1
    wolf_chat_turn_index: int = 0
    event_seq: int = 0
    message_seq: int = 0
    message_log: list[TableMessage] = field(default_factory=list)
    agent_memories: dict[int, AgentMemory] = field(default_factory=dict)
    agent_states: dict[int, PlayerAgentState] = field(default_factory=dict)
    camp_memories: dict[Camp, CampSharedMemory] = field(default_factory=dict)
    decision_audits: list[DecisionAudit] = field(default_factory=list)
    seer_inspection_facts: list[SeerInspectionFact] = field(default_factory=list)
    witch_action_facts: list[WitchActionFact] = field(default_factory=list)
    death_facts: list[DeathFact] = field(default_factory=list)
    idiot_reveal_facts: list[IdiotRevealFact] = field(default_factory=list)
    rule_profile: RuleProfile = field(default_factory=RuleProfile)
    sheriff_id: int | None = None
    sheriff_candidate_ids: list[int] = field(default_factory=list)
    sheriff_vote_tally: dict[int, float] = field(default_factory=dict)
    sheriff_pk_candidate_ids: list[int] = field(default_factory=list)
    exile_pk_candidate_ids: list[int] = field(default_factory=list)
    speech_order: list[int] = field(default_factory=list)
    speech_cursor: int = 0
    last_words_queue: list[int] = field(default_factory=list)
    death_resolution_player_ids: list[int] = field(default_factory=list)
    current_exile_target_id: int | None = None
    death_resolution_source: str = ""
    pending_hunter_id: int | None = None
    last_night_deaths: list[int] = field(default_factory=list)
    first_day_death_announcement_pending: bool = False
    hunter_poisoned: bool = False
    timer_label: str = ""
    time_limit_seconds: int = 0
    deadline_ts: float | None = None
    timer_signature: str = ""
    auto_step_ready_ts: float = 0.0
    runtime: OpenAIAgentRuntime = field(default_factory=OpenAIAgentRuntime)
    decision_pipeline: DecisionPipeline | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def initialize_agent_state(self) -> None:
        """初始化每名玩家的 Agent 状态。"""
        self.agent_memories = {player.id: AgentMemory(player_id=player.id) for player in self.players}
        self.agent_states = {
            player.id: PlayerAgentState(
                player_id=player.id,
                seat_no=player.id + 1,
                role=player.role,
                camp=player.camp,
                persona_style=player.persona_style,
                strategy_style=player.strategy_style,
                private_summary=f"我是{player.id + 1}号，底牌{player.role.value}。",
            )
            for player in self.players
        }
        self.camp_memories = {Camp.WEREWOLF: CampSharedMemory(camp=Camp.WEREWOLF)}
        self.decision_pipeline = DecisionPipeline(self.runtime)

    @classmethod
    def create(cls, player_count: int) -> "WerwolfGame":
        """创建新对局。"""
        if player_count != 12:
            raise ValueError("当前默认主规则仅支持 12 人局。")

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
        game.rule_profile = RuleProfile(player_count=player_count, role_pool=roles[:])
        game.initialize_agent_state()
        for player in players:
            game._remember_private(
                player,
                "初始化底牌：" + f"{player.id + 1}号，身份{player.role.value}，阵营{'狼人' if player.camp == Camp.WEREWOLF else '好人'}。",
                {"role": player.role.value, "camp": player.camp.value, "seat_no": player.id + 1},
            )
        game._add_event("setup", "游戏创建完成，已随机分配角色，当前进入首夜狼人协商阶段。")
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
            alive_desc.append(f"player_id={player.id}, seat_no={player.id + 1}: {player.name} - {status}{sheriff}")

        last_speeches = [f"{record.player_name}: {record.content}" for record in self.speeches[-10:]]
        recent_votes = [
            f"第{vote.day}天{vote.voter_name} -> {vote.target_name}（{vote.vote_type}）"
            for vote in self.votes[-10:]
        ]
        recent_events = [
            f"{event.phase}: {event.message}"
            for event in self._visible_events_for_player(None)[-8:]
        ]
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

    def _player_private_context(self, player: PlayerState, *, human_readable: bool = False) -> str:
        """构建玩家私有信息。"""
        if human_readable:
            return self._human_private_context(player)
        notes: list[str] = [
            f"你的座位：player_id={player.id}，{player.id + 1}号",
            f"你的姓名：{player.name}",
            f"你的身份：{player.role.value}",
            f"你的阵营：{'狼人阵营' if player.camp == Camp.WEREWOLF else '好人阵营'}",
            f"你的存活状态：{'存活' if player.alive else '死亡'}",
            f"你的投票权：{'有' if player.can_vote else '无'}",
        ]
        if player.camp == Camp.WEREWOLF:
            teammates = [
                f"player_id={teammate.id}（{teammate.id + 1}号 {teammate.name}，狼人）"
                for teammate in self.players
                if teammate.camp == Camp.WEREWOLF and teammate.id != player.id
            ]
            legal_targets = [
                f"player_id={target.id}（{target.id + 1}号 {target.name}）"
                for target in self.alive_players()
                if target.camp != Camp.WEREWOLF
            ]
            notes.extend(
                [
                    "狼人队友：" + ("、".join(teammates) if teammates else "无"),
                    "狼人夜间合法刀口：" + ("、".join(legal_targets) if legal_targets else "无"),
                    "硬规则：狼人不能刀自己，也不能刀狼人同伴。讨论刀口时不要把狼人队友当作候选。",
                ]
            )
        if player.role == RoleName.WITCH and self.phase == Phase.NIGHT:
            notes.append(self._witch_private_night_info(player))
        if player.role == RoleName.HUNTER:
            if self.hunter_poisoned and not player.alive:
                notes.append("猎人技能：本次出局来自女巫毒杀，不能开枪。")
            elif self.phase == Phase.HUNTER_SHOT and self.pending_hunter_id == player.id:
                notes.append("猎人技能：当前可以开枪带走一名存活玩家。")
            else:
                notes.append("猎人技能：若被狼刀或白天放逐出局，可以开枪；被女巫毒杀不能开枪。")
        if player.role == RoleName.IDIOT:
            notes.append(
                "白痴技能："
                + ("已翻牌，免于白天放逐出局，但已失去投票权。" if player.idiot_revealed else "尚未翻牌；被白天公投放逐时可翻牌免死。")
            )
        if player.private_note:
            notes.append(player.private_note)
        if player.is_human and self.last_human_seer_result:
            notes.append(self.last_human_seer_result)
        memory = self.agent_memories.get(player.id)
        if memory and memory.private_observations:
            recent = memory.private_observations[-6:]
            notes.append("你的历史私有记忆：")
            notes.extend(f"- 第{item.day}天/{item.phase}: {item.content}" for item in recent)
        return "\n".join(notes) if notes else "暂无私有信息。"

    def _human_private_context(self, player: PlayerState) -> str:
        """给前端真人看的私有信息，隐藏内部 player_id 细节。"""
        notes: list[str] = [
            f"你是 {player.id + 1} 号位，名字：{player.name}",
            f"身份：{player.role.value}",
            f"阵营：{'狼人阵营' if player.camp == Camp.WEREWOLF else '好人阵营'}",
            f"状态：{'存活' if player.alive else '死亡'}，投票权：{'有' if player.can_vote else '无'}",
        ]
        if player.camp == Camp.WEREWOLF:
            teammates = [
                f"{teammate.id + 1}号 {teammate.name}"
                for teammate in self.players
                if teammate.camp == Camp.WEREWOLF and teammate.id != player.id
            ]
            legal_targets = [
                f"{target.id + 1}号 {target.name}"
                for target in self.alive_players()
                if target.camp != Camp.WEREWOLF
            ]
            notes.extend(
                [
                    "狼队友：" + ("、".join(teammates) if teammates else "无"),
                    "本夜可刀目标：" + ("、".join(legal_targets) if legal_targets else "无"),
                    "规则提醒：不能刀自己，也不能刀狼队友。",
                ]
            )
        if player.role == RoleName.WITCH and self.phase == Phase.NIGHT:
            notes.append(self._witch_private_night_info(player, human_readable=True))
        if player.role == RoleName.HUNTER:
            if self.hunter_poisoned and not player.alive:
                notes.append("猎人技能：本次出局来自女巫毒杀，不能开枪。")
            elif self.phase == Phase.HUNTER_SHOT and self.pending_hunter_id == player.id:
                notes.append("猎人技能：当前可以开枪带走一名存活玩家。")
            else:
                notes.append("猎人技能：被狼刀或白天放逐出局可以开枪；被女巫毒杀不能开枪。")
        if player.role == RoleName.IDIOT:
            notes.append(
                "白痴技能："
                + ("已翻牌，免于白天放逐出局，但已失去投票权。" if player.idiot_revealed else "尚未翻牌；被白天公投放逐时可翻牌免死。")
            )
        if player.private_note:
            notes.append(player.private_note)
        if player.is_human and self.last_human_seer_result:
            notes.append(self.last_human_seer_result)
        memory = self.agent_memories.get(player.id)
        if memory and memory.private_observations:
            recent = memory.private_observations[-4:]
            notes.append("你的私有记忆：")
            notes.extend(f"- 第{item.day}天/{item.phase}: {item.content}" for item in recent)
        return "\n".join(notes)

    def _witch_private_night_info(self, player: PlayerState, *, human_readable: bool = False) -> str:
        """女巫夜间必须知道当晚狼刀目标和药瓶状态。"""
        info = self._witch_night_info(player)
        if info.wolf_target_id is None:
            target_text = "暂无明确刀口"
        else:
            target = self.players[info.wolf_target_id]
            if human_readable:
                target_text = f"{target.id + 1}号 {target.name}"
            else:
                target_text = f"player_id={target.id}（{target.id + 1}号 {target.name}）"
        return (
            "女巫夜间信息："
            f"今晚狼人刀口是 {target_text}；"
            f"解药{'可用' if info.save_available else '已用'}，"
            f"{'本夜可以救该目标' if info.can_save_target else '本夜不能救该目标'}；"
            f"毒药{'可用' if info.poison_available else '已用'}。"
        )

    def _witch_night_info(self, player: PlayerState) -> WitchNightInfo:
        """构建女巫本夜结构化私有事实。"""
        wolf_target_id = self.wolf_consensus_target_id
        can_save_target = (
            self.witch_state.save_available
            and self._witch_can_save_target(player.id, wolf_target_id)
            and wolf_target_id is not None
        )
        return WitchNightInfo(
            witch_id=player.id,
            day=self.day,
            night_id=self.night_id,
            wolf_target_id=wolf_target_id,
            wolf_target_seat_no=wolf_target_id + 1 if wolf_target_id is not None else None,
            save_available=self.witch_state.save_available,
            poison_available=self.witch_state.poison_available,
            can_save_target=can_save_target,
            can_self_save=bool(can_save_target and wolf_target_id == player.id),
        )

    def _remember_private(self, player: PlayerState, content: str, data: dict | None = None) -> None:
        """写入单玩家私有记忆，避免 private_note 覆盖历史。"""
        memory = self.agent_memories.setdefault(player.id, AgentMemory(player_id=player.id))
        memory.private_observations.append(
            PrivateObservation(day=self.day, night_id=self.night_id, phase=self.phase.value, content=content, data=data or {})
        )
        if len(memory.private_observations) > 80:
            memory.private_observations = memory.private_observations[-80:]
        state = self.agent_states.get(player.id)
        if state:
            state.private_summary = self._compact_summary([state.private_summary, content], limit=360)
            state.memory_version += 1

    def _remember_observation(
        self,
        player: PlayerState,
        content: str,
        phase: str | None = None,
        data: dict | None = None,
    ) -> None:
        """写入某玩家可见观察。"""
        memory = self.agent_memories.setdefault(player.id, AgentMemory(player_id=player.id))
        memory.public_observations.append(
            PrivateObservation(
                day=self.day,
                night_id=self.night_id,
                phase=phase or self.phase.value,
                content=content,
                data=data or {},
            )
        )
        if len(memory.public_observations) > 80:
            memory.public_observations = memory.public_observations[-80:]
        state = self.agent_states.get(player.id)
        if state:
            state.public_summary = self._compact_summary([state.public_summary, content], limit=520)
            state.memory_version += 1

    def _broadcast_public_observation(self, content: str, phase: str | None = None, data: dict | None = None) -> None:
        """把公开可见信息写入每个玩家观察记忆。"""
        for player in self.players:
            self._remember_observation(player, content, phase=phase, data=data)

    def _record_message(
        self,
        message_type: str,
        phase: str,
        content: str,
        *,
        visibility: str = "public",
        speaker: PlayerState | None = None,
        action: str = "",
        target_id: int | None = None,
        target_role: RoleName | None = None,
        round_id: int | None = None,
        turn_index: int | None = None,
        visible_to_player_ids: list[int] | None = None,
    ) -> TableMessage:
        """写入统一消息日志。规则引擎仍是事实源，日志只承载可见事实。"""
        if visible_to_player_ids is None:
            if visibility == "public":
                visible_to_player_ids = [player.id for player in self.players]
            elif visibility == "wolf":
                visible_to_player_ids = [player.id for player in self.players if player.camp == Camp.WEREWOLF]
            elif speaker is not None:
                visible_to_player_ids = [speaker.id]
            else:
                visible_to_player_ids = []
        message = TableMessage(
            message_id=self.message_seq,
            day=self.day,
            night_id=self.night_id,
            phase=phase,
            message_type=message_type,  # type: ignore[arg-type]
            visibility=visibility,  # type: ignore[arg-type]
            speaker_id=speaker.id if speaker else None,
            speaker_seat_no=speaker.id + 1 if speaker else None,
            speaker_name=speaker.name if speaker else "",
            speaker_is_sheriff=speaker.is_sheriff if speaker else False,
            round_id=round_id,
            turn_index=turn_index,
            action=action,
            content=content,
            target_id=target_id,
            target_seat_no=target_id + 1 if target_id is not None else None,
            target_role=target_role,
            visible_to_player_ids=visible_to_player_ids,
            created_at=time.time(),
        )
        self.message_seq += 1
        self.message_log.append(message)
        if len(self.message_log) > 600:
            self.message_log = self.message_log[-600:]
        return message

    def _visible_messages_for_player(
        self,
        player: PlayerState,
        limit: int = 40,
        *,
        phase_scope: str | None = None,
    ) -> list[TableMessage]:
        """返回指定玩家能看到的 talk/whisper/action 事件。"""
        visible = [
            message
            for message in self.message_log
            if message.visibility == "public"
            or player.id in message.visible_to_player_ids
            or (message.visibility == "wolf" and player.camp == Camp.WEREWOLF)
        ]
        if phase_scope == "wolf_chat":
            # 狼聊原文只属于当前夜。历史夜晚只能通过摘要进入狼人私有视图。
            visible = [
                message
                for message in visible
                if not (message.visibility == "wolf" and message.phase == "wolf_chat" and message.night_id != self.night_id)
            ]
        else:
            # 狼人历史夜聊只能以摘要进入后续决策；原文不进入白天/投票上下文。
            visible = [
                message
                for message in visible
                if not (message.visibility == "wolf" and message.phase == "wolf_chat")
            ]
        return visible[-limit:]

    def _build_visible_timeline(self, player: PlayerState, limit: int = 80) -> list[VisibleTimelineItem]:
        """后端统一生成玩家可见时间线，避免前端多事实流重复/串场。"""
        items: list[VisibleTimelineItem] = []
        seen: set[str] = set()

        def add(item: VisibleTimelineItem) -> None:
            key = item.occurrence_key or item.item_id
            if key in seen:
                return
            seen.add(key)
            items.append(item)

        for event in self._visible_events_for_player(player):
            if event.phase == "wolf_chat" and not self._should_show_wolf_chat_event(event):
                continue
            add(
                VisibleTimelineItem(
                    item_id=f"event:{event.event_id}",
                    kind="event",
                    day=event.day,
                    night_id=event.night_id,
                    phase=event.phase,
                    visibility=event.visibility,
                    order=event.created_at or float(event.seq),
                    content=event.message,
                    occurrence_key=event.occurrence_key,
                )
            )

        for message in self._visible_messages_for_player(player, limit=300, phase_scope=self.phase.value):
            if message.phase == "wolf_chat" and message.night_id != self.night_id:
                continue
            add(
                VisibleTimelineItem(
                    item_id=f"message:{message.message_id}",
                    kind="message" if message.message_type != "whisper" else "wolf_chat",
                    day=message.day,
                    night_id=message.night_id,
                    phase=message.phase,
                    visibility=message.visibility,
                    order=message.created_at,
                    speaker_id=message.speaker_id,
                    speaker_seat_no=message.speaker_seat_no,
                    speaker_name=message.speaker_name,
                    speaker_is_sheriff=message.speaker_is_sheriff,
                    message_type=message.message_type,
                    action=message.action,
                    content=message.content,
                    target_id=message.target_id,
                    target_seat_no=message.target_seat_no,
                    occurrence_key=f"message:{message.message_id}",
                )
            )

        message_speech_keys = {
            (message.day, message.speaker_id, message.action)
            for message in self.message_log
            if message.message_type in {"talk", "last_words"} and message.speaker_id is not None
        }
        speech_action_by_type = {
            "campaign": "campaign_speech",
            "pk_campaign": "sheriff_pk_speech",
            "exile_pk": "exile_pk_speech",
            "day": "day_speech",
            "last_words": "last_words",
        }
        for index, speech in enumerate(self.speeches):
            action = speech_action_by_type.get(speech.speech_type, speech.speech_type)
            if (speech.day, speech.player_id, action) in message_speech_keys:
                continue
            add(
                VisibleTimelineItem(
                    item_id=f"speech:{speech.day}:{speech.player_id}:{speech.speech_type}:{index}",
                    kind="speech",
                    day=speech.day,
                    night_id=None,
                    phase=speech.speech_type,
                    visibility="public",
                    order=4000000000.0 + index,
                    speaker_id=speech.player_id,
                    speaker_seat_no=speech.player_id + 1,
                    speaker_name=speech.player_name,
                    content=speech.content,
                    occurrence_key=f"speech:{speech.day}:{speech.player_id}:{speech.speech_type}:{speech.content}",
                )
            )

        message_vote_keys = {
            (message.day, message.speaker_id, message.action, message.target_id)
            for message in self.message_log
            if message.message_type == "vote" and message.speaker_id is not None
        }
        vote_action_by_round = {
            "sheriff_vote": "sheriff_vote",
            "sheriff_pk_vote": "sheriff_pk_vote",
        }
        for index, vote in enumerate(self.votes):
            action = vote_action_by_round.get(vote.vote_round)
            if action is None:
                if vote.vote_round == f"day_{vote.day}_pk_exile":
                    action = "exile_pk_vote"
                elif vote.vote_type == "exile":
                    action = "day_vote"
                else:
                    action = vote.vote_type
            if (vote.day, vote.voter_id, action, vote.target_id) in message_vote_keys:
                continue
            add(
                VisibleTimelineItem(
                    item_id=f"vote:{vote.day}:{vote.voter_id}:{vote.vote_type}:{index}",
                    kind="vote",
                    day=vote.day,
                    phase=vote.vote_type,
                    visibility="public",
                    order=5000000000.0 + index,
                    speaker_id=vote.voter_id,
                    speaker_seat_no=vote.voter_id + 1,
                    speaker_name=vote.voter_name,
                    message_type="vote",
                    action=vote.vote_type,
                    content=f"{vote.voter_name} 投给 {vote.target_name}",
                    target_id=vote.target_id,
                    target_seat_no=vote.target_id + 1,
                    occurrence_key=f"vote:{vote.vote_round}:{vote.day}:{vote.voter_id}:{vote.target_id}",
                )
            )

        items.sort(key=lambda item: (item.day or 0, item.night_id or 0, item.order, item.item_id))
        return items[-limit:]

    def _new_visible_messages_for_player(
        self,
        player: PlayerState,
        memory: AgentMemory,
        limit: int = 16,
        *,
        phase_scope: str | None = None,
    ) -> list[TableMessage]:
        """返回该 Agent 上次决策后新增的可见消息。"""
        visible = [
            message
            for message in self._visible_messages_for_player(player, limit=200, phase_scope=phase_scope)
            if message.message_id > memory.last_seen_message_id
        ]
        return visible[-limit:]

    def _new_visible_events_for_player(self, player: PlayerState, memory: AgentMemory, limit: int = 12) -> list[GameEvent]:
        """返回该 Agent 上次决策后新增的可见系统事件。"""
        visible = [
            event
            for event in self._visible_events_for_player(player)
            if event.seq > memory.last_seen_event_seq
        ]
        return visible[-limit:]

    def _mark_agent_visibility_seen(
        self,
        memory: AgentMemory,
        visible_messages: list[TableMessage],
        visible_events: list[GameEvent],
    ) -> None:
        """提交本次 Agent 视角游标，下一次只强调新增信息。"""
        if visible_messages:
            memory.last_seen_message_id = max(memory.last_seen_message_id, max(message.message_id for message in visible_messages))
        if visible_events:
            memory.last_seen_event_seq = max(memory.last_seen_event_seq, max(event.seq for event in visible_events))

    def _status_map(self) -> dict[int, str]:
        """AIWolf 风格存活表。"""
        if self._should_hide_first_day_deaths():
            return {player.id: "ALIVE" for player in self.players}
        return {player.id: ("ALIVE" if player.alive else "DEAD") for player in self.players}

    def _known_role_map_for_player(self, player: PlayerState) -> dict[int, RoleName]:
        """按玩家视角返回已知身份，不给上帝视角。"""
        known = {player.id: player.role}
        if player.camp == Camp.WEREWOLF:
            for teammate in self.players:
                if teammate.camp == Camp.WEREWOLF:
                    known[teammate.id] = teammate.role
        if self.phase == Phase.GAME_OVER:
            return {item.id: item.role for item in self.players}
        return known

    def _talk_quota(self) -> dict[int, int]:
        """当前阶段每名玩家剩余公开发言次数。"""
        quota = {player.id: 0 for player in self.players}
        if self.phase in {
            Phase.DAY_SPEECH,
            Phase.SHERIFF_SPEECH,
            Phase.SHERIFF_PK_SPEECH,
            Phase.EXILE_PK_SPEECH,
            Phase.LAST_WORDS,
        }:
            for player_id in self.speech_order[self.speech_cursor :]:
                quota[player_id] = 1
            if self.phase == Phase.LAST_WORDS and self.current_exile_target_id is not None:
                quota[self.current_exile_target_id] = 1
        return quota

    def _whisper_quota(self) -> dict[int, int]:
        """当前夜狼人剩余夜聊发言次数。"""
        quota = {player.id: 0 for player in self.players}
        if self.phase == Phase.WOLF_CHAT:
            for player_id in self.speech_order[self.speech_cursor :]:
                quota[player_id] = 1
        return quota

    def _add_event(
        self,
        phase: str,
        message: str,
        visibility: str = "public",
        *,
        day: int | None = None,
        night_id: int | None = None,
        occurrence_key: str | None = None,
        visible_to_player_ids: list[int] | None = None,
    ) -> None:
        """写入带可见域的系统事件。"""
        event_day = self.day if day is None else day
        event_night_id = self.night_id if night_id is None else night_id
        event_key = occurrence_key or f"{phase}:{visibility}:{event_day}:{event_night_id}:{message}"
        if any(event.occurrence_key == event_key for event in self.events):
            return
        if visible_to_player_ids is None:
            if visibility == "public":
                visible_to_player_ids = [player.id for player in self.players]
            elif visibility == "wolf":
                visible_to_player_ids = [player.id for player in self.players if player.camp == Camp.WEREWOLF]
            else:
                visible_to_player_ids = []
        self.events.append(
            GameEvent(
                phase=phase,
                message=message,
                visibility=visibility,  # type: ignore[arg-type]
                day=event_day,
                night_id=event_night_id,
                seq=self.event_seq,
                created_at=time.time(),
                occurrence_key=event_key,
                visible_to_player_ids=visible_to_player_ids,
            )
        )
        self.event_seq += 1

    def _visible_events_for_player(self, player: PlayerState | None) -> list[GameEvent]:
        """按玩家过滤系统播报，狼聊内部事件不进入普通公开流。"""
        visible: list[GameEvent] = []
        seen_keys: set[str] = set()
        for event in self.events:
            if (
                event.visibility == "wolf"
                and event.phase == "wolf_chat"
                and not self._should_show_wolf_chat_event(event)
            ):
                continue
            if event.visibility == "public":
                allowed = True
            elif event.visibility == "wolf" and player is not None and player.camp == Camp.WEREWOLF:
                allowed = True
            elif event.visibility == "private" and player is not None and player.id in event.visible_to_player_ids:
                allowed = True
            else:
                allowed = False
            if not allowed:
                continue
            key = event.occurrence_key or f"{event.phase}:{event.visibility}:{event.day}:{event.night_id}:{event.message}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            visible.append(event)
        return visible

    def _should_show_wolf_chat_event(self, event: GameEvent) -> bool:
        """狼聊系统播报只展示当前夜上下文，历史夜晚改由摘要承载。"""
        if event.night_id != self.night_id:
            return False
        # 当前夜的狼聊事件对狼人始终可见；跨夜原文不展示，只保留摘要。
        return True

    def _update_agent_state_after_public_action(
        self,
        player: PlayerState,
        content: str,
        action_type: str,
        target_id: int | None = None,
    ) -> None:
        """更新单玩家持续状态，避免每次发言像重新开局。"""
        state = self.agent_states.get(player.id)
        if not state:
            return
        if action_type == "wolf_chat":
            plan_text = "本夜狼聊已发言"
            if target_id is not None:
                plan_text += "，已给出一个合法刀口方向"
            state.last_internal_plan = self._compact_summary(
                [state.last_internal_plan, f"第{self.day}天夜晚#{self.night_id}/狼聊: {plan_text}"],
                limit=520,
            )
            state.memory_version += 1
            return
        target_text = f"{target_id + 1}号" if target_id is not None else ""
        state.last_public_position = self._compact_summary(
            [state.last_public_position, f"第{self.day}天/{action_type}: {content} {target_text}"],
            limit=420,
        )
        state.current_focus = target_text or state.current_focus
        state.memory_version += 1

    def _compact_summary(self, lines: list[str], limit: int = 480) -> str:
        """本地压缩摘要，避免上下文无限增长。"""
        text = "；".join(line.strip() for line in lines if line and line.strip())
        if len(text) <= limit:
            return text
        return text[-limit:]

    def _current_wolf_chat_records(self) -> list[WolfChatRecord]:
        """只返回当前夜狼聊，历史夜晚必须走摘要。"""
        return [record for record in self.wolf_chat_records if record.night_id == self.night_id]

    def _current_wolf_night_plan(self) -> WolfNightPlan | None:
        """只返回当前夜狼队计划，避免前一夜计划泄漏到新夜快照。"""
        if self.wolf_night_plan is None or self.wolf_night_plan.night_id != self.night_id:
            return None
        return self.wolf_night_plan

    def _refresh_pending_human_action(self) -> None:
        """从当前阶段重建真人待操作标记，避免依赖结算函数副作用。"""
        if self.phase == Phase.GAME_OVER:
            self.pending_human_action = None
            return
        if self.phase == Phase.SETUP:
            self.pending_human_action = None
            return
        if self.phase == Phase.NIGHT:
            self.pending_human_action = "night" if self._human_has_required_night_action() else None
            return
        if self.phase == Phase.WOLF_CHAT:
            self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None
            return
        if self.phase in {Phase.DAY_SPEECH, Phase.EXILE_PK_SPEECH, Phase.SHERIFF_SPEECH, Phase.SHERIFF_PK_SPEECH}:
            phase_to_action = {
                Phase.DAY_SPEECH: "day_speech",
                Phase.EXILE_PK_SPEECH: "exile_pk_speech",
                Phase.SHERIFF_SPEECH: "sheriff_speech",
                Phase.SHERIFF_PK_SPEECH: "sheriff_pk_speech",
            }
            self.pending_human_action = phase_to_action[self.phase] if self.current_speaker_id == self.human_player_id else None
            return
        if self.phase == Phase.DAY_SPEECH:
            if self.sheriff_id == self.human_player_id and not self.speech_order and self.human_player.alive:
                self.pending_human_action = "choose_speech_order"
            else:
                self.pending_human_action = "day_speech" if self.current_speaker_id == self.human_player_id else None
            return
        if self.phase == Phase.DAY_VOTE:
            self.pending_human_action = "day_vote" if self.human_player.alive and self.human_player.can_vote else None
            return
        if self.phase == Phase.EXILE_PK_VOTE:
            self.pending_human_action = "day_vote" if self.human_player.alive and self.human_player.can_vote else None
            return
        if self.phase == Phase.SHERIFF_ELECTION:
            self.pending_human_action = "sheriff_election" if self.human_player.alive else None
            return
        if self.phase == Phase.SHERIFF_VOTE:
            self.pending_human_action = (
                "sheriff_vote"
                if self.human_player.alive and self.human_player_id not in self.sheriff_candidate_ids
                else None
            )
            return
        if self.phase == Phase.SHERIFF_PK_VOTE:
            self.pending_human_action = (
                "sheriff_vote"
                if self.human_player.alive and self.human_player_id not in self.sheriff_pk_candidate_ids
                else None
            )
            return
        if self.phase == Phase.LAST_WORDS:
            self.pending_human_action = "last_words" if self.current_exile_target_id == self.human_player_id else None
            return
        if self.phase == Phase.HUNTER_SHOT:
            self.pending_human_action = "hunter_shot" if self.pending_hunter_id == self.human_player_id else None
            return
        if self.phase == Phase.BADGE_TRANSFER:
            self.pending_human_action = "badge_transfer" if self.current_exile_target_id == self.human_player_id else None
            return
        self.pending_human_action = None

    def _wolf_history_summaries(self, *, include_current: bool = True) -> list[str]:
        """狼队可见的跨夜摘要，不包含历史夜晚原始聊天流。"""
        memory = self.camp_memories.get(Camp.WEREWOLF)
        if not memory:
            return []
        summaries = [
            summary
            for summary in memory.summaries
            if include_current or f"夜晚#{self.night_id}:" not in summary
        ]
        return [self._sanitize_wolf_history_summary(summary) for summary in summaries[-5:]]

    def _sanitize_wolf_history_summary(self, summary: str) -> str:
        """历史狼队摘要只给战略记忆，不保留会被复述成当前夜指令的原始标记。"""
        strategy_bits: list[str] = []
        if "队友确认" in summary:
            strategy_bits.append("上夜由队友明确确认，后续发言要围绕同一口径收束。")
        elif "提案多数" in summary:
            strategy_bits.append("上夜存在多个提案，后续需要更早统一刀口。")
        elif "规则兜底" in summary:
            strategy_bits.append("上夜共识不足，后续必须给出明确刀口和理由。")
        else:
            strategy_bits.append("上夜已形成狼队共识，后续注意白天口径不要暴露夜聊。")
        if "无明确建议" in summary:
            strategy_bits.append("上一夜缺少有效提案，本夜要直接给目标收益。")
        else:
            strategy_bits.append("只保留策略教训，不复述旧夜具体刀口。")
        return "过往夜晚复盘：" + "".join(strategy_bits)

    def _seat_ref(self, player: PlayerState, reveal_role: bool = False) -> SeatRef:
        """构建结构化座位引用。"""
        return SeatRef(
            player_id=player.id,
            seat_no=player.id + 1,
            name=player.name,
            alive=player.alive,
            is_sheriff=player.is_sheriff,
            role=player.role if reveal_role else None,
            camp=player.camp if reveal_role else None,
        )

    def _legal_action(self, action_type: str, target_ids: list[int], required: bool = False, note: str = "") -> LegalAction:
        """构建合法动作。"""
        return LegalAction(
            action_type=action_type,
            target_ids=target_ids,
            target_seats=[target_id + 1 for target_id in target_ids],
            required=required,
            note=note,
        )

    def _extract_mentioned_seats(self, content: str) -> list[int]:
        """从公开文本里提取被点到的号位。"""
        seats = []
        for raw in re.findall(r"(\d{1,2})号", content):
            seat = int(raw)
            if 1 <= seat <= self.player_count and seat not in seats:
                seats.append(seat)
        return seats

    def _extract_stance_keywords(self, content: str) -> list[str]:
        """提取可用于发言引用的立场关键词。"""
        candidates = [
            "查杀",
            "金水",
            "预言家",
            "女巫",
            "猎人",
            "白痴",
            "平民",
            "站边",
            "归票",
            "保",
            "打",
            "票",
            "抗推",
            "悍跳",
            "倒钩",
            "冲票",
            "逻辑",
            "身份",
            "发言",
        ]
        return [token for token in candidates if token in content][:6]

    def _infer_public_claims_from_speech(self, record: SpeechRecord) -> list[PublicClaimEvidence]:
        """从公开发言里粗提身份宣称，只作为可见证据，不作规则事实。"""
        role_patterns = {
            RoleName.SEER: [
                "我是预言家",
                "我跳预言家",
                "我起跳预言家",
                "我直接跳预言家",
                "我也把身份拍了：预言家",
                "我也把身份拍了:预言家",
                "我拍身份带节奏：预言家",
                "我拍身份带节奏:预言家",
                "我这里是真预视角",
                "我是真预",
                "我才是预言家",
                "昨晚我验到",
                "验人链",
                "报完整",
                "我的夜里信息指向",
                "夜里拿到的好人信息",
                "偏好人信息在我这里成立",
            ],
            RoleName.WITCH: ["我是女巫", "我跳女巫", "女巫牌"],
            RoleName.HUNTER: ["我是猎人", "我跳猎人", "猎人牌", "带枪"],
            RoleName.IDIOT: ["我是白痴", "我跳白痴", "白痴牌"],
            RoleName.VILLAGER: ["我是平民", "民牌", "平民牌"],
        }
        claims: list[PublicClaimEvidence] = []
        for role, patterns in role_patterns.items():
            if any(pattern in record.content for pattern in patterns):
                inspections = self._extract_public_inspections(record.content) if role == RoleName.SEER else []
                if not inspections:
                    inspections = [(None, None, None)]
                for inspected_target_id, inspected_target_seat_no, inspected_result in inspections:
                    claims.append(
                        PublicClaimEvidence(
                            day=record.day,
                            speaker_id=record.player_id,
                            speaker_seat_no=record.player_id + 1,
                            claimed_role=role,
                            source_text=record.content[:140],
                            inspected_target_id=inspected_target_id if role == RoleName.SEER else None,
                            inspected_target_seat_no=inspected_target_seat_no if role == RoleName.SEER else None,
                            inspected_result=inspected_result if role == RoleName.SEER else None,
                        )
                    )
        return claims

    def _extract_public_inspection(self, content: str) -> tuple[int | None, int | None, str | None]:
        """解析公开预言家口径里的查验对象和结果，仅作为公开宣称证据。"""
        inspections = self._extract_public_inspections(content)
        return inspections[-1] if inspections else (None, None, None)

    def _extract_public_inspections(self, content: str) -> list[tuple[int, int, str]]:
        """解析公开预言家发言中的一个或多个验人结果。"""
        patterns = [
            r"(?:验了|查验|验到|摸了)\s*(\d{1,2})号[^。！？!?，,；;]{0,12}(狼人|好人|金水|查杀)",
            r"(\d{1,2})号\s*给\s*(狼人|好人|金水|查杀)",
            r"(\d{1,2})号\s*(?:是|为)?\s*(狼人|好人|金水|查杀)",
            r"(\d{1,2})号\s*(金水|查杀)",
        ]
        inspections: list[tuple[int, int, str]] = []
        seen: set[tuple[int, str]] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                seat_no = int(match.group(1))
                if not 1 <= seat_no <= self.player_count:
                    continue
                raw_result = match.group(2)
                result = "狼人" if raw_result in {"狼人", "查杀"} else "好人"
                key = (seat_no, result)
                if key in seen:
                    continue
                seen.add(key)
                inspections.append((seat_no - 1, seat_no, result))
        return inspections

    def _recent_public_speech_evidence(self, limit: int = 10) -> list[PublicSpeechEvidence]:
        """构建最近公开发言证据。"""
        evidence: list[PublicSpeechEvidence] = []
        for record in self.speeches[-limit:]:
            evidence.append(
                PublicSpeechEvidence(
                    day=record.day,
                    speaker_id=record.player_id,
                    speaker_seat_no=record.player_id + 1,
                    speech_type=record.speech_type,
                    content=record.content,
                    mentioned_seat_nos=self._extract_mentioned_seats(record.content),
                    stance_keywords=self._extract_stance_keywords(record.content),
                )
            )
        return evidence

    def _recent_vote_evidence(self, limit: int = 12) -> list[VoteEvidence]:
        """构建最近公开票型证据。"""
        return [
            VoteEvidence(
                day=vote.day,
                voter_id=vote.voter_id,
                voter_seat_no=vote.voter_id + 1,
                target_id=vote.target_id,
                target_seat_no=vote.target_id + 1,
                vote_type=vote.vote_type,
                vote_round=vote.vote_round,
            )
            for vote in self.votes[-limit:]
        ]

    def _public_claim_evidence(self, limit: int = 12) -> list[PublicClaimEvidence]:
        """构建公开身份宣称证据。"""
        claims: list[PublicClaimEvidence] = []
        # 身份宣称是跨天锚点，不能像普通发言一样只看最近十几条。
        for record in self.speeches[-max(limit * 4, 48):]:
            claims.extend(self._infer_public_claims_from_speech(record))
        return claims[-limit:]

    def _seer_inspections_for_player(self, player: PlayerState) -> list[SeerInspectionFact]:
        """只把预言家本人的验人事实给本人。"""
        if player.role != RoleName.SEER:
            return []
        return [fact for fact in self.seer_inspection_facts if fact.seer_id == player.id]

    def _death_facts_for_player(self, player: PlayerState) -> list[DeathFact]:
        """给 Agent 可用的死亡事实。私有原因只在应公开后出现。"""
        if player.role == RoleName.HUNTER:
            own_facts = [fact for fact in self.death_facts if fact.player_id == player.id]
            if own_facts:
                return own_facts[-4:]
        return self.death_facts[-12:]

    def _death_fact_for(self, player_id: int, *, cause: str | None = None) -> DeathFact | None:
        """读取最近一次死亡事实。"""
        for fact in reversed(self.death_facts):
            if fact.player_id != player_id:
                continue
            if cause is not None and fact.cause != cause:
                continue
            return fact
        return None

    def _record_death_fact(
        self,
        player_id: int,
        cause: str,
        *,
        source_player_id: int | None = None,
        night_id: int | None = None,
    ) -> DeathFact:
        """记录单次死亡原因，供规则、Agent和测试共享。"""
        player = self.players[player_id]
        fact = DeathFact(
            player_id=player_id,
            seat_no=player_id + 1,
            cause=cause,
            day=self.day,
            night_id=night_id if night_id is not None else self.night_id,
            source_player_id=source_player_id,
            can_hunter_shoot=player.role == RoleName.HUNTER and cause in {"wolf_kill", "exile"},
        )
        self.death_facts.append(fact)
        return fact

    def _build_agent_context(
        self,
        player: PlayerState,
        phase: str,
        allowed_target_ids: list[int],
        prompt: str,
        action_type: str | None = None,
    ) -> AIContext:
        """统一生成 Agent 上下文。"""
        memory = self.agent_memories.setdefault(player.id, AgentMemory(player_id=player.id))
        private_observations = list(memory.private_observations[-12:])
        visible_messages = self._visible_messages_for_player(player, phase_scope=phase)
        new_visible_messages = self._new_visible_messages_for_player(player, memory, phase_scope=phase)
        new_visible_events = self._new_visible_events_for_player(player, memory)
        wolf_teammates: list[SeatRef] = []
        wolf_records: list[WolfChatRecord] = []
        if player.camp == Camp.WEREWOLF:
            wolf_teammates = [
                self._seat_ref(teammate, reveal_role=True)
                for teammate in self.players
                if teammate.camp == Camp.WEREWOLF and teammate.id != player.id
            ]
            wolf_records = self._current_wolf_chat_records()[-12:] if phase == "wolf_chat" else []
        if phase == "wolf_chat" and player.camp == Camp.WEREWOLF:
            visible_messages = self._filter_current_wolf_chat_view_messages(visible_messages)
            new_visible_messages = self._filter_current_wolf_chat_view_messages(new_visible_messages)
        agent_state = self.agent_states.get(player.id)

        structured = AgentVisibleContext(
            self_player=self._seat_ref(player, reveal_role=True),
            day=self.day,
            night_id=self.night_id,
            phase=phase,
            public_players=self._agent_public_seat_refs(),
            status_map=self._status_map(),
            known_role_map=self._known_role_map_for_player(player),
            talk_quota=self._talk_quota(),
            whisper_quota=self._whisper_quota(),
            visible_messages=visible_messages,
            new_visible_messages=new_visible_messages,
            new_visible_events=new_visible_events,
            private_observations=private_observations,
            recent_public_speeches=self._recent_public_speech_evidence(),
            recent_votes=self._recent_vote_evidence(),
            public_claims=self._public_claim_evidence(),
            seer_inspections=self._seer_inspections_for_player(player),
            witch_night_info=self._witch_night_info(player) if player.role == RoleName.WITCH and phase == "night_action" else None,
            death_facts=self._death_facts_for_player(player),
            idiot_reveals=list(self.idiot_reveal_facts[-8:]),
            legal_actions=[self._legal_action(action_type or phase, allowed_target_ids, note=prompt)],
            wolf_teammates=wolf_teammates,
            wolf_chat_records=wolf_records,
            wolf_history_summaries=self._wolf_history_summaries(include_current=phase != "wolf_chat") if player.camp == Camp.WEREWOLF else [],
            private_summary=agent_state.private_summary if agent_state else "",
            public_summary=agent_state.public_summary if agent_state else "",
            current_focus=agent_state.current_focus if agent_state else "",
        )
        self._mark_agent_visibility_seen(memory, new_visible_messages, new_visible_events)
        return AIContext(
            player_id=player.id,
            role=player.role,
            day=self.day,
            phase=phase,
            visible_state=self._agent_visible_state_text(player),
            allowed_target_ids=allowed_target_ids,
            prompt=prompt,
            persona_style=player.persona_style,
            strategy_style=player.strategy_style,
            structured=structured,
        )

    def _filter_current_wolf_chat_view_messages(self, messages: list[TableMessage]) -> list[TableMessage]:
        """狼聊决策只接收当前夜原始狼聊；历史夜晚只能走摘要，避免串夜复读。"""
        return [
            message
            for message in messages
            if not (
                message.visibility == "wolf"
                and message.phase == "wolf_chat"
                and message.night_id != self.night_id
            )
        ]

    def _agent_visible_state_text(self, player: PlayerState) -> str:
        """给 Agent 的可见文本：结构化摘要优先，避免白天机械复述原文。"""
        if self.phase == Phase.WOLF_CHAT and player.camp == Camp.WEREWOLF:
            return self._build_wolf_visible_state()

        alive = "、".join(
            f"{item.seat_no}号{'警长' if item.is_sheriff else ''}".strip()
            for item in self._agent_public_seat_refs()
            if item.alive
        )
        lines = [
            f"当前：第{self.day}天/{self.phase.value}",
            f"存活玩家：{alive or '无'}",
            "你的私有信息：",
            self._player_private_context(player),
        ]
        recent_speeches = self._recent_public_speech_evidence(limit=8)
        if recent_speeches:
            lines.append("公开发言证据摘要：")
            for item in recent_speeches[-6:]:
                mentioned = "、".join(f"{seat}号" for seat in item.mentioned_seat_nos) or "未明确点人"
                keywords = "、".join(item.stance_keywords) or "无"
                lines.append(f"- {item.speaker_seat_no}号提到{mentioned}，关键词[{keywords}]。")
        recent_votes = self._recent_vote_evidence(limit=8)
        if recent_votes:
            lines.append("公开票型摘要：")
            for vote in recent_votes[-8:]:
                lines.append(f"- 第{vote.day}天 {vote.voter_seat_no}号投{vote.target_seat_no}号。")
        public_events = [event for event in self._visible_events_for_player(player)[-8:] if event.visibility == "public"]
        if public_events:
            lines.append("公开系统播报：")
            lines.extend(f"- {event.message}" for event in public_events[-5:])
        visible_messages = self._visible_messages_for_player(player, limit=8, phase_scope=self.phase.value)
        if visible_messages:
            lines.append("可见桌面消息：")
            for message in visible_messages[-6:]:
                phase_label = {
                    "setup": "准备",
                    "wolf_chat": "狼人夜聊",
                    "night": "夜晚行动",
                    "last_words": "遗言",
                    "day_speech": "白天发言",
                    "day_vote": "放逐投票",
                    "exile_pk_speech": "放逐PK发言",
                    "exile_pk_vote": "放逐PK投票",
                    "hunter_shot": "猎人开枪",
                }.get(message.phase, message.phase)
                type_label = {
                    "talk": "发言",
                    "whisper": "夜聊",
                    "vote": "投票",
                    "night_action": "夜间信息",
                    "system": "系统",
                    "last_words": "遗言",
                }.get(message.message_type, message.message_type)
                speaker = f"{message.speaker_seat_no}号" if message.speaker_seat_no else "系统"
                target = ""
                if message.target_seat_no is not None:
                    if message.message_type == "vote":
                        target = f"，投给{message.target_seat_no}号"
                    elif message.message_type == "whisper":
                        target = f"，建议刀{message.target_seat_no}号"
                    else:
                        target = f"，指向{message.target_seat_no}号"
                lines.append(f"- 第{message.day}天{phase_label}，{speaker}{type_label}{target}：{message.content[:90]}")
        return "\n".join(lines)

    def _day_speech_goal(self, player: PlayerState) -> str:
        """按身份构造白天发言目标。"""
        base = (
            "请进行白天发言。不要机械点评上一位，也不要用'我先接X号'开头。"
            "直接给自己的视角：暂放谁、重点听谁、最不舒服谁、今天可能投谁。"
            "如果你点人，必须点透一个具体事实。第一天没有票型时，不要多人连续围绕同一句话打转。"
            "发言 45-100 字，像真人桌游发言。"
        )
        if player.role == RoleName.SEER:
            return (
                base
                + "你是预言家。必须把你的验人结果和白天归票逻辑关联起来。"
                "如果你选择起跳，要自然报验人和今天想出的目标；如果你选择暂藏，也要围绕验人结果保护或施压，不能忘记私有验人。"
            )
        if player.role == RoleName.WEREWOLF:
            teammates = [teammate.id + 1 for teammate in self.alive_wolves() if teammate.id != player.id]
            return (
                base
                + f"你是狼人，存活狼队友号位：{teammates or '无'}。"
                "你的白天目标是制造可执行好人票坑，必要时可以冲锋、倒钩、卖队友或装理中客。"
                "不要暴露狼队夜聊和队友身份；如果队友被打，可以选择硬保、轻踩切割或转移焦点，但必须像真好人。"
            )
        if player.role == RoleName.WITCH:
            return (
                base
                + f"你是女巫。解药{'还在' if self.witch_state.save_available else '已用'}，毒药{'还在' if self.witch_state.poison_available else '已用'}。"
                "白天不要随便暴露女巫视角，但你的发言要服务用药轮次：谁值得救、谁值得毒、谁在逼神。"
            )
        if player.role == RoleName.HUNTER:
            return (
                base
                + "你是猎人。白天要保留开枪威慑，不要把自己聊成纯暴民。"
                "如果你强点人，要让别人觉得你是有枪位压迫感，而不是无脑冲。"
            )
        if player.role == RoleName.IDIOT:
            return (
                base
                + "你是白痴。白天要像能抗推的民牌，不要提前暴露'我能翻牌'的松弛感。"
                "如果被推边缘，要留下能翻牌后反打的信息。"
            )
        return base + "你是平民。你的价值来自公开发言和票型判断，每轮至少给一个明确倾向，不要装神视角。"

    async def _decide_with_pipeline(
        self,
        player: PlayerState,
        phase: str,
        target_ids: list[int],
        action_kind: ActionKind,
        stage_goal: str,
    ):
        """通过统一管线进行 Agent 决策。"""
        if self.decision_pipeline is None or self.decision_pipeline.runtime is not self.runtime:
            self.decision_pipeline = DecisionPipeline(self.runtime)
        action_space = ActionSpace(
            phase=phase,
            actor_id=player.id,
            options=[
                ActionOption(
                    kind=action_kind,
                    label=stage_goal,
                    target_ids=target_ids,
                    required=bool(target_ids),
                    guidance="只能从这些目标里选择；没有目标则 target_id 为 null。",
                )
            ],
        )
        context = self._build_agent_context(
            player,
            phase,
            target_ids,
            prompt=stage_goal,
            action_type=action_kind,
        )
        request = DecisionRequest(
            player=player,
            context=context,
            action_space=action_space,
            preferred_action=action_kind,
            stage_goal=stage_goal,
        )
        return await self.decision_pipeline.decide(request, self.agent_states.get(player.id))

    async def _decide_target_with_pipeline(
        self,
        player: PlayerState,
        phase: str,
        target_ids: list[int],
        action_kind: ActionKind,
        stage_goal: str,
        fallback_target_id: int | None = None,
    ) -> int | None:
        """通过管线决策目标，并交给规则引擎做最终裁定。"""
        decision = await self._decide_with_pipeline(player, phase, target_ids, action_kind, stage_goal)
        return self._validated_target(
            player,
            action_kind,
            decision.target_id,
            target_ids,
            fallback_target_id=fallback_target_id if fallback_target_id is not None else (target_ids[0] if target_ids else None),
            reason=decision.reason,
        )

    def _validated_target(
        self,
        player: PlayerState,
        action: str,
        requested_target_id: int | None,
        legal_target_ids: list[int],
        fallback_target_id: int | None = None,
        reason: str = "",
        required: bool = True,
    ) -> int | None:
        """规则引擎统一校验 Agent/真人目标。"""
        corrected = False
        final_target_id = requested_target_id
        if final_target_id not in legal_target_ids:
            if requested_target_id is None and not required:
                final_target_id = None
            else:
                final_target_id = fallback_target_id if fallback_target_id in legal_target_ids else None
                corrected = requested_target_id is not None
        self.decision_audits.append(
            DecisionAudit(
                day=self.day,
                phase=self.phase.value,
                player_id=player.id,
                action=action,
                requested_target_id=requested_target_id,
                final_target_id=final_target_id,
                legal_target_ids=legal_target_ids[:],
                corrected=corrected,
                reason=reason,
            )
        )
        if corrected:
            # 审计只进入内部 decision_audits，不作为玩家可见系统播报。
            pass
        return final_target_id

    def _arm_auto_step_delay(self, seconds: float = 2.2) -> None:
        """给下一次自动推进设置短暂停顿，避免节奏过快。"""
        self.auto_step_ready_ts = time.time() + seconds

    def _announce_last_night_deaths_if_needed(self) -> None:
        """在需要时公布上一夜死讯。"""
        if self.day == 1 and not self.first_day_death_announcement_pending:
            return
        if self.last_night_deaths:
            death_names = "、".join(self.players[player_id].name for player_id in self.last_night_deaths)
            message = f"昨夜死亡：{death_names}"
        else:
            message = "昨夜是平安夜。"
        self._add_event("night", message)
        self._broadcast_public_observation(message, phase="night_result", data={"deaths": self.last_night_deaths[:]})
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
            if player.id != self.human_player_id:
                snapshot_player.private_note = ""
                snapshot_player.strategy_style = ""
                if not (self.human_player.camp == Camp.WEREWOLF and player.camp == Camp.WEREWOLF):
                    snapshot_player.role = RoleName.VILLAGER
                    snapshot_player.camp = Camp.VILLAGER
            snapshot_players.append(snapshot_player)
        return snapshot_players

    def _agent_public_seat_refs(self) -> list[SeatRef]:
        """构建 Agent 可见的公开座位表，不复用真人前端快照以免串视角。"""
        hide_deaths = self._should_hide_first_day_deaths()
        refs: list[SeatRef] = []
        for player in self.players:
            ref = self._seat_ref(player, reveal_role=False)
            if hide_deaths:
                ref.alive = True
            refs.append(ref)
        return refs

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
        self._refresh_pending_human_action()
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
            snapshot_seq=self.event_seq + self.message_seq + len(self.speeches) + len(self.votes) + len(self.night_summaries),
            phase=self.phase,
            day=self.day,
            night_id=self.night_id,
            sheriff_enabled=self.rule_profile.sheriff_enabled,
            guard_enabled=self.rule_profile.guard_enabled,
            human_player_id=self.human_player_id,
            human_role=self.human_player.role,
            human_alive=self.human_player.alive,
            human_is_wolf=self.human_player.camp == Camp.WEREWOLF,
            wolf_teammate_ids=wolf_teammate_ids,
            winner=self.winner,
            human_private_message=self.last_human_seer_result,
            human_private_context=self._player_private_context(self.human_player, human_readable=True),
            current_hint=self._build_current_hint(),
            human_allowed_night_actions=self._get_human_allowed_night_actions(),
            human_target_candidates=human_target_candidates,
            sheriff_id=self.sheriff_id,
            sheriff_candidates=self._preview_sheriff_candidates(),
            wolf_chat_records=self._current_wolf_chat_records() if self.human_player.camp == Camp.WEREWOLF else [],
            wolf_history_summaries=self._wolf_history_summaries() if self.human_player.camp == Camp.WEREWOLF else [],
            wolf_night_plan=self._current_wolf_night_plan() if self.human_player.camp == Camp.WEREWOLF else None,
            players=snapshot_players,
            speeches=self.speeches,
            votes=self.votes,
            night_summaries=self._build_snapshot_night_summaries(),
            events=self._visible_events_for_player(self.human_player)[-30:],
            visible_timeline=self._build_visible_timeline(self.human_player),
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
            return ["wolf_chat", "wolf_confirm"]
        if self.phase != Phase.NIGHT:
            return []
        if role == RoleName.SEER:
            return ["inspect", "skip"]
        if self.rule_profile.guard_enabled and role == RoleName.GUARD:
            return ["guard", "skip"]
        if role == RoleName.WITCH:
            actions = ["skip"]
            if self.witch_state.save_available and self._witch_can_save_target(self.human_player_id, self.wolf_consensus_target_id):
                actions.append("save")
            if self.witch_state.poison_available:
                actions.append("poison")
            return actions
        return ["skip"]

    def _human_has_required_night_action(self) -> bool:
        """真人有夜间身份动作时，轮询不能替他自动 skip。"""
        return self.phase == Phase.NIGHT and self.human_player.alive and bool(self._get_human_allowed_night_actions())

    def _auto_vote_candidates(self) -> list[int]:
        """自动推进投票时使用规则候选，不受真人是否死亡影响。"""
        if self.phase == Phase.DAY_VOTE:
            return [player.id for player in self.alive_players()]
        if self.phase == Phase.EXILE_PK_VOTE:
            return [player_id for player_id in self.exile_pk_candidate_ids]
        if self.phase == Phase.SHERIFF_VOTE:
            return [player_id for player_id in self.sheriff_candidate_ids]
        if self.phase == Phase.SHERIFF_PK_VOTE:
            return [player_id for player_id in self.sheriff_pk_candidate_ids]
        return []

    def _get_human_target_candidates(self) -> list[int]:
        """返回真人当前可选目标。"""
        if not self.human_player.alive and self.phase not in {Phase.HUNTER_SHOT, Phase.BADGE_TRANSFER}:
            return []

        if self.phase == Phase.DAY_VOTE:
            return [player.id for player in self.alive_players()]

        if self.phase == Phase.WOLF_CHAT:
            if self.human_player.role == RoleName.WEREWOLF:
                return [player.id for player in self.alive_players() if player.camp != Camp.WEREWOLF]
            return []

        if self.phase == Phase.NIGHT:
            role = self.human_player.role
            if self.rule_profile.guard_enabled and role == RoleName.GUARD:
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
        if self.phase == Phase.EXILE_PK_VOTE:
            return [player_id for player_id in self.exile_pk_candidate_ids]
        if self.phase == Phase.HUNTER_SHOT:
            if self.pending_hunter_id == self.human_player_id:
                return [player.id for player in self.alive_players() if player.id != self.pending_hunter_id]
            return []

        if self.phase == Phase.BADGE_TRANSFER:
            if self.current_exile_target_id != self.human_player_id:
                return []
            return [
                player.id
                for player in self.alive_players()
                if player.id != self.human_player_id
            ]

        return [player.id for player in self.alive_players() if player.id != self.human_player_id]

    def _build_current_hint(self) -> str:
        """构建当前阶段提示语。"""
        if self.phase == Phase.GAME_OVER:
            return f"本局已结束，获胜方：{self.winner or '未知'}。"
        if not self.human_player.alive and self.phase not in {Phase.LAST_WORDS, Phase.HUNTER_SHOT, Phase.BADGE_TRANSFER}:
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
            if self.rule_profile.guard_enabled and role == RoleName.GUARD:
                return "夜晚阶段：你是守卫，请选择一名玩家守护。"
            if role == RoleName.WITCH:
                return "夜晚阶段：你是女巫，可以选择救人、毒人或跳过。"
            return "夜晚阶段：你的角色今晚没有主动技能，系统会自动按跳过处理。"
        if self.phase == Phase.SHERIFF_ELECTION:
            return "上警报名阶段：决定你是否竞选警长。"
        if self.phase == Phase.SHERIFF_SPEECH:
            if self.current_speaker_id == self.human_player_id:
                return "警上发言轮到你了，请发表竞选发言。"
            speaker_name = self._current_speaker_name()
            return f"警上发言阶段：当前轮到 {speaker_name} 发言。" if speaker_name else "警上发言阶段：等待进入投票。"
        if self.phase == Phase.SHERIFF_PK_SPEECH:
            if self.current_speaker_id == self.human_player_id:
                return "警长 PK 发言轮到你了，请做最后陈述。"
            speaker_name = self._current_speaker_name()
            return f"警长 PK 发言阶段：当前轮到 {speaker_name} 发言。" if speaker_name else "警长 PK 发言阶段：等待进入投票。"
        if self.phase == Phase.SHERIFF_VOTE:
            return "警长投票阶段：未上警玩家对警上玩家投票。"
        if self.phase == Phase.SHERIFF_PK_VOTE:
            return "警长 PK 投票阶段：警下玩家重新投票。"
        if self.phase == Phase.EXILE_PK_SPEECH:
            if self.current_speaker_id == self.human_player_id:
                return "放逐 PK 发言轮到你了，请解释为什么今天不该出你。"
            speaker_name = self._current_speaker_name()
            return f"放逐 PK 发言阶段：当前轮到 {speaker_name} 发言。" if speaker_name else "放逐 PK 发言结束，等待重新投票。"
        if self.phase == Phase.EXILE_PK_VOTE:
            return "放逐 PK 投票阶段：只能在 PK 玩家中选择一名放逐目标。"
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
        if self.phase == Phase.HUNTER_SHOT:
            if self.pending_hunter_id == self.human_player_id:
                return "你是出局猎人，请选择一名存活玩家开枪带走。"
            if self.pending_hunter_id is not None:
                return f"{self.players[self.pending_hunter_id].name} 正在选择猎人开枪目标。"
            return "猎人开枪阶段。"
        if self.phase == Phase.BADGE_TRANSFER:
            if self.current_exile_target_id == self.human_player_id:
                return "你是死亡警长，请选择移交警徽或撕毁警徽。"
            return "警徽移交阶段。"
        return "准备开始。"

    def _current_speaker_name(self) -> str:
        """安全返回当前发言人姓名。"""
        speaker_id = self.current_speaker_id
        if speaker_id is None:
            return ""
        return self.players[speaker_id].name

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
        if self.phase == Phase.EXILE_PK_SPEECH and self.current_speaker_id is not None:
            speaker = self.players[self.current_speaker_id]
            return (f"exile_pk_speech_{self.day}_{self.speech_cursor}", f"{speaker.name} 放逐PK发言限时", 60)
        if self.phase == Phase.EXILE_PK_VOTE:
            return (f"exile_pk_vote_{self.day}", "放逐PK投票限时", 45)
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
        if self.phase == Phase.HUNTER_SHOT and self.pending_hunter_id is not None:
            speaker = self.players[self.pending_hunter_id]
            return (f"hunter_shot_{self.day}_{self.pending_hunter_id}", f"{speaker.name} 猎人开枪限时", 45)
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
            candidate_ids = self._auto_vote_candidates()
            await self.resolve_sheriff_election(
                SheriffAction(vote_target_id=candidate_ids[0] if candidate_ids else None)
            )
            self._arm_auto_step_delay()
            return
        if self.phase == Phase.SHERIFF_PK_VOTE and (not self.human_player.alive or self.pending_human_action is None):
            candidate_ids = self._auto_vote_candidates()
            await self.resolve_sheriff_election(
                SheriffAction(vote_target_id=candidate_ids[0] if candidate_ids else None)
            )
            self._arm_auto_step_delay()
            return
        if self.phase == Phase.EXILE_PK_VOTE and (not self.human_player.alive or self.pending_human_action is None or not self.human_player.can_vote):
            candidate_ids = self._auto_vote_candidates()
            if candidate_ids:
                await self.resolve_votes(candidate_ids[0])
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.NIGHT and self.pending_human_action is None and not self._human_has_required_night_action():
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
        if self.phase == Phase.EXILE_PK_SPEECH and self.current_speaker_id is not None:
            if self.current_speaker_id != self.human_player_id:
                await self._advance_exile_pk_speech("")
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.EXILE_PK_SPEECH and self.current_speaker_id is None:
            await self._advance_exile_pk_speech("")
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
        if self.phase == Phase.HUNTER_SHOT and self.pending_hunter_id is not None:
            if self.pending_hunter_id != self.human_player_id:
                await self.resolve_hunter_shot(None)
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.BADGE_TRANSFER and self.current_exile_target_id is not None:
            if self.current_exile_target_id != self.human_player_id:
                await self.resolve_badge_transfer(None)
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.DAY_VOTE and (not self.human_player.alive or self.pending_human_action is None or not self.human_player.can_vote):
            candidates = self._auto_vote_candidates()
            if candidates:
                await self.resolve_votes(candidates[0])
                self._arm_auto_step_delay()
            return
        if self.phase == Phase.WOLF_CHAT and self.current_speaker_id is not None:
            if self.current_speaker_id != self.human_player_id:
                await self.resolve_wolf_chat(None)
                self._arm_auto_step_delay()

    async def advance_ready_ai_steps(self, max_steps: int = 6, *, ignore_delay: bool = False) -> None:
        """一次请求内快速推进连续 AI 步骤，避免前端轮询一格一格卡住。"""
        with self._auto_runtime_mode():
            for _ in range(max_steps):
                before = (self.phase, self.current_speaker_id, self.speech_cursor, len(self.wolf_chat_records), len(self.speeches), len(self.votes))
                if ignore_delay:
                    self.auto_step_ready_ts = 0.0
                await self.advance_ready_ai_step_if_needed()
                after = (self.phase, self.current_speaker_id, self.speech_cursor, len(self.wolf_chat_records), len(self.speeches), len(self.votes))
                if after == before:
                    break
                if self.pending_human_action is not None or self.phase in {Phase.GAME_OVER, Phase.DAY_VOTE, Phase.NIGHT}:
                    break

    @contextmanager
    def _auto_runtime_mode(self):
        """自动轮询默认使用快速 fallback，防止外部 LLM 卡住页面。"""
        if settings.auto_ai_live:
            yield
            return
        if not hasattr(self.runtime, "enabled"):
            yield
            return
        previous_enabled = self.runtime.enabled
        self.runtime.enabled = False
        try:
            yield
        finally:
            self.runtime.enabled = previous_enabled

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
        if self.phase == Phase.EXILE_PK_SPEECH:
            await self._advance_exile_pk_speech("过。")
            return
        if self.phase == Phase.EXILE_PK_VOTE:
            candidates = self._get_human_target_candidates()
            if candidates:
                await self.resolve_votes(candidates[0])
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
        if self.phase == Phase.HUNTER_SHOT and self.pending_hunter_id is not None:
            await self.resolve_hunter_shot(None)
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
        if self.wolf_chat_prepared_night_id == self.night_id and self.speech_order:
            self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None
            return
        wolves = [player.id for player in self.alive_wolves()]
        self.speech_order = wolves
        self.speech_cursor = 0
        self.wolf_chat_round = 1
        self.wolf_chat_turn_index = 0
        if self.wolf_night_plan is None or self.wolf_night_plan.night_id != self.night_id:
            self.wolf_night_plan = WolfNightPlan(day=self.day, night_id=self.night_id)
        self.wolf_chat_prepared_night_id = self.night_id
        self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None
        if wolves and not self._current_wolf_chat_records() and not self._has_event_key(f"wolf_chat_start:{self.night_id}"):
            self._add_event(
                "wolf_chat",
                "狼人开始夜聊：先交换刀口收益，最后统一今晚落点。",
                "wolf",
                occurrence_key=f"wolf_chat_start:{self.night_id}",
            )

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

        if self.wolf_night_plan is None or self.wolf_night_plan.night_id != self.night_id:
            self.wolf_night_plan = WolfNightPlan(day=self.day, night_id=self.night_id)
        if self.wolf_consensus_target_id is not None or self.wolf_night_plan.finalized:
            self._enter_night_after_wolf_chat()
            return

        if not self.speech_order:
            self._prepare_wolf_chat_order()

        speaker_id = self.current_speaker_id
        if speaker_id is None:
            self._finish_or_continue_wolf_chat(candidate_ids)
            return

        speaker = self.players[speaker_id]
        is_final_confirm = bool(human_action and human_action.action_type == "wolf_confirm")
        previous_plan_target_id = self.wolf_night_plan.current_target_id if self.wolf_night_plan else None
        if speaker.is_human:
            requested_target_id = human_action.target_id if human_action else None
            if is_final_confirm and requested_target_id not in candidate_ids:
                raise ValueError("确认最终刀口时必须显式选择一个合法目标。")
            proposed_target_id = self._validated_target(
                speaker,
                "wolf_confirm" if is_final_confirm else "wolf_chat",
                requested_target_id,
                candidate_ids,
                fallback_target_id=self._fallback_wolf_target(candidate_ids),
                reason="human wolf chat",
            )
            content = (human_action.chat_content if human_action else "").strip()
            if not content:
                content = "我跟当前计划，理由是这张牌活到白天容易收束票口；明天口径分散，别一起补刀。"
        else:
            decision = await self._decide_with_pipeline(
                speaker,
                "wolf_chat",
                candidate_ids,
                "wolf_chat",
                (
                    f"你在第{self.wolf_chat_round}轮狼队夜聊。"
                    "目标是讨论出真实可执行刀口，不是讲套话。"
                    "说清楚：你想刀谁、为什么这个人今晚死收益最大、是否支持或反对当前计划。"
                    "禁止建议刀狼人队友；不要只说高价值位，要讲具体收益。"
                ),
            )
            proposed_target_id = self._validated_target(
                speaker,
                "wolf_chat",
                decision.target_id,
                candidate_ids,
                fallback_target_id=self._fallback_wolf_target(candidate_ids),
                reason=decision.reason,
            )
            content = (decision.content or "前面队友的刀口我能接，我补一条理由：先处理信息位。").strip()

        content = self._normalize_wolf_chat_content(content, proposed_target_id)
        stance = "final_confirm" if is_final_confirm else self._infer_wolf_stance(proposed_target_id, previous_plan_target_id)
        self._update_wolf_plan(speaker.id, proposed_target_id, candidate_ids, final_confirm=is_final_confirm)
        self.wolf_chat_records.append(
            WolfChatRecord(
                day=self.day,
                night_id=self.night_id,
                round_id=self.wolf_chat_round,
                turn_index=self.wolf_chat_turn_index,
                player_id=speaker.id,
                speaker_seat_no=speaker.id + 1,
                player_name=speaker.name,
                speaker_is_sheriff=speaker.is_sheriff,
                content=content,
                proposed_target_id=proposed_target_id,
                proposed_target_seat_no=proposed_target_id + 1 if proposed_target_id is not None else None,
                stance_to_previous=stance,
                is_valid_target=proposed_target_id in candidate_ids,
                created_at=time.time(),
            )
        )
        self._record_message(
            "whisper",
            "wolf_chat",
            content,
            visibility="wolf",
            speaker=speaker,
            action="wolf_confirm" if is_final_confirm else "wolf_chat",
            target_id=proposed_target_id,
            round_id=self.wolf_chat_round,
            turn_index=self.wolf_chat_turn_index,
        )
        self._append_wolf_shared_memory(speaker, content, proposed_target_id, is_final_confirm)
        self._update_agent_state_after_public_action(speaker, content, "wolf_chat", proposed_target_id)

        if self.wolf_night_plan.locked:
            self._finalize_wolf_chat(candidate_ids)
            return

        self.wolf_chat_turn_index += 1
        self.speech_cursor += 1
        if self.current_speaker_id is None:
            self._finish_or_continue_wolf_chat(candidate_ids)
            return
        self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None

    def _fallback_wolf_target(self, candidate_ids: list[int]) -> int | None:
        """无有效共识时给出稳定合法刀口。"""
        if self.wolf_night_plan and self.wolf_night_plan.current_target_id in candidate_ids:
            return self.wolf_night_plan.current_target_id
        for record in reversed(self.wolf_chat_records):
            if record.night_id == self.night_id and record.proposed_target_id in candidate_ids:
                return record.proposed_target_id
        return candidate_ids[0] if candidate_ids else None

    def _normalize_wolf_chat_content(self, content: str, proposed_target_id: int | None) -> str:
        """确保狼聊文本里的主目标与规则确认目标一致。"""
        content = content.strip()
        if proposed_target_id is None:
            return content
        target_seat = proposed_target_id + 1
        content = self._normalize_chinese_seat_mentions(content)
        wolf_seats = {player.id + 1 for player in self.alive_wolves()}
        for wolf_seat in sorted(wolf_seats):
            content = re.sub(rf"{wolf_seat}号\s*(?:是|为)?\s*(?:狼人|队友)", "队友身份不用在夜聊里重复", content)
            content = re.sub(rf"(?:刀|杀|砍|动|出|处理)\s*{wolf_seat}号", f"改刀{target_seat}号", content)
        mentioned_seats = [int(item) for item in re.findall(r"(\d+)号", content)]
        illegal_mentions = [seat for seat in mentioned_seats if seat in wolf_seats and seat != target_seat]
        if illegal_mentions:
            for illegal_seat in sorted(set(illegal_mentions)):
                content = re.sub(rf"{illegal_seat}号", f"{target_seat}号", content)
            mentioned_seats = [int(item) for item in re.findall(r"(\d+)号", content)]
        if target_seat in mentioned_seats:
            return content
        if mentioned_seats:
            first = mentioned_seats[0]
            content = re.sub(rf"{first}号", f"{target_seat}号", content, count=1)
            if target_seat in [int(item) for item in re.findall(r"(\d+)号", content)]:
                return content
        return f"我明确给刀口：{target_seat}号。{content}"

    def _normalize_chinese_seat_mentions(self, content: str) -> str:
        """把一号/二号这类中文号位转成 1号/2号，方便规则侧统一修正。"""
        chinese_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
            "十一": 11,
            "十二": 12,
        }
        for text, value in sorted(chinese_digits.items(), key=lambda item: len(item[0]), reverse=True):
            content = re.sub(rf"{text}号", f"{value}号", content)
        return content

    def _finish_or_continue_wolf_chat(self, candidate_ids: list[int]) -> None:
        """一轮夜聊结束后决定继续下一轮还是入夜。"""
        if self.wolf_chat_round < self.rule_profile.wolf_chat_rounds and not (self.wolf_night_plan and self.wolf_night_plan.locked):
            self.wolf_chat_round += 1
            self.speech_order = [player.id for player in self.alive_wolves()]
            self.speech_cursor = 0
            self.pending_human_action = "wolf_chat" if self.current_speaker_id == self.human_player_id else None
            round_event_key = f"wolf_chat_round:{self.night_id}:{self.wolf_chat_round}"
            if not self._has_event_key(round_event_key):
                self._add_event(
                    "wolf_chat",
                    f"狼队进入本夜第 {self.wolf_chat_round} 轮协商，继续复盘刀口。",
                    "wolf",
                    occurrence_key=round_event_key,
                )
            return
        self._finalize_wolf_chat(candidate_ids)

    def _has_wolf_event(self, message_prefix: str) -> bool:
        """判断当前夜是否已经写入某类狼队事件。"""
        return any(
            event.phase == "wolf_chat"
            and event.visibility == "wolf"
            and event.night_id == self.night_id
            and event.message.startswith(message_prefix)
            for event in self.events
        )

    def _has_event_key(self, occurrence_key: str) -> bool:
        """按稳定业务键判断事件是否已经写入。"""
        return any(event.occurrence_key == occurrence_key for event in self.events)

    def _finalize_wolf_chat(self, candidate_ids: list[int]) -> None:
        """锁定狼队刀口并进入夜晚技能阶段。"""
        if self.phase == Phase.NIGHT:
            return
        if self.wolf_night_plan is not None and self.wolf_night_plan.finalized:
            self._enter_night_after_wolf_chat()
            return
        if self.wolf_consensus_target_id is not None:
            if self.wolf_night_plan is not None:
                self.wolf_night_plan.finalized = True
            self._enter_night_after_wolf_chat()
            return
        target_id = None
        source = "default"
        if self.wolf_night_plan and self.wolf_night_plan.current_target_id in candidate_ids:
            target_id = self.wolf_night_plan.current_target_id
            source = self.wolf_night_plan.final_source or "plan"
        if target_id is None:
            proposals = [
                record.proposed_target_id
                for record in self.wolf_chat_records
                if record.night_id == self.night_id and record.proposed_target_id in candidate_ids
            ]
            if proposals:
                counts = Counter(proposals)
                target_id = sorted(counts, key=lambda item: (-counts[item], item))[0]
                source = "proposal_vote"
        if target_id is None:
            target_id = candidate_ids[0] if candidate_ids else None
            source = "engine_default"

        self.wolf_consensus_target_id = target_id
        if self.wolf_night_plan is None:
            self.wolf_night_plan = WolfNightPlan(day=self.day, night_id=self.night_id)
        self.wolf_night_plan.current_target_id = target_id
        self.wolf_night_plan.locked = True
        self.wolf_night_plan.finalized = True
        self.wolf_night_plan.final_source = source
        self._summarize_current_wolf_night(target_id, source)
        final_event_key = f"wolf_chat_final:{self.night_id}"
        if not self._has_event_key(final_event_key):
            self._add_event(
                "wolf_chat",
                (
                    "狼人夜谈结束，最终刀口："
                    + (f"{self.players[target_id].name}（{target_id + 1}号）。" if target_id is not None else "无合法目标。")
                ),
                "wolf",
                occurrence_key=final_event_key,
            )
        self._enter_night_after_wolf_chat()
        self._arm_auto_step_delay(1.4)

    def _enter_night_after_wolf_chat(self) -> None:
        """狼聊只允许单向进入夜晚，避免重复确认或改刀。"""
        self.phase = Phase.NIGHT
        self.pending_human_action = None
        self.speech_order = []
        self.speech_cursor = 0
        self.wolf_chat_turn_index = 0
        self._refresh_pending_human_action()

    def _summarize_current_wolf_night(self, target_id: int | None, source: str) -> None:
        """把当前夜狼聊压缩成跨夜摘要，避免下一夜串原始聊天。"""
        memory = self.camp_memories.setdefault(Camp.WEREWOLF, CampSharedMemory(camp=Camp.WEREWOLF))
        if any(summary.startswith(f"第{self.day}天夜晚#{self.night_id}") for summary in memory.summaries):
            return
        records = self._current_wolf_chat_records()
        source_text = {
            "human_confirm": "队友确认",
            "plan": "狼队共识",
            "proposal_vote": "提案多数",
            "engine_default": "规则兜底",
            "default": "规则兜底",
        }.get(source, "狼队共识")
        valid_proposal_count = sum(1 for record in records if record.proposed_target_id is not None)
        summary = (
            f"第{self.day}天夜晚#{self.night_id}: 形成方式{source_text}；"
            f"{'队友确认' if source == 'human_confirm' else '提案收束'}；"
            f"{'无明确建议' if valid_proposal_count == 0 else '本夜有明确刀口建议'}。"
        )
        memory.summaries.append(summary)
        if len(memory.summaries) > 8:
            memory.summaries = memory.summaries[-8:]

    def _update_wolf_plan(
        self,
        speaker_id: int,
        target_id: int | None,
        candidate_ids: list[int],
        *,
        final_confirm: bool = False,
    ) -> None:
        """根据夜聊建议更新狼队计划。"""
        if self.wolf_night_plan is None or target_id not in candidate_ids:
            return
        previous = self.wolf_night_plan.current_target_id
        if self.wolf_night_plan.locked:
            return
        if final_confirm:
            self.wolf_night_plan.current_target_id = target_id
            self.wolf_night_plan.final_confirmer_id = speaker_id
            self.wolf_night_plan.locked = True
            self.wolf_night_plan.final_source = "human_confirm"
            self.wolf_night_plan.supporters = sorted(set([*self.wolf_night_plan.supporters, speaker_id]))
            return
        proposals = [
            record.proposed_target_id
            for record in self.wolf_chat_records
            if record.night_id == self.night_id and record.proposed_target_id in candidate_ids
        ]
        proposals.append(target_id)
        counts = Counter(proposals)
        leading_target_id = sorted(counts, key=lambda item: (-counts[item], item))[0]
        self.wolf_night_plan.current_target_id = leading_target_id
        self.wolf_night_plan.final_source = "proposal_vote" if len(proposals) > 1 else "plan"
        supporter_ids = [
            record.player_id
            for record in self.wolf_chat_records
            if record.night_id == self.night_id and record.proposed_target_id == leading_target_id
        ]
        if target_id == leading_target_id:
            supporter_ids.append(speaker_id)
        self.wolf_night_plan.supporters = sorted(set(supporter_ids))
        self.wolf_night_plan.opponents = sorted(
            {
                record.player_id
                for record in self.wolf_chat_records
                if record.night_id == self.night_id and record.proposed_target_id in candidate_ids and record.proposed_target_id != leading_target_id
            }
            | ({speaker_id} if target_id != leading_target_id else set())
        )

    def _infer_wolf_stance(self, target_id: int | None, previous_target_id: int | None) -> str:
        """粗粒度标记本条狼聊对前序计划的态度。"""
        if target_id is None:
            return "skip"
        if previous_target_id is None:
            return "proposal"
        if target_id == previous_target_id:
            return "support"
        return "switch"

    def _append_wolf_shared_memory(
        self,
        speaker: PlayerState,
        content: str,
        proposed_target_id: int | None,
        is_final_confirm: bool,
    ) -> None:
        """写入狼队共享记忆。"""
        memory = self.camp_memories.setdefault(Camp.WEREWOLF, CampSharedMemory(camp=Camp.WEREWOLF))
        memory.records.append(
            {
                "day": self.day,
                "night_id": self.night_id,
                "round_id": self.wolf_chat_round,
                "speaker_id": speaker.id,
                "speaker_seat_no": speaker.id + 1,
                "content": content,
                "proposed_target_id": proposed_target_id,
                "proposed_target_seat_no": proposed_target_id + 1 if proposed_target_id is not None else None,
                "is_final_confirm": is_final_confirm,
            }
        )

    def _build_wolf_visible_state(self) -> str:
        """给狼人看的夜聊局面。"""
        wolf_names = [f"player_id={player.id}（{player.id + 1}号 {player.name}）" for player in self.alive_wolves()]
        legal_targets = [
            f"player_id={player.id}（{player.id + 1}号 {player.name}）"
            for player in self.alive_players()
            if player.camp != Camp.WEREWOLF
        ]
        public = self.public_state_text()
        wolf_chat_lines = [
            f"第{record.round_id}轮 {record.player_name}（{record.speaker_seat_no}号）: {record.content}"
            f"（建议刀 {record.proposed_target_seat_no if record.proposed_target_seat_no is not None else '无'}号）"
            for record in self._current_wolf_chat_records()[-8:]
        ]
        plan = "暂无"
        if self.wolf_night_plan and self.wolf_night_plan.current_target_id is not None:
            target_id = self.wolf_night_plan.current_target_id
            plan = f"{self.players[target_id].name}（{target_id + 1}号）"
        return "\n".join(
            [
                public,
                "狼人队友（不能作为刀口）：",
                *(wolf_names or ["无"]),
                "本夜合法刀口候选（只能从这里选）：",
                *(legal_targets or ["无"]),
                f"当前狼队计划刀口：{plan}",
                "本轮狼队夜聊：",
                *(wolf_chat_lines or ["暂无夜聊记录"]),
            ]
        )

    async def resolve_night(self, human_action: HumanNightAction | None = None) -> None:
        """执行夜晚结算。"""
        if self.phase not in (Phase.NIGHT, Phase.SETUP):
            return

        if not self.human_player.alive:
            human_action = None
        elif self.human_player.role not in (RoleName.SEER, RoleName.WITCH) and not (
            self.rule_profile.guard_enabled and self.human_player.role == RoleName.GUARD
        ):
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
            if self.rule_profile.guard_enabled and player.role == RoleName.GUARD:
                candidate_ids = [candidate.id for candidate in alive if candidate.id != self.guard_last_target_id]
                if player.is_human:
                    self.pending_human_action = "night"
                    if human_action and human_action.action_type == "guard" and human_action.target_id in candidate_ids:
                        guard_target_id = human_action.target_id
                else:
                    guard_target_id = await self._decide_target_with_pipeline(
                        player,
                        "night_action",
                        candidate_ids,
                        "guard",
                        "你是守卫。请选择今晚要守护的玩家，不能连续守护同一人。优先考虑白天被多方认可、可能吃刀的信息位。",
                        fallback_target_id=candidate_ids[0] if candidate_ids else None,
                    )
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
                    seer_target_id = await self._decide_target_with_pipeline(
                        player,
                        "night_action",
                        candidate_ids,
                        "inspect",
                        "你是预言家。先结合历史验人和白天发言判断今晚最该验谁；不要重复浪费已经形成铁信息的位置。",
                        fallback_target_id=candidate_ids[0] if candidate_ids else None,
                    )
                if seer_target_id is not None:
                    target = self.players[seer_target_id]
                    seer_result = "狼人" if target.camp == Camp.WEREWOLF else "好人"
                    self.seer_inspection_facts.append(
                        SeerInspectionFact(
                            seer_id=player.id,
                            target_id=target.id,
                            target_seat_no=target.id + 1,
                            result=seer_result,
                            day=self.day,
                            night_id=self.night_id,
                        )
                    )
                    player.private_note = f"你是预言家。历史查验结果：第{self.day}夜查验 {target.name} -> {seer_result}。"
                    self._record_message(
                        "night_action",
                        "night_action",
                        f"查验 {target.name}（{target.id + 1}号） -> {seer_result}",
                        visibility="private",
                        speaker=player,
                        action="inspect",
                        target_id=target.id,
                        target_role=target.role,
                    )
                    self._add_event(
                        "night_action",
                        f"你查验 {target.name}（{target.id + 1}号）的结果是：{seer_result}。",
                        "private",
                        visible_to_player_ids=[player.id],
                    )
                    self._remember_private(
                        player,
                        f"第{self.day}夜查验 {target.name}（{target.id + 1}号） -> {seer_result}。",
                        {"target_id": target.id, "target_seat_no": target.id + 1, "result": seer_result},
                    )
                    if player.is_human:
                        self.last_human_seer_result = f"{target.name}（{target.id + 1}号） 的查验结果是：{seer_result}"

        for player in alive:
            if player.role == RoleName.WITCH:
                poison_candidate_ids = [candidate.id for candidate in alive if candidate.id != player.id]
                action_candidate_ids = poison_candidate_ids[:]
                if (
                    self.witch_state.save_available
                    and self._witch_can_save_target(player.id, wolf_target_id)
                    and wolf_target_id is not None
                    and wolf_target_id not in action_candidate_ids
                ):
                    action_candidate_ids.append(wolf_target_id)
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
                            and human_action.target_id in poison_candidate_ids
                        ):
                            witch_poison_target_id = human_action.target_id
                            witch_used_action = True
                else:
                    decision = await self._decide_with_pipeline(
                        player,
                        "night_action",
                        action_candidate_ids,
                        "witch_action",
                        (
                            f"今晚狼人目标是 player_id={wolf_target_id}，"
                            f"{wolf_target_id + 1 if wolf_target_id is not None else '无'}号。"
                            "你是女巫。先判断是否值得用解药，再判断是否有足够理由开毒。"
                            "如果想救人，target_id 必须等于狼人目标；如果想毒人，target_id 选择你认为最高风险的存活玩家；如果不行动，target_id 返回 null。"
                            "注意：解药和毒药同一夜只能用一个。"
                        ),
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
                        and decision.target_id in poison_candidate_ids
                        and not witch_used_action
                    ):
                        witch_poison_target_id = decision.target_id
                        witch_used_action = True

                if witch_saved:
                    self.witch_state.save_available = False
                    player.private_note = f"你是女巫。你在第{self.day}夜使用了解药，救下了 {self.players[wolf_target_id].name if wolf_target_id is not None else '未知目标'}。"
                    self._record_message(
                        "night_action",
                        "night_action",
                        f"使用解药救下 {self.players[wolf_target_id].name if wolf_target_id is not None else '未知目标'}",
                        visibility="private",
                        speaker=player,
                        action="save",
                        target_id=wolf_target_id,
                        target_role=self.players[wolf_target_id].role if wolf_target_id is not None else None,
                    )
                    self._add_event(
                        "night_action",
                        f"你使用解药救下 {self.players[wolf_target_id].name if wolf_target_id is not None else '未知目标'}。",
                        "private",
                        visible_to_player_ids=[player.id],
                    )
                    self._remember_private(
                        player,
                        f"第{self.day}夜使用解药，救下 {self.players[wolf_target_id].name if wolf_target_id is not None else '未知目标'}。",
                        {"saved_target_id": wolf_target_id},
                    )
                if witch_poison_target_id is not None:
                    self.witch_state.poison_available = False
                    player.private_note = f"你是女巫。你在第{self.day}夜使用了毒药，毒杀了 {self.players[witch_poison_target_id].name}。"
                    self._record_message(
                        "night_action",
                        "night_action",
                        f"使用毒药毒杀 {self.players[witch_poison_target_id].name}（{witch_poison_target_id + 1}号）",
                        visibility="private",
                        speaker=player,
                        action="poison",
                        target_id=witch_poison_target_id,
                        target_role=self.players[witch_poison_target_id].role,
                    )
                    self._add_event(
                        "night_action",
                        f"你使用毒药毒杀 {self.players[witch_poison_target_id].name}（{witch_poison_target_id + 1}号）。",
                        "private",
                        visible_to_player_ids=[player.id],
                    )
                    self._remember_private(
                        player,
                        f"第{self.day}夜使用毒药，毒杀 {self.players[witch_poison_target_id].name}（{witch_poison_target_id + 1}号）。",
                        {"poison_target_id": witch_poison_target_id},
                    )
                self.witch_action_facts.append(
                    WitchActionFact(
                        witch_id=player.id,
                        day=self.day,
                        night_id=self.night_id,
                        wolf_target_id=wolf_target_id,
                        saved_target_id=wolf_target_id if witch_saved else None,
                        poison_target_id=witch_poison_target_id,
                        save_available_after=self.witch_state.save_available,
                        poison_available_after=self.witch_state.poison_available,
                    )
                )

        deaths: set[int] = set()
        death_causes: dict[int, str] = {}
        if wolf_target_id is not None and wolf_target_id != guard_target_id and not witch_saved:
            deaths.add(wolf_target_id)
            death_causes[wolf_target_id] = "wolf_kill"
        if witch_poison_target_id is not None:
            deaths.add(witch_poison_target_id)
            death_causes[witch_poison_target_id] = "witch_poison"
            if self.players[witch_poison_target_id].role == RoleName.HUNTER:
                self.hunter_poisoned = True

        for player_id in deaths:
            self.players[player_id].alive = False
            self._record_death_fact(player_id, death_causes.get(player_id, "wolf_kill"), night_id=self.night_id)

        summary = NightSummary(
            day=self.day,
            night_id=self.night_id,
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
        self.death_resolution_player_ids = sorted(deaths)
        self._check_winner()
        if self.phase == Phase.GAME_OVER:
            return
        if self.last_night_deaths:
            self.last_words_queue = self.last_night_deaths[:]
        if self.day == 1 and self.rule_profile.sheriff_enabled and self.sheriff_id is None:
            self.first_day_death_announcement_pending = True
            self.phase = Phase.SHERIFF_ELECTION
            self.pending_human_action = "sheriff_election" if self.human_player.alive else None
        else:
            self._announce_last_night_deaths_if_needed()
            if self.last_words_queue:
                self.phase = Phase.LAST_WORDS
                self.death_resolution_source = "night"
                self.current_exile_target_id = self.last_words_queue.pop(0)
                self.pending_human_action = "last_words" if self.current_exile_target_id == self.human_player_id else None
                return
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
            self._add_event("sheriff", "本局无人上警，跳过警长竞选。")
            self._announce_last_night_deaths_if_needed()
            self._prepare_day_speech_order()
            return

        self._add_event("sheriff", "上警名单：" + "、".join(self.players[player_id].name for player_id in self.sheriff_candidate_ids))
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
            decision = await self._decide_with_pipeline(
                speaker,
                "campaign_speech",
                [],
                "campaign_speech",
                (
                    "你正在竞选警长。这不是喊口号，必须像真人警上发言："
                    "要么给站边倾向，要么点一个你最在意的位置，要么说明你拿警徽后的第一处理顺序。"
                    "如果你像预言家视角，可以自然体现警徽流意识，但不要背模板。"
                    "不要只说'我带队''我控场''给我警徽'这种空话。"
                ),
            )
            content = decision.content.strip() or "我上警是为了整理逻辑，带大家找狼。"

        speaker.last_speech = content
        self._update_agent_state_after_public_action(speaker, content, "campaign_speech")
        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="campaign",
            )
        )
        self._record_message(
            "talk",
            "campaign_speech",
            content,
            speaker=speaker,
            action="campaign_speech",
            turn_index=self.speech_cursor,
        )
        self._broadcast_public_observation(
            f"{speaker.name}（{speaker.id + 1}号）警上发言：{content}",
            phase="campaign_speech",
            data={"speaker_id": speaker.id, "speech_type": "campaign"},
        )
        self.speech_cursor += 1

        if self.current_speaker_id is None:
            self._add_event("sheriff", "警上发言结束，开始投票选警长。")
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
            decision = await self._decide_with_pipeline(
                speaker,
                "pk_campaign_speech",
                [],
                "pk_campaign_speech",
                (
                    "你正在进行警长 PK 发言。必须正面回应别人为什么不该把警徽给对手，"
                    "而不是重复自己上一轮的话。优先抓对手一个具体矛盾、改口点或警徽流漏洞狠狠干。"
                ),
            )
            content = decision.content.strip() or "请大家回看我前面的逻辑和站边，我更适合拿警徽。"

        speaker.last_speech = content
        self._update_agent_state_after_public_action(speaker, content, "sheriff_pk_speech")
        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="pk_campaign",
            )
        )
        self._record_message(
            "talk",
            "pk_campaign_speech",
            content,
            speaker=speaker,
            action="pk_campaign_speech",
            turn_index=self.speech_cursor,
        )
        self._broadcast_public_observation(
            f"{speaker.name}（{speaker.id + 1}号）警长PK发言：{content}",
            phase="pk_campaign_speech",
            data={"speaker_id": speaker.id, "speech_type": "pk_campaign"},
        )
        self.speech_cursor += 1

        if self.current_speaker_id is None:
            self._add_event("sheriff", "警长 PK 发言结束，进入重新投票。")
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
                    target_id = await self._decide_target_with_pipeline(
                        player,
                        "sheriff_vote",
                        self.sheriff_candidate_ids,
                        "sheriff_vote",
                        "请在警上玩家中选择一位你认为更适合做警长的人。必须依据警上发言质量、信息量和带队可信度投票。",
                        fallback_target_id=self.sheriff_candidate_ids[0],
                    )

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
                self._record_message(
                    "vote",
                    "sheriff_vote",
                    f"警长投票给 {self.players[target_id].name}（{target_id + 1}号）",
                    speaker=player,
                    action="sheriff_vote",
                    target_id=target_id,
                )
                self._broadcast_public_observation(
                    f"{player.name}（{player.id + 1}号）警长投票给 {self.players[target_id].name}（{target_id + 1}号）。",
                    phase="sheriff_vote",
                    data={"voter_id": player.id, "target_id": target_id, "vote_type": "sheriff"},
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
                self._add_event("sheriff", "警长投票平票，进入 PK 发言环节。")
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
                target_id = await self._decide_target_with_pipeline(
                    player,
                    "sheriff_pk_vote",
                    self.sheriff_pk_candidate_ids,
                    "sheriff_pk_vote",
                    "警长 PK 投票，请在候选人中选一人。只能比较 PK 发言，不要投给候选外玩家。",
                    fallback_target_id=self.sheriff_pk_candidate_ids[0],
                )
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
            self._record_message(
                "vote",
                "sheriff_pk_vote",
                f"警长PK投票给 {self.players[target_id].name}（{target_id + 1}号）",
                speaker=player,
                action="sheriff_pk_vote",
                target_id=target_id,
            )
            self._broadcast_public_observation(
                f"{player.name}（{player.id + 1}号）警长PK投票给 {self.players[target_id].name}（{target_id + 1}号）。",
                phase="sheriff_pk_vote",
                data={"voter_id": player.id, "target_id": target_id, "vote_type": "sheriff_pk"},
            )

        max_vote = max(tally.values()) if tally else 0.0
        top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
        if len(top_ids) > 1:
            self._add_event("sheriff", "警长 PK 再次平票，本局警徽流失。")
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
        self._add_event("sheriff", f"{self.players[self.sheriff_id].name} 当选警长。")
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
            decision = await self._decide_with_pipeline(
                sheriff,
                "choose_speech_order",
                [],
                "choose_speech_order",
                "你是警长，请在 left 和 right 中选择一个白天发言方向，并在 content 中只输出 left 或 right。",
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
            decision = await self._decide_with_pipeline(
                speaker,
                "day_speech",
                [],
                "speak",
                self._day_speech_goal(speaker),
            )
            content = decision.content.strip() or "我先保留一点身份信息，重点看前后位逻辑。"

        speaker.last_speech = content
        self._update_agent_state_after_public_action(speaker, content, "day_speech")
        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="day",
            )
        )
        self._record_message(
            "talk",
            "day_speech",
            content,
            speaker=speaker,
            action="day_speech",
            turn_index=self.speech_cursor,
        )
        self._broadcast_public_observation(
            f"{speaker.name}（{speaker.id + 1}号）白天发言：{content}",
            phase="day_speech",
            data={"speaker_id": speaker.id, "speech_type": "day"},
        )
        self.speech_cursor += 1

        if self.current_speaker_id is None:
            self._add_event("speech", f"第 {self.day} 天白天发言结束。")
            self.phase = Phase.DAY_VOTE
            self.pending_human_action = "day_vote" if self.human_player.alive and self.human_player.can_vote else None
            return

        self.pending_human_action = "day_speech" if self.current_speaker_id == self.human_player_id else None

    async def resolve_votes(self, human_target_id: int) -> None:
        """执行白天投票。"""
        if self.phase == Phase.EXILE_PK_SPEECH and self.exile_pk_candidate_ids:
            await self._advance_exile_pk_speech("过。" if self.current_speaker_id != self.human_player_id else "")
            return
        if self.phase == Phase.EXILE_PK_VOTE and self.exile_pk_candidate_ids:
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
            candidates = alive_ids[:]
            if not candidates:
                continue
            if player.is_human:
                target_id = human_target_id if human_target_id in candidates else candidates[0]
            else:
                ai_candidates = [candidate_id for candidate_id in candidates if candidate_id != player.id]
                target_id = await self._decide_target_with_pipeline(
                    player,
                    "day_vote",
                    ai_candidates,
                    "vote",
                    "请选择今天放逐投票的目标。结合白天发言、票型预期、你的身份目标和私有记忆，投给你最想推出局的人。",
                    fallback_target_id=ai_candidates[0] if ai_candidates else candidates[0],
                )

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
            self._record_message(
                "vote",
                "day_vote",
                f"放逐投票给 {self.players[target_id].name}（{target_id + 1}号）",
                speaker=player,
                action="day_vote",
                target_id=target_id,
            )
            self._broadcast_public_observation(
                f"{player.name}（{player.id + 1}号）放逐投票给 {self.players[target_id].name}（{target_id + 1}号）。",
                phase="day_vote",
                data={"voter_id": player.id, "target_id": target_id, "vote_type": "exile"},
            )

        if not tally:
            self._add_event("vote", "本轮无人能够投票，直接进入黑夜。")
            self._advance_to_next_day()
            return

        max_vote = max(tally.values())
        top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
        if len(top_ids) > 1:
            self.exile_pk_candidate_ids = top_ids
            self.speech_order = top_ids[:]
            self.speech_cursor = 0
            self.phase = Phase.EXILE_PK_SPEECH
            self.pending_human_action = "exile_pk_speech" if self.current_speaker_id == self.human_player_id else None
            self._add_event("vote", "本轮放逐投票平票，进入 PK 辩护发言与重新投票。")
            return

        await self._execute_exile_outcome(top_ids[0])

    async def _advance_exile_pk_speech(self, human_speech: str) -> None:
        """推进放逐 PK 辩护发言。"""
        speaker_id = self.current_speaker_id
        if speaker_id is None:
            self.phase = Phase.EXILE_PK_VOTE
            self.pending_human_action = "day_vote" if self.human_player.alive and self.human_player.can_vote else None
            return

        speaker = self.players[speaker_id]
        if speaker.is_human:
            content = human_speech.strip() or "我补充一下，刚才那轮不该把我直接打死。"
        else:
            decision = await self._decide_with_pipeline(
                speaker,
                "exile_pk_speech",
                [candidate_id for candidate_id in self.exile_pk_candidate_ids if candidate_id != speaker.id],
                "exile_pk_speech",
                (
                    "你正在进行白天放逐 PK 辩护发言。必须解释为什么自己不该被今天推出去，"
                    "并点对手一个最硬的逻辑漏洞。不要复读上一轮，发言要更像被顶到刀口上的真人反击。"
                ),
            )
            opponent_id = next((candidate_id for candidate_id in self.exile_pk_candidate_ids if candidate_id != speaker.id), None)
            opponent = f"{opponent_id + 1}号" if opponent_id is not None else "对手位"
            fallback_lines = [
                f"PK我先保自己，今天直接出我收益太低。{opponent}一直借公共压力补刀，却没讲清自己的第一视角。",
                f"我这轮不该出。{opponent}的问题更具体：发言一直留退路，等票口成型后才补结论。",
                f"二选一我会反打{opponent}。我至少给了自己的判断，他更多是在顺着场上风向做抗推。",
            ]
            content = decision.content.strip() or fallback_lines[speaker.id % len(fallback_lines)]

        self.speeches.append(
            SpeechRecord(
                day=self.day,
                player_id=speaker.id,
                player_name=speaker.name,
                content=content,
                speech_type="exile_pk",
            )
        )
        self._record_message(
            "talk",
            "exile_pk_speech",
            content,
            speaker=speaker,
            action="exile_pk_speech",
            turn_index=self.speech_cursor,
        )
        self._update_agent_state_after_public_action(speaker, content, "exile_pk_speech")
        self._broadcast_public_observation(
            f"{speaker.name}（{speaker.id + 1}号）放逐PK发言：{content}",
            phase="exile_pk_speech",
            data={"speaker_id": speaker.id, "speech_type": "exile_pk"},
        )
        self.speech_cursor += 1
        self.pending_human_action = "exile_pk_speech" if self.current_speaker_id == self.human_player_id else None

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
                target_id = await self._decide_target_with_pipeline(
                    player,
                    "exile_pk_vote",
                    self.exile_pk_candidate_ids,
                    "exile_pk_vote",
                    "现在是放逐 PK 重新投票，只能在 PK 玩家里选一人出局。比较两人的辩解和前置发言，不要投给候选外玩家。",
                    fallback_target_id=self.exile_pk_candidate_ids[0],
                )

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
            self._record_message(
                "vote",
                "exile_pk_vote",
                f"PK投票给 {self.players[target_id].name}（{target_id + 1}号）",
                speaker=player,
                action="exile_pk_vote",
                target_id=target_id,
            )
            self._broadcast_public_observation(
                f"{player.name}（{player.id + 1}号）PK投票给 {self.players[target_id].name}（{target_id + 1}号）。",
                phase="exile_pk_vote",
                data={"voter_id": player.id, "target_id": target_id, "vote_type": "exile_pk"},
            )

        max_vote = max(tally.values()) if tally else 0.0
        top_ids = [candidate_id for candidate_id, score in tally.items() if score == max_vote]
        self.exile_pk_candidate_ids = []
        self.speech_order = []
        self.speech_cursor = 0

        if len(top_ids) != 1:
            message = "放逐 PK 再次平票，无人出局，直接进入黑夜。"
            self._add_event("vote", message)
            self._broadcast_public_observation(message, phase="exile_pk_vote")
            self.death_resolution_player_ids = []
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
            self.idiot_reveal_facts.append(
                IdiotRevealFact(player_id=out_player.id, seat_no=out_player.id + 1, day=self.day)
            )
            message = f"{out_player.name} 翻牌白痴，免于出局，但失去投票权。"
            self._add_event("vote", message)
            self._broadcast_public_observation(message, phase="exile_result", data={"player_id": out_player.id, "idiot_revealed": True})
            self._advance_to_next_day()
            return

        out_player.alive = False
        self._record_death_fact(out_id, "exile", night_id=None)
        message = f"{out_player.name} 被公投出局。"
        self._add_event("vote", message)
        self._broadcast_public_observation(message, phase="exile_result", data={"player_id": out_player.id})
        self.last_words_queue = [out_id]
        self.death_resolution_player_ids = [out_id]
        self.phase = Phase.LAST_WORDS
        self.death_resolution_source = "exile"
        self.current_exile_target_id = self.last_words_queue.pop(0)
        self.pending_human_action = "last_words" if self.current_exile_target_id == self.human_player_id else None

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
            speech = ""
        else:
            if player.is_human:
                speech = content.strip() or "我遗言结束，祝大家好运。"
            else:
                decision = await self._decide_with_pipeline(
                    player,
                    "last_words",
                    [],
                    "last_words",
                    (
                        "你已出局，请发表简短遗言。遗言只留最关键的信息："
                        "你最认的一张、最想点的一张、或你为什么会被推出去。不要像赛后总结，不要长篇大论。"
                    ),
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
            self._record_message(
                "last_words",
                "last_words",
                speech,
                speaker=player,
                action="last_words",
            )
            self._update_agent_state_after_public_action(player, speech, "last_words")
            self._broadcast_public_observation(
                f"{player.name}（{player.id + 1}号）遗言：{speech}",
                phase="last_words",
                data={"speaker_id": player.id, "speech_type": "last_words"},
            )

        if self.last_words_queue:
            self.current_exile_target_id = self.last_words_queue.pop(0)
            self.pending_human_action = "last_words" if self.current_exile_target_id == self.human_player_id else None
            return

        await self._continue_after_death_resolution()

    async def _continue_after_death_resolution(self) -> None:
        """遗言结束后继续处理猎人、警徽、胜负和入夜。"""
        if not self.death_resolution_player_ids:
            self._check_winner()
            if self.phase != Phase.GAME_OVER:
                self._continue_after_death_chain()
            return

        player_id = self.death_resolution_player_ids.pop(0)
        player = self.players[player_id]
        if player.role == RoleName.HUNTER:
            death_fact = self._death_fact_for(player.id)
            if death_fact and not death_fact.can_hunter_shoot:
                message = f"{player.name} 被女巫毒死，无法开枪。"
                self._add_event("hunter", message)
                self._broadcast_public_observation(message, phase="hunter_shot", data={"hunter_id": player.id, "blocked": True})
            else:
                candidates = [candidate.id for candidate in self.alive_players() if candidate.id != player.id]
                if candidates:
                    self.phase = Phase.HUNTER_SHOT
                    self.pending_hunter_id = player.id
                    self.pending_human_action = "hunter_shot" if player.is_human else None
                    return

        if player.is_sheriff:
            self.phase = Phase.BADGE_TRANSFER
            self.current_exile_target_id = player.id
            self.pending_human_action = "badge_transfer" if player.is_human else None
            return

        await self._continue_after_death_resolution()

    def _continue_after_death_chain(self) -> None:
        """死亡衍生流程结束后回到正确主线。"""
        source = self.death_resolution_source
        self.death_resolution_source = ""
        self.current_exile_target_id = None
        if source == "night":
            self._prepare_day_speech_order()
            return
        self._advance_to_next_day()

    async def resolve_badge_transfer(self, action: SheriffAction | None = None) -> None:
        """结算警徽移交。"""
        if self.phase != Phase.BADGE_TRANSFER or self.current_exile_target_id is None:
            return

        dead_sheriff = self.players[self.current_exile_target_id]
        dead_sheriff.is_sheriff = False
        self.sheriff_id = None
        alive_target_ids = [player.id for player in self.alive_players() if player.id != dead_sheriff.id]

        if dead_sheriff.is_human:
            if action and action.tear_badge:
                self._add_event("badge", f"{dead_sheriff.name} 选择撕毁警徽。")
            else:
                target_id = action.badge_target_id if action and action.badge_target_id in alive_target_ids else None
                if target_id is not None:
                    self.sheriff_id = target_id
                    self.players[target_id].is_sheriff = True
                    self._add_event("badge", f"{dead_sheriff.name} 将警徽移交给 {self.players[target_id].name}。")
                else:
                    self._add_event("badge", f"{dead_sheriff.name} 未移交警徽，警徽作废。")
        else:
            if alive_target_ids:
                decision = await self._decide_with_pipeline(
                    dead_sheriff,
                    "badge_transfer",
                    alive_target_ids,
                    "badge_transfer",
                    "你是死亡警长。请在存活玩家里选择一名你最认的人移交警徽；若都不认可，可返回空 target_id 视为撕徽。",
                )
                target_id = decision.target_id if decision.target_id in alive_target_ids else None
            else:
                target_id = None
            if target_id is not None:
                self.sheriff_id = target_id
                self.players[target_id].is_sheriff = True
                self._add_event("badge", f"{dead_sheriff.name} 将警徽移交给 {self.players[target_id].name}。")
            else:
                self._add_event("badge", f"{dead_sheriff.name} 撕毁了警徽。")

        self._check_winner()
        if self.phase != Phase.GAME_OVER:
            if self.death_resolution_player_ids:
                await self._continue_after_death_resolution()
            else:
                self._continue_after_death_chain()

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
            self._add_event("explode", f"{player.name} 在警上发言阶段自爆，警徽流失，直接进入黑夜。")
        else:
            self._add_event("explode", f"{player.name} 选择自爆，白天流程立即结束，直接进入黑夜。")
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
        self.death_resolution_player_ids = []
        self.death_resolution_source = ""
        self.pending_human_action = None
        self.speech_order = []
        self.speech_cursor = 0
        self.last_night_deaths = []
        self.day += 1
        self.night_id += 1
        self.wolf_consensus_target_id = None
        self.wolf_night_plan = None
        self.wolf_chat_prepared_night_id = None
        self.wolf_chat_round = 1
        self.wolf_chat_turn_index = 0
        self.phase = Phase.WOLF_CHAT
        self._prepare_wolf_chat_order()

    def _witch_can_save_target(self, witch_id: int, wolf_target_id: int | None) -> bool:
        """女巫首夜可自救，后续夜晚不可自救。"""
        if wolf_target_id is None:
            return False
        return self.day == 1 or wolf_target_id != witch_id

    def _check_winner(self) -> None:
        """检查胜负。"""
        wolves = len(self.alive_wolves())
        alive_goods = self.alive_villagers()
        gods = [player for player in alive_goods if player.role != RoleName.VILLAGER]
        villagers = [player for player in alive_goods if player.role == RoleName.VILLAGER]

        if wolves == 0:
            self.phase = Phase.GAME_OVER
            self.winner = "好人阵营"
            self._add_event("result", "游戏结束：好人阵营获胜。")
        elif not gods or not villagers:
            self.phase = Phase.GAME_OVER
            self.winner = "狼人阵营"
            self._add_event("result", "游戏结束：狼人阵营获胜。")

    async def resolve_hunter_shot(self, target_id: int | None) -> None:
        """结算猎人开枪。"""
        if self.phase != Phase.HUNTER_SHOT or self.pending_hunter_id is None:
            return

        hunter = self.players[self.pending_hunter_id]
        candidates = [player.id for player in self.alive_players() if player.id != hunter.id]
        if not candidates:
            self.pending_hunter_id = None
            self._check_winner()
            if self.phase != Phase.GAME_OVER:
                await self._continue_after_death_resolution()
            return

        if target_id not in candidates:
            if hunter.is_human:
                target_id = candidates[0]
            else:
                target_id = await self._decide_target_with_pipeline(
                    hunter,
                    "hunter_shot",
                    candidates,
                    "hunter_shot",
                    "你是出局猎人，请选择一名存活玩家开枪带走。优先带走你遗言前最确信的狼坑，不要开枪给空目标。",
                    fallback_target_id=candidates[0],
                )

        target = self.players[target_id]
        target.alive = False
        self._record_death_fact(target.id, "hunter_shot", source_player_id=hunter.id, night_id=None)
        self.death_resolution_player_ids.append(target.id)
        self.pending_hunter_id = None
        self.pending_human_action = None
        self._add_event("hunter", f"{hunter.name} 死亡开枪，带走了 {target.name}。")
        self._broadcast_public_observation(
            f"{hunter.name} 死亡开枪，带走了 {target.name}。",
            phase="hunter_shot",
            data={"hunter_id": hunter.id, "target_id": target.id},
        )
        self._check_winner()
        if self.phase != Phase.GAME_OVER:
            await self._continue_after_death_resolution()


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
