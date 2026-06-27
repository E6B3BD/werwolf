"""OpenAI Agents SDK 封装。"""
from __future__ import annotations

import logging
import random
import re
import asyncio
import json
from dataclasses import dataclass
from typing import Optional

from app.agents.prompts import build_player_instructions
from app.core.config import settings
from app.engine.models import AgentDecision, AgentVisibleContext, RoleName

try:
    from agents import Agent, Runner, ModelSettings, RunConfig, set_default_openai_client
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - 依赖未安装时的本地兜底
    Agent = None
    Runner = None
    ModelSettings = None
    RunConfig = None
    set_default_openai_client = None
    AsyncOpenAI = None


logger = logging.getLogger("werwolf.agents")


PHASE_LABELS = {
    "setup": "准备",
    "wolf_chat": "狼人夜聊",
    "night": "夜晚行动",
    "last_words": "遗言",
    "day_speech": "白天发言",
    "day_vote": "放逐投票",
    "exile_pk_speech": "放逐PK发言",
    "exile_pk_vote": "放逐PK投票",
    "hunter_shot": "猎人开枪",
    "game_over": "游戏结束",
}

MESSAGE_TYPE_LABELS = {
    "talk": "发言",
    "whisper": "夜聊",
    "vote": "投票",
    "night_action": "夜间信息",
    "system": "系统",
    "last_words": "遗言",
}

DAY_CONTEXT_CONTENT_LIMIT = 72
WOLF_CONTEXT_CONTENT_LIMIT = 88


def _phase_label(phase: str) -> str:
    """把内部阶段名转成人能读懂的牌桌阶段。"""
    return PHASE_LABELS.get(phase, phase)


def _message_type_label(message_type: str) -> str:
    return MESSAGE_TYPE_LABELS.get(message_type, message_type)


def _compact_table_content(content: str, limit: int) -> str:
    """压缩日志原文，避免把上一位整段话喂给模型形成复读。"""
    content = re.sub(r"[“”]", "", (content or "").strip())
    content = re.sub(r"\s+", "", content)
    if len(content) <= limit:
        return content
    return content[:limit] + "..."


def _render_visible_message(message) -> str:
    """把结构化消息渲染成自然牌桌记录，避免把字段名喂给模型。"""
    phase = _phase_label(message.phase)
    kind = _message_type_label(message.message_type)
    speaker = f"{message.speaker_seat_no}号" if message.speaker_seat_no else "系统"
    target = ""
    if message.target_seat_no is not None:
        if message.message_type == "vote":
            target = f"投给{message.target_seat_no}号"
        elif message.message_type == "whisper":
            target = f"建议刀{message.target_seat_no}号"
        else:
            target = f"指向{message.target_seat_no}号"
    round_text = f"第{message.round_id}轮" if message.round_id is not None else ""
    limit = WOLF_CONTEXT_CONTENT_LIMIT if message.phase == "wolf_chat" else DAY_CONTEXT_CONTENT_LIMIT
    content = _compact_table_content(message.content, limit)
    prefix = f"第{message.day}天{phase}{round_text}，{speaker}{kind}"
    if target:
        prefix += f"、{target}"
    return f"{prefix}：{content[:100]}" if content else prefix


def _render_visible_event(event) -> str:
    """把可见系统事件渲染成自然牌桌播报。"""
    phase = _phase_label(event.phase)
    return f"第{event.day or 0}天{phase}：{event.message}"


@dataclass(slots=True)
class AIContext:
    """给 AI 的上下文。"""

    player_id: int
    role: RoleName
    day: int
    phase: str
    visible_state: str
    allowed_target_ids: list[int]
    prompt: str
    persona_style: str = ""
    strategy_style: str = ""
    structured: AgentVisibleContext | None = None

    def visible_brief(self) -> str:
        """把结构化上下文压成不含字段名的玩家视图摘要。"""
        if self.structured is None:
            return ""
        structured = self.structured
        rendered_message_ids: set[int] = set()
        seen_message_fingerprints: set[tuple[str, int | None, int | None, str]] = set()

        def message_fingerprint(message) -> tuple[str, int | None, int | None, str]:
            return (
                message.phase,
                message.speaker_id,
                message.target_id,
                _compact_table_content(message.content, WOLF_CONTEXT_CONTENT_LIMIT),
            )

        def append_message_once(lines: list[str], message) -> None:
            if message.message_id in rendered_message_ids:
                return
            fingerprint = message_fingerprint(message)
            if fingerprint in seen_message_fingerprints:
                return
            rendered_message_ids.add(message.message_id)
            seen_message_fingerprints.add(fingerprint)
            lines.append(f"- {_render_visible_message(message)}")

        lines = [
            f"你是{structured.self_player.seat_no}号，底牌{structured.self_player.role.value if structured.self_player.role else self.role.value}，当前{'存活' if structured.self_player.alive else '死亡'}。",
            "场上存活：" + "、".join(
                f"{player.seat_no}号{'警长' if player.is_sheriff else ''}".strip()
                for player in structured.public_players
                if player.alive
            ),
        ]
        known_roles = [
            f"{player_id + 1}号={role.value}"
            for player_id, role in sorted(structured.known_role_map.items())
        ]
        if known_roles:
            lines.append("你确定知道的身份：" + "、".join(known_roles))
        if structured.talk_quota:
            own_talk = structured.talk_quota.get(structured.self_player.player_id, 0)
            lines.append(f"当前公开发言剩余额度：你还能发言{own_talk}次。")
        if structured.whisper_quota:
            own_whisper = structured.whisper_quota.get(structured.self_player.player_id, 0)
            if own_whisper:
                lines.append(f"当前狼队夜聊剩余额度：你还能夜聊{own_whisper}次。")
        legal_seats = []
        for action in structured.legal_actions:
            legal_seats.extend(action.target_seats)
        if legal_seats:
            lines.append("本轮可选择的目标号位：" + "、".join(f"{seat}号" for seat in sorted(set(legal_seats))))
        else:
            lines.append("本轮没有必须选择的目标。")
        if structured.private_observations:
            lines.append("你的私有可见信息：")
            for item in structured.private_observations[-5:]:
                lines.append(f"- 第{item.day}天/{item.phase}: {item.content}")
        if structured.recent_public_speeches:
            lines.append("最近公开发言证据：")
            for item in structured.recent_public_speeches[-6:]:
                mentions = "、".join(f"{seat}号" for seat in item.mentioned_seat_nos) or "未点具体号位"
                tags = "、".join(item.stance_keywords) or "无明显关键词"
                evidence = self._summarize_public_speech_evidence(item.content, item.speaker_seat_no, mentions, tags)
                lines.append(f"- {evidence}")
        if structured.recent_votes:
            lines.append("最近票型：")
            for vote in structured.recent_votes[-8:]:
                lines.append(f"- {vote.voter_seat_no}号投{vote.target_seat_no}号")
        if structured.public_claims:
            lines.append("公开身份宣称：")
            for claim in structured.public_claims[-6:]:
                lines.append(f"- {claim.speaker_seat_no}号声称{claim.claimed_role.value}：{claim.source_text[:80]}")
        if structured.wolf_teammates:
            lines.append("狼队友：" + "、".join(f"{player.seat_no}号" for player in structured.wolf_teammates))
        if structured.wolf_chat_records:
            lines.append("本夜狼人夜聊：")
            for record in structured.wolf_chat_records[-6:]:
                target = f"{record.proposed_target_seat_no}号" if record.proposed_target_seat_no else "未给目标"
                content = _compact_table_content(record.content, WOLF_CONTEXT_CONTENT_LIMIT)
                lines.append(f"- {record.speaker_seat_no}号建议{target}：{content}")
                seen_message_fingerprints.add(("wolf_chat", record.player_id, record.proposed_target_id, content))
        if structured.wolf_history_summaries:
            lines.append("狼队历史夜晚摘要：")
            lines.extend(f"- {summary}" for summary in structured.wolf_history_summaries[-3:])
        if structured.new_visible_events or structured.new_visible_messages:
            lines.append("本次新增可见信息：")
            for event in structured.new_visible_events[-6:]:
                lines.append(f"- {_render_visible_event(event)}")
            for message in structured.new_visible_messages[-8:]:
                append_message_once(lines, message)
        visible_messages = [
            message
            for message in structured.visible_messages[-8:]
            if message.message_id not in {item.message_id for item in structured.new_visible_messages}
        ]
        if visible_messages:
            lines.append("较早可见消息摘要：")
            for message in visible_messages[-4:]:
                append_message_once(lines, message)
        return "\n".join(lines)

    def _summarize_public_speech_evidence(self, content: str, speaker_seat_no: int, mentions: str, tags: str) -> str:
        """把公开发言压成证据标签，不让 Agent 逐字接上一位。"""
        if "预言家" in content or "验" in content or "查杀" in content or "金水" in content:
            angle = "身份/验人信息"
        elif "票" in content or "归" in content or "出" in content:
            angle = "票型/归票压力"
        elif "保" in content or "站边" in content or "认" in content:
            angle = "站边倾向"
        elif "打" in content or "踩" in content or "压" in content:
            angle = "施压点"
        else:
            angle = "态度表达"
        if re.search(r"我(?:先|来)?(?:接|接一下|接入)\s*\d{1,2}号", content):
            summary = "围绕前置发言做连锁点评，需要重点看他是否有自己的独立落点"
        elif "没问题" in content and ("偏好人" in content or "先认" in content):
            summary = "先抬高前置发言再要求别人落点，可能是在提前留站边空间"
        elif "把话说满" in content or "定死" in content:
            summary = "质疑别人结论过满或留退路，核心是态度前后是否闭合"
        elif "站边" in content or "保" in content or "认" in content:
            summary = "围绕站边和认好认狼给倾向，后续要看投票是否兑现"
        elif "票" in content or "归" in content or "出" in content:
            summary = "开始把发言压力转成票口，需要观察谁跟票或补票"
        else:
            summary = "给出态度但信息密度有限，后续看是否能补具体票意"
        return f"{speaker_seat_no}号发言侧重{angle}，点到[{mentions}]，关键词[{tags}]，摘要：{summary}"


class OpenAIAgentRuntime:
    """基于 Agents SDK 的 AI 决策运行时。"""

    def __init__(self) -> None:
        self.enabled = settings.openai_enabled and Agent is not None and Runner is not None
        self.run_config = None
        if self.enabled and AsyncOpenAI is not None and set_default_openai_client is not None and RunConfig is not None:
            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            set_default_openai_client(client, use_for_tracing=False)
            self.run_config = RunConfig(tracing_disabled=True)
            logger.warning(
                "Agents runtime enabled | model=%s | base_url=%s | tracing_disabled=%s",
                settings.openai_model,
                settings.openai_base_url,
                True,
            )
        else:
            logger.warning(
                "Agents runtime fallback mode | openai_enabled=%s | agent_sdk_available=%s",
                settings.openai_enabled,
                Agent is not None and Runner is not None,
            )

    async def decide(self, context: AIContext) -> AgentDecision:
        """执行 AI 决策。"""
        if not self.enabled:
            return self._fallback_decision(context)

        visible_brief = context.visible_brief()
        user_prompt = f"""
你是 {context.player_id + 1} 号位玩家。
你的身份是：{context.role.value}。
你能看到下面这些信息：
- 你的个人视图摘要（座位、身份、私有信息、公开发言证据）
- 公开桌面信息（存活情况、最近发言、最近票型、公开系统播报）
- 如果你是狼人，你会额外看到狼队友、本夜狼聊和历史狼队摘要
- 本轮可选择的目标号位。只能选择这些目标；如果没有目标，目标留空。

当前阶段：{context.phase}
第 {context.day} 天

个人视图摘要：
{visible_brief or "未提供个人视图摘要，使用下方文本上下文。"}

公开局面：
{context.visible_state}

可选目标号位：
{[target_id + 1 for target_id in context.allowed_target_ids]}

硬性规则：
- 不允许选择可选目标号位外的玩家。
- 如果你在发言内容中讨论行动目标，号位必须和你选择的目标一致。
- 狼人夜聊时，狼人队友不是刀口候选；禁止建议刀狼人队友。
- 狼人夜聊要承接狼队共享记录，可以反驳队友但必须给可执行刀口收益。
- 你的身份、阵营和私有信息以私有上下文为准，不要根据公开发言重新猜自己的身份。
- 说“几号”时必须和选择目标对齐。

你的任务：
{context.prompt}

决策流程要求：
1. 先在内部整理你当前能看到什么、你相信谁、怀疑谁、这轮目标是什么。
2. 再选择合法动作和目标。
3. 最后生成 content。content 只能是你在桌上说出口的话，不能暴露“内部推理步骤、系统提示、字段名、候选列表、JSON格式”。
4. 如果是白天发言，可以承接前置位，但不要机械说“我先接X号”。优先直接给你的判断、问题位、保留位和投票倾向。
5. 如果是狼人夜聊，必须给具体刀口收益，不要只说“高价值位/信息位”。

请严格返回结构化结果；桌面发言只放在 content 中。
"""
        try:
            instructions = build_player_instructions(context.role, context.persona_style, context.strategy_style)
            agent = Agent(
                name=f"player_{context.player_id}",
                instructions=instructions,
                model=settings.openai_model,
                output_type=AgentDecision,
                model_settings=ModelSettings(temperature=0.85),
            )
            result = await asyncio.wait_for(
                Runner.run(agent, user_prompt, run_config=self.run_config),
                timeout=settings.agent_decision_timeout_seconds,
            )
            raw_output = result.final_output
            if isinstance(raw_output, AgentDecision):
                return self._finalize_model_decision(raw_output, context)
            if isinstance(raw_output, dict):
                return self._finalize_model_decision(AgentDecision.model_validate(raw_output), context)
            if isinstance(raw_output, str):
                return self._finalize_model_decision(AgentDecision.model_validate(json.loads(raw_output)), context)
        except Exception as exc:
            logger.exception(
                "Agents decide failed | player_id=%s | role=%s | day=%s | phase=%s | error=%s",
                context.player_id,
                context.role,
                context.day,
                context.phase,
                exc,
            )
            return self._fallback_decision(context)

        return self._fallback_decision(context)

    def _finalize_model_decision(self, decision: AgentDecision, context: AIContext) -> AgentDecision:
        """模型输出先清洗，再做出戏硬校验；失败则丢弃话术走兜底。"""
        raw_content = decision.content or ""
        sanitized = self._sanitize_decision(decision)
        if self._content_needs_fallback(raw_content, context) or self._content_needs_fallback(sanitized.content, context):
            fallback = self._fallback_decision(context)
            if sanitized.target_id in context.allowed_target_ids:
                fallback.target_id = sanitized.target_id
            fallback.reason = "模型输出含出戏表达或内部字段，已切换为本地真人化兜底。"
            return fallback
        return sanitized

    def _sanitize_decision(self, decision: AgentDecision) -> AgentDecision:
        """清理明显出戏表达，减少 AI 味。"""
        banned_replacements = {
            "接入": "看",
            "上下文": "前面这些话",
            "结构化信息": "场上信息",
            "结构化上下文": "场上信息",
            "候选列表": "能选的人",
            "target_id": "目标",
            "作为AI": "我",
            "根据提示词": "按我听到的",
            "系统要求": "这轮",
            "系统提示": "场上信息",
            "我先接一下": "我说下",
            "我先接": "我说下",
            "我接一下": "我说下",
            "接一下": "看一下",
        }
        content = decision.content or ""
        content = re.sub(r"```[\s\S]*?```", "", content)
        content = re.sub(r"\{[\s\S]{0,300}?(?:target_id|action|reason)[\s\S]{0,300}?\}", "", content)
        for banned, replacement in banned_replacements.items():
            content = content.replace(banned, replacement)
        content = re.sub(r"我(?:现在)?先(?:看|说下|看一下)(\d{1,2}号)", r"\1", content)
        content = re.sub(r"我(?:现在)?先(?:看|说下|看一下)(前置位|上一位|这段)", r"\1", content)
        content = re.sub(r"\b(JSON|json|markdown|schema|prompt)\b", "", content)
        content = re.sub(r"\s+", " ", content)
        decision.content = content.strip()
        return decision

    def _content_needs_fallback(self, content: str, context: AIContext) -> bool:
        """玩家可见文本不能泄漏提示词/字段/AI痕迹。"""
        if context.phase in {"day_vote", "exile_pk_vote", "sheriff_vote", "sheriff_pk_vote", "night_action"}:
            return False
        if not content.strip():
            return True
        if self._anchors_dead_seat_in_day_content(content, context):
            return True
        if context.phase not in {"wolf_chat"}:
            private_leak_patterns = [
                r"狼队友",
                r"狼人夜聊",
                r"夜聊",
                r"刀口",
                r"今晚(?:先)?刀",
                r"昨晚(?:先)?刀",
                r"夜谈结束",
                r"我是狼人",
                r"过往夜晚复盘",
                r"上夜",
                r"旧夜",
                r"狼队共识",
            ]
            if any(re.search(pattern, content) for pattern in private_leak_patterns):
                return True
        banned_patterns = [
            r"接入",
            r"上下文",
            r"提示词",
            r"系统(?:要求|提示)",
            r"结构化",
            r"候选列表",
            r"target[_ ]?id",
            r"\b(?:JSON|json|schema|prompt|markdown)\b",
            r"作为\s*AI",
            r"我(?:现在)?先(?:接|接一下|接入)",
            r"我(?:现在)?先(?:抓|看|说)\s*\d{1,2}号[^。！？!?]{0,12}(?:这句|刚才|这段)",
            r"\d{1,2}号[^。！？!?]{0,12}(?:这句|刚才那句|这段)[^。！？!?]{0,28}(?:我先|我会|继续盯|没问题)",
            r"只保留策略教训",
            r"不复述旧夜具体刀口",
        ]
        return any(re.search(pattern, content) for pattern in banned_patterns)

    def _anchors_dead_seat_in_day_content(self, content: str, context: AIContext) -> bool:
        """白天可见发言不能继续把已死亡号位当成本轮主目标。"""
        if context.phase not in {"day_speech", "exile_pk_speech", "last_words"} or context.structured is None:
            return False
        alive_seats = {
            player.seat_no
            for player in context.structured.public_players
            if player.alive
        }
        for raw in re.findall(r"(\d{1,2})号", content):
            seat_no = int(raw)
            if seat_no not in alive_seats:
                return True
        return False

    def _fallback_decision(self, context: AIContext) -> AgentDecision:
        """未配置 OpenAI 时的本地兜底逻辑。"""
        scored_targets = self._score_targets(context)
        target_id: Optional[int] = self._select_fallback_target(context, scored_targets)
        if context.role == RoleName.SEER and context.phase == "night_action":
            target_id = self._select_seer_inspection_target(context, scored_targets)

        if context.phase in {"day_speech", "campaign_speech", "pk_campaign_speech", "exile_pk_speech", "last_words"}:
            content = self._build_fallback_speech(context, scored_targets)
            return AgentDecision(
                action="speak",
                target_id=None,
                content=content,
                reason=f"未启用 OpenAI，按 {context.player_id + 1} 号位的公开信息、身份目标与私有上下文生成兜底发言。",
            )
        if context.phase in {"day_vote", "exile_pk_vote", "sheriff_vote", "sheriff_pk_vote"}:
            return AgentDecision(
                action="vote",
                target_id=target_id,
                content="",
                reason=self._fallback_vote_reason(context, target_id),
            )
        if context.phase == "wolf_chat":
            target_id = self._select_wolf_chat_target(context, scored_targets) if scored_targets else None
            return AgentDecision(
                action="night_action",
                target_id=target_id,
                content=self._build_fallback_wolf_chat(context, scored_targets, selected_target_id=target_id),
                reason=f"未启用 OpenAI，依据 {context.player_id + 1} 号位狼人可见局势与刀口收益生成夜聊。",
            )
        if context.role == RoleName.WITCH and context.phase == "night_action":
            target_id, reason = self._fallback_witch_target(context, scored_targets)
            return AgentDecision(
                action="night_action",
                target_id=target_id,
                content="",
                reason=reason,
            )
        return AgentDecision(
            action="night_action",
            target_id=target_id,
            content="",
            reason=f"未启用 OpenAI，依据 {context.player_id + 1} 号位的身份目标、公开信息与私有上下文生成夜间动作。",
        )

    def _score_targets(self, context: AIContext) -> list[int]:
        """给候选目标做一个粗粒度排序，避免本地模式完全随机。"""
        scores = {target_id: 0 for target_id in context.allowed_target_ids}
        lines = context.visible_state.splitlines()
        for line in lines:
            line = line.strip()
            match = re.match(r"(?:player_id=)?(\d+)(?:,\s*seat_no=\d+)?:\s*(玩家\d+)\s*-\s*(存活|死亡)(.*)", line)
            if not match:
                continue
            player_id = int(match.group(1))
            suffix = match.group(4)
            if player_id not in scores:
                continue
            if "警长" in suffix:
                scores[player_id] += 3
            if context.phase in {"day_vote", "night_action", "wolf_chat"}:
                scores[player_id] += 1

        recent_speeches = [line for line in lines if ":" in line and "玩家" in line]
        for text in recent_speeches:
            player_match = re.match(r"(玩家\d+):\s*(.*)", text)
            if not player_match:
                continue
            name, content = player_match.groups()
            id_match = re.search(r"玩家(\d+)", name)
            if not id_match:
                continue
            player_id = int(id_match.group(1)) - 1
            if player_id not in scores:
                continue
            suspicious_tokens = ["站边", "查杀", "警徽", "归票", "身份", "逻辑", "保", "打"]
            if any(token in content for token in suspicious_tokens):
                scores[player_id] += 1
            if len(content) > 24:
                scores[player_id] += 1

        if context.role == RoleName.WEREWOLF:
            for target_id in scores:
                scores[target_id] += 2
                if context.structured:
                    target = next(
                        (item for item in context.structured.public_players if item.player_id == target_id),
                        None,
                    )
                    if target and target.is_sheriff:
                        scores[target_id] += 2
        if context.role == RoleName.SEER and context.phase == "night_action":
            inspected_ids = set(self._seer_inspection_history(context))
            for target_id in scores:
                scores[target_id] += 1
                if target_id in inspected_ids:
                    scores[target_id] -= 12
        if context.phase in {"sheriff_vote", "sheriff_pk_vote", "exile_pk_vote", "hunter_shot"}:
            for target_id in scores:
                scores[target_id] += 2
        if context.role == RoleName.WITCH and context.phase == "night_action":
            for target_id in scores:
                scores[target_id] += 1

        self._score_from_structured_evidence(context, scores)

        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        if ranked:
            return ranked
        if context.structured:
            candidates = [
                player.player_id
                for player in context.structured.public_players
                if player.alive and player.player_id != context.player_id
            ]
            if context.role == RoleName.WEREWOLF:
                teammate_ids = {teammate.player_id for teammate in context.structured.wolf_teammates}
                candidates = [player_id for player_id in candidates if player_id not in teammate_ids]
            return candidates[:4]
        return []

    def _score_from_structured_evidence(self, context: AIContext, scores: dict[int, int]) -> None:
        """根据公开发言、票型和身份宣称调整目标排序。"""
        structured = context.structured
        if structured is None or not scores:
            return

        mentioned_by_table: dict[int, int] = {}
        for speech in structured.recent_public_speeches:
            speaker_id = speech.speaker_id
            speaker_is_candidate = speaker_id in scores
            risky_speaker_tokens = {"悍跳", "站边", "归票", "身份", "查杀", "金水", "保", "打", "票"}
            if speaker_is_candidate and any(token in risky_speaker_tokens for token in speech.stance_keywords):
                scores[speaker_id] += 2
            if speaker_is_candidate and len(speech.mentioned_seat_nos) >= 3:
                scores[speaker_id] += 1
            for seat in speech.mentioned_seat_nos:
                target_id = seat - 1
                if target_id in scores:
                    mentioned_by_table[target_id] = mentioned_by_table.get(target_id, 0) + 1
                    if context.phase in {"day_vote", "exile_pk_vote"}:
                        scores[target_id] += 1

        for target_id, count in mentioned_by_table.items():
            if count >= 2:
                scores[target_id] += 2

        for vote in structured.recent_votes:
            if vote.target_id in scores and context.phase in {"day_vote", "exile_pk_vote"}:
                scores[vote.target_id] += 1

        strong_claimed_wolf_target = self._strong_claimed_wolf_target(context, list(scores))
        for claim in structured.public_claims:
            inspected_target_id = getattr(claim, "inspected_target_id", None)
            inspected_result = getattr(claim, "inspected_result", None)
            if context.role == RoleName.WEREWOLF and context.phase in {"wolf_chat", "night_action"}:
                if claim.speaker_id not in scores:
                    continue
                role_threat = {
                    RoleName.SEER: 8,
                    RoleName.WITCH: 6,
                    RoleName.HUNTER: 3,
                    RoleName.IDIOT: 1,
                    RoleName.VILLAGER: 1,
                }
                scores[claim.speaker_id] += role_threat.get(claim.claimed_role, 0)
            elif context.phase in {"day_vote", "exile_pk_vote"}:
                if claim.claimed_role == RoleName.SEER and inspected_target_id in scores:
                    if inspected_result == "狼人":
                        if context.role == RoleName.WEREWOLF:
                            if claim.speaker_id in scores:
                                scores[claim.speaker_id] += 3
                            if "倒钩" in context.strategy_style or context.player_id % 3 == 0:
                                scores[inspected_target_id] += 5
                        else:
                            scores[inspected_target_id] += 14 if inspected_target_id == strong_claimed_wolf_target else 7
                            if claim.speaker_id in scores:
                                scores[claim.speaker_id] -= 2
                    elif inspected_result == "好人":
                        scores[inspected_target_id] -= 5
                        if context.role == RoleName.WEREWOLF and claim.speaker_id in scores:
                            scores[claim.speaker_id] += 2
                    continue
                if claim.speaker_id in scores:
                    if context.role == RoleName.WEREWOLF:
                        role_pressure = {
                            RoleName.WITCH: 5,
                            RoleName.HUNTER: 3,
                            RoleName.IDIOT: 1,
                            RoleName.VILLAGER: 1,
                        }
                        scores[claim.speaker_id] += role_pressure.get(claim.claimed_role, 2)
                    elif claim.claimed_role in {RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT}:
                        scores[claim.speaker_id] -= 4
                    else:
                        scores[claim.speaker_id] += 1

        chain_target = self._seer_chain_pressure_target(context, list(scores), direct=False)
        if chain_target in scores and context.phase in {"day_vote", "exile_pk_vote", "hunter_shot"}:
            scores[chain_target] += 18 if context.phase == "hunter_shot" else 8

    def _select_fallback_target(self, context: AIContext, scored_targets: list[int]) -> Optional[int]:
        """按阶段选择目标，保留少量桌游式分歧，避免所有兜底 Agent 同票同刀。"""
        if not scored_targets:
            return None
        if context.phase in {"day_vote", "exile_pk_vote", "hunter_shot"}:
            chain_target = self._seer_chain_pressure_target(context, scored_targets, direct=True)
            if chain_target is not None and context.role != RoleName.WEREWOLF:
                return chain_target
        if context.phase in {"day_vote", "exile_pk_vote"} and len(scored_targets) > 1:
            crowded_target = self._crowded_vote_target(context, scored_targets)
            if crowded_target is not None:
                crowd_split_target = self._crowded_vote_split_target(context, crowded_target, scored_targets)
                if crowd_split_target is not None:
                    return crowd_split_target
        if context.phase == "wolf_chat" and len(scored_targets) > 1:
            return self._select_wolf_chat_target(context, scored_targets)
        if context.phase in {"day_vote", "exile_pk_vote"} and len(scored_targets) > 1:
            claimed_wolf_target = self._claimed_wolf_target(context, scored_targets)
            if claimed_wolf_target is not None:
                crowd_split_target = self._crowded_vote_split_target(context, claimed_wolf_target, scored_targets)
                if crowd_split_target is not None:
                    return crowd_split_target
                if context.role == RoleName.WEREWOLF:
                    counter_target = self._wolf_counter_seer_vote_target(context, claimed_wolf_target, scored_targets)
                    if counter_target is not None:
                        return counter_target
                    if "倒钩" in context.strategy_style or context.player_id % 4 == 0:
                        return claimed_wolf_target
                    return scored_targets[0] if scored_targets[0] != claimed_wolf_target else scored_targets[min(1, len(scored_targets) - 1)]
                if self._strong_claimed_wolf_target(context, scored_targets) == claimed_wolf_target and self._direct_claimed_wolf_vote_allowed(
                    context, claimed_wolf_target
                ):
                    return claimed_wolf_target
                counterclaim_target = self._counterclaim_vote_split_target(context, claimed_wolf_target, scored_targets)
                if counterclaim_target is not None:
                    return counterclaim_target
                claim_split_target = self._claimed_wolf_vote_split_target(context, claimed_wolf_target, scored_targets)
                if claim_split_target is not None:
                    return claim_split_target
                return claimed_wolf_target
            persona = context.persona_style or ""
            wolf_role_claim_target = self._wolf_public_power_claim_target(context, scored_targets)
            if wolf_role_claim_target is not None:
                return wolf_role_claim_target
            crowd_target = self._crowded_vote_target(context, scored_targets)
            if crowd_target == scored_targets[0]:
                offset = 1 + ((context.player_id + context.day) % min(2, len(scored_targets) - 1))
                return scored_targets[min(offset, len(scored_targets) - 1)]
            top_claim = None
            if context.structured:
                top_claim = next(
                    (claim for claim in reversed(context.structured.public_claims) if claim.speaker_id == scored_targets[0]),
                    None,
                )
            top_claimed = top_claim is not None
            if top_claimed:
                if context.role == RoleName.WEREWOLF and context.player_id % 3 != 0:
                    return scored_targets[0]
                if context.role != RoleName.WEREWOLF and top_claim.claimed_role == RoleName.SEER and context.day <= 2:
                    return scored_targets[min(1, len(scored_targets) - 1)]
                if ("铁腕" in persona or "冲锋" in context.strategy_style or "强预" in context.strategy_style) and context.player_id % 2 == 0:
                    return scored_targets[0]
                if context.structured and any(speech.speaker_id == scored_targets[0] for speech in context.structured.recent_public_speeches[-4:]):
                    return scored_targets[0]
                return scored_targets[min(1 + (context.player_id % min(2, len(scored_targets) - 1)), len(scored_targets) - 1)]
            if context.role == RoleName.WEREWOLF:
                if len(scored_targets) >= 3 and context.player_id % 2 == 0:
                    return scored_targets[2]
                return scored_targets[min(1, len(scored_targets) - 1)]
            if "谨慎" in persona or "圆滑" in persona:
                return scored_targets[min(1, len(scored_targets) - 1)]
            if "反骨" in persona or "赌徒" in persona:
                return scored_targets[(context.player_id + 1) % min(3, len(scored_targets))]
            if context.player_id % 3 == 1:
                return scored_targets[min(1, len(scored_targets) - 1)]
            if context.player_id % 5 == 0 and len(scored_targets) >= 3:
                return scored_targets[2]
            if context.player_id % 4 == 0:
                return scored_targets[1]
        return scored_targets[0]

    def _wolf_public_power_claim_target(self, context: AIContext, scored_targets: list[int]) -> int | None:
        """狼人白天可以主动压公开神职宣称，制造逼身份/抗推收益。"""
        if context.structured is None or context.role != RoleName.WEREWOLF:
            return None
        for claim in reversed(context.structured.public_claims):
            if claim.speaker_id not in scored_targets:
                continue
            if claim.claimed_role in {RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT}:
                return claim.speaker_id
        return None

    def _wolf_counter_seer_vote_target(
        self,
        context: AIContext,
        claimed_wolf_target: int,
        scored_targets: list[int],
    ) -> int | None:
        """狼队友对跳后，其他狼人优先制造反票，而不是机械跟真预出队友。"""
        if context.structured is None or context.role != RoleName.WEREWOLF:
            return None
        teammate_ids = {teammate.player_id for teammate in context.structured.wolf_teammates}
        if claimed_wolf_target not in teammate_ids:
            return None
        seer_claims = [claim for claim in context.structured.public_claims if claim.claimed_role == RoleName.SEER]
        teammate_claims = [claim for claim in seer_claims if claim.speaker_id in teammate_ids]
        if not teammate_claims:
            return None
        pressure_claim = next(
            (
                claim
                for claim in reversed(seer_claims)
                if claim.inspected_result == "狼人" and claim.inspected_target_id == claimed_wolf_target
            ),
            None,
        )
        if pressure_claim is None:
            return None
        if pressure_claim.speaker_id in scored_targets:
            return pressure_claim.speaker_id
        teammate_latest = teammate_claims[-1]
        inspected_target_id = getattr(teammate_latest, "inspected_target_id", None)
        if inspected_target_id in scored_targets and inspected_target_id not in teammate_ids:
            return inspected_target_id
        for target_id in scored_targets:
            if target_id not in teammate_ids and target_id != claimed_wolf_target:
                return target_id
        return None

    def _crowded_vote_split_target(
        self,
        context: AIContext,
        anchor_target: int,
        scored_targets: list[int],
    ) -> int | None:
        """公开宣称票口过度拥挤时拆出自然分歧，但不绕过规则级硬锚点。"""
        if context.structured is None or context.phase not in {"day_vote", "exile_pk_vote"}:
            return None
        if len(scored_targets) < 2:
            return None
        if self._direct_claimed_wolf_vote_allowed(context, anchor_target):
            return None
        crowded_target = self._crowded_vote_target(context, scored_targets)
        if crowded_target != anchor_target:
            return None
        if context.role == RoleName.SEER and self._own_seer_claimed_target(context, anchor_target):
            return None

        avoid_ids = {anchor_target}
        if context.role == RoleName.WEREWOLF and context.structured:
            avoid_ids.update(teammate.player_id for teammate in context.structured.wolf_teammates)
            avoid_ids.add(context.player_id)
        lane = (context.player_id + context.day) % max(1, len(scored_targets))
        alternatives = [target_id for target_id in scored_targets if target_id not in avoid_ids]
        if not alternatives:
            alternatives = [target_id for target_id in scored_targets if target_id != anchor_target]
        if not alternatives:
            return None
        return alternatives[lane % len(alternatives)]

    def _own_seer_claimed_target(self, context: AIContext, target_id: int) -> bool:
        """真预自己的查杀口不参与拥挤分票。"""
        if context.structured is None:
            return False
        return any(
            claim.speaker_id == context.player_id
            and claim.claimed_role == RoleName.SEER
            and claim.inspected_result == "狼人"
            and claim.inspected_target_id == target_id
            for claim in context.structured.public_claims
        )

    def _counterclaim_vote_split_target(
        self,
        context: AIContext,
        claimed_wolf_target: int,
        scored_targets: list[int],
    ) -> int | None:
        """多预言家对跳时，非预言家阵营保留自然分歧，避免全场机械冲同一张。"""
        if context.structured is None or context.role == RoleName.SEER:
            return None
        seer_claims = [claim for claim in context.structured.public_claims if claim.claimed_role == RoleName.SEER]
        if len(seer_claims) < 2:
            return None
        if self._strong_claimed_wolf_target(context, scored_targets) == claimed_wolf_target and self._direct_claimed_wolf_vote_allowed(
            context, claimed_wolf_target
        ):
            return None
        latest = seer_claims[-1]
        prior_claimer_ids = {claim.speaker_id for claim in seer_claims[:-1]}
        if len(seer_claims) == 2 and latest.inspected_target_id in prior_claimer_ids:
            return None
        if latest.speaker_id == context.player_id:
            return None
        lane = (context.player_id + context.day) % 4
        if lane == 0 and latest.speaker_id in scored_targets:
            return latest.speaker_id
        if lane == 1 and latest.inspected_target_id in scored_targets:
            return latest.inspected_target_id
        if lane == 2:
            for target_id in scored_targets:
                if target_id not in {claimed_wolf_target, latest.speaker_id, latest.inspected_target_id}:
                    return target_id
        return None

    def _claimed_wolf_vote_split_target(
        self,
        context: AIContext,
        claimed_wolf_target: int,
        scored_targets: list[int],
    ) -> int | None:
        """单预查杀局保留少量自然分歧，避免全场机械跟查杀。"""
        if context.structured is None or context.role in {RoleName.SEER, RoleName.WITCH, RoleName.HUNTER}:
            return None
        if self._strong_claimed_wolf_target(context, scored_targets) == claimed_wolf_target and self._direct_claimed_wolf_vote_allowed(
            context, claimed_wolf_target
        ):
            return None
        latest_claim = next(
            (
                claim
                for claim in reversed(context.structured.public_claims)
                if claim.claimed_role == RoleName.SEER
                and claim.inspected_result == "狼人"
                and claim.inspected_target_id == claimed_wolf_target
            ),
            None,
        )
        if latest_claim is None:
            return None
        if context.day <= 1 and context.role == RoleName.IDIOT:
            return None
        lane = (context.player_id + context.day) % 5
        if lane == 0 and latest_claim.speaker_id in scored_targets:
            return latest_claim.speaker_id
        if lane in {1, 2}:
            for target_id in scored_targets:
                if target_id not in {claimed_wolf_target, latest_claim.speaker_id}:
                    return target_id
        return None

    def _select_wolf_chat_target(self, context: AIContext, scored_targets: list[int]) -> int:
        """狼队夜聊目标选择：有公开身份威胁才优先刀，否则首夜盲刀要分散。"""
        if context.structured:
            threat_claim = next(
                (
                    claim.speaker_id
                    for claim in reversed(context.structured.public_claims)
                    if claim.speaker_id in scored_targets and claim.claimed_role in {RoleName.SEER, RoleName.WITCH}
                ),
                None,
            )
            round_id = self._current_wolf_chat_round(context)
            current_round_records = [
                record
                for record in context.structured.wolf_chat_records
                if record.round_id == round_id and record.proposed_target_id in scored_targets
            ]
            if current_round_records:
                if (
                    threat_claim is None
                    and round_id == 1
                    and len(current_round_records) <= 2
                    and len({record.proposed_target_id for record in current_round_records}) == 1
                    and len(scored_targets) > 1
                ):
                    return scored_targets[1] if scored_targets[0] == current_round_records[0].proposed_target_id else scored_targets[0]
                if threat_claim is not None and any(record.proposed_target_id == threat_claim for record in current_round_records):
                    return threat_claim
                return current_round_records[-1].proposed_target_id
            if threat_claim is not None:
                return threat_claim
            if context.structured.recent_public_speeches:
                return scored_targets[0]
        offset = (context.player_id + context.day + len(scored_targets) // 2) % len(scored_targets)
        return scored_targets[offset]

    def _current_wolf_chat_round(self, context: AIContext) -> int:
        """从阶段提示中读取当前狼聊轮次，读取不到则按 1 处理。"""
        round_match = re.search(r"第(\d+)轮", context.prompt)
        return int(round_match.group(1)) if round_match else 1

    def _claimed_wolf_target(self, context: AIContext, scored_targets: list[int]) -> int | None:
        """公开预言家查杀是白天投票强锚点。"""
        if context.structured is None:
            return None
        if context.role == RoleName.WEREWOLF:
            teammate_ids = {teammate.player_id for teammate in context.structured.wolf_teammates}
            teammate_ids.add(context.player_id)
            claims = list(reversed(context.structured.public_claims))
            for claim in claims:
                inspected_target_id = getattr(claim, "inspected_target_id", None)
                inspected_result = getattr(claim, "inspected_result", None)
                if (
                    claim.claimed_role == RoleName.SEER
                    and claim.speaker_id not in teammate_ids
                    and inspected_result == "狼人"
                    and inspected_target_id in scored_targets
                ):
                    return inspected_target_id
            return None

        strong_target = self._strong_claimed_wolf_target(context, scored_targets)
        if strong_target is not None:
            return strong_target

        prior_seer_claimers: set[int] = set()
        ranked_claims: list[tuple[int, int, int]] = []
        for index, claim in enumerate(context.structured.public_claims):
            inspected_target_id = getattr(claim, "inspected_target_id", None)
            inspected_result = getattr(claim, "inspected_result", None)
            if claim.claimed_role == RoleName.SEER and inspected_result == "狼人" and inspected_target_id in scored_targets:
                credibility = 20 - index
                if inspected_target_id in prior_seer_claimers:
                    credibility -= 12
                if claim.speaker_id in prior_seer_claimers:
                    credibility += 3
                ranked_claims.append((credibility, -index, inspected_target_id))
            if claim.claimed_role == RoleName.SEER:
                prior_seer_claimers.add(claim.speaker_id)
        if not ranked_claims:
            return None
        ranked_claims.sort(reverse=True)
        if ranked_claims[0][0] <= 0:
            return None
        return ranked_claims[0][2]

    def _strong_claimed_wolf_target(self, context: AIContext, scored_targets: list[int]) -> int | None:
        """识别可信查杀锚点：连续验人链优先，不让后跳反打真预轻易带走票流。"""
        if context.structured is None:
            return None
        seer_claims = [claim for claim in context.structured.public_claims if claim.claimed_role == RoleName.SEER]
        if not seer_claims:
            return None
        prior_seer_claimers: set[int] = set()
        claim_count_by_speaker: dict[int, int] = {}
        best: tuple[int, int] | None = None
        for index, claim in enumerate(seer_claims):
            inspected_target_id = getattr(claim, "inspected_target_id", None)
            inspected_result = getattr(claim, "inspected_result", None)
            previous_claims = claim_count_by_speaker.get(claim.speaker_id, 0)
            if inspected_result == "狼人" and inspected_target_id in scored_targets:
                strength = 0
                if previous_claims:
                    strength += 18
                if claim.speaker_id in prior_seer_claimers:
                    strength += 8
                if inspected_target_id in prior_seer_claimers and claim.speaker_id not in prior_seer_claimers:
                    strength -= 20
                if index == len(seer_claims) - 1:
                    strength += 2
                if strength >= 16 and (best is None or strength > best[0]):
                    best = (strength, inspected_target_id)
            prior_seer_claimers.add(claim.speaker_id)
            claim_count_by_speaker[claim.speaker_id] = previous_claims + 1
        return best[1] if best else None

    def _direct_claimed_wolf_vote_allowed(self, context: AIContext, claimed_wolf_target: int) -> bool:
        """查杀宣称可做强锚点，但只有有额外可信条件时才允许全场直接跟票。"""
        if context.structured is None:
            return False
        alive_ids = {player.player_id for player in context.structured.public_players if player.alive}
        if len(alive_ids) <= 6:
            return True
        latest_claim = next(
            (
                claim
                for claim in reversed(context.structured.public_claims)
                if claim.claimed_role == RoleName.SEER
                and claim.inspected_result == "狼人"
                and claim.inspected_target_id == claimed_wolf_target
            ),
            None,
        )
        if latest_claim is None:
            return False
        if latest_claim.speaker_id not in alive_ids:
            return True
        return any(
            claim.speaker_id == claimed_wolf_target
            and claim.claimed_role == RoleName.SEER
            and claim.inspected_target_id == latest_claim.speaker_id
            and claim.inspected_result == "狼人"
            for claim in context.structured.public_claims
        )

    def _seer_chain_pressure_target(
        self,
        context: AIContext,
        scored_targets: list[int],
        *,
        direct: bool = False,
    ) -> int | None:
        """公共验人链只在足够可信时强压目标，避免把发言宣称当规则事实。"""
        if context.structured is None:
            return None
        claims_by_speaker: dict[int, list] = {}
        for claim in context.structured.public_claims:
            if claim.claimed_role != RoleName.SEER:
                continue
            if claim.inspected_result != "狼人" or claim.inspected_target_id not in scored_targets:
                continue
            claims_by_speaker.setdefault(claim.speaker_id, []).append(claim)
        best: tuple[int, int] | None = None
        alive_ids = {player.player_id for player in context.structured.public_players if player.alive}
        for speaker_id, claims in claims_by_speaker.items():
            unique_targets = []
            for claim in claims:
                if claim.inspected_target_id not in unique_targets:
                    unique_targets.append(claim.inspected_target_id)
            if len(unique_targets) < 2:
                continue
            speaker_dead = speaker_id not in alive_ids
            endgame = len(alive_ids) <= 6
            if direct and context.phase != "hunter_shot" and not (speaker_dead or endgame):
                continue
            speaker_dead_bonus = 6 if speaker_id not in alive_ids else 0
            for recency, target_id in enumerate(unique_targets):
                if direct and not (speaker_dead or endgame or context.phase == "hunter_shot"):
                    continue
                strength = len(unique_targets) * 8 + speaker_dead_bonus + recency
                if best is None or strength > best[0]:
                    best = (strength, target_id)
        return best[1] if best else None

    def _crowded_vote_target(self, context: AIContext, scored_targets: list[int]) -> int | None:
        """当前白天已明显集中时，fallback 适度分票，避免所有 AI 机械归同一人。"""
        if context.structured is None:
            return None
        current_round = f"day_{context.day}_exile" if context.phase == "day_vote" else f"day_{context.day}_pk_exile"
        votes = [vote for vote in context.structured.recent_votes if vote.vote_round == current_round]
        if len(votes) < 3:
            return None
        counts: dict[int, int] = {}
        for vote in votes:
            if vote.target_id in scored_targets:
                counts[vote.target_id] = counts.get(vote.target_id, 0) + 1
        if not counts:
            return None
        target_id, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        if count / len(votes) >= 0.6:
            return target_id
        return None

    def _fallback_vote_reason(self, context: AIContext, target_id: int | None) -> str:
        """生成可审计的 fallback 投票依据。"""
        seat = f"{target_id + 1}号" if target_id is not None else "空目标"
        if context.structured and target_id is not None:
            for claim in reversed(context.structured.public_claims):
                if claim.speaker_id == target_id:
                    return f"未启用 OpenAI，投{seat}：该位置公开声称{claim.claimed_role.value}，需要用投票压力验证其发言链路。"
            mentions = [
                speech.speaker_seat_no
                for speech in context.structured.recent_public_speeches
                if target_id + 1 in speech.mentioned_seat_nos
            ]
            if mentions:
                return f"未启用 OpenAI，投{seat}：该位置被{len(mentions)}条公开发言点到，当前轮次最需要进票型检验。"
        return f"未启用 OpenAI，依据 {context.player_id + 1} 号位的公开信息与身份目标选择投{seat}。"

    def _fallback_witch_target(self, context: AIContext, scored_targets: list[int]) -> tuple[int | None, str]:
        """女巫 fallback 不能永远跳过，否则狼队无成本稳定滚雪球。"""
        witch_info = context.structured.witch_night_info if context.structured else None
        if witch_info:
            wolf_target_id = witch_info.wolf_target_id
            save_available = witch_info.save_available and witch_info.can_save_target
            poison_available = witch_info.poison_available
        else:
            visible_text = "\n".join([context.visible_state, context.prompt])
            wolf_target_id = self._parse_wolf_target_id(visible_text)
            save_available = "解药可用" in visible_text or "解药还在" in visible_text
            poison_available = "毒药可用" in visible_text or "毒药还在" in visible_text

        if save_available and wolf_target_id in context.allowed_target_ids:
            should_save = False
            if wolf_target_id == context.player_id and context.day == 1:
                should_save = True
            if context.structured:
                for claim in context.structured.public_claims[-6:]:
                    if claim.speaker_id == wolf_target_id and claim.claimed_role in {RoleName.SEER, RoleName.WITCH, RoleName.HUNTER}:
                        should_save = True
            if should_save:
                return wolf_target_id, f"未启用 OpenAI，女巫依据刀口价值选择救{wolf_target_id + 1}号，保留白天信息位和轮次。"

        claimed_wolf_target = self._claimed_wolf_target(context, scored_targets)
        if poison_available and context.day >= 2 and claimed_wolf_target in context.allowed_target_ids:
            return claimed_wolf_target, f"未启用 OpenAI，女巫依据公开查杀选择毒{claimed_wolf_target + 1}号。"

        if poison_available and context.day >= 3 and scored_targets:
            for target_id in scored_targets:
                if target_id in context.allowed_target_ids and target_id != context.player_id:
                    return target_id, f"未启用 OpenAI，女巫在中后期依据公开疑点选择毒{target_id + 1}号。"

        return None, f"未启用 OpenAI，{context.player_id + 1}号位女巫本夜信息不足，保留药瓶。"

    def _select_seer_inspection_target(self, context: AIContext, scored_targets: list[int]) -> int | None:
        """预言家夜间验人优先覆盖未验争议位，不重复浪费验人。"""
        inspected_ids = set(self._seer_inspection_history(context))
        uninspected = [target_id for target_id in scored_targets if target_id not in inspected_ids]
        if not uninspected:
            return scored_targets[0] if scored_targets else None
        if context.structured:
            for claim in reversed(context.structured.public_claims):
                if claim.speaker_id in uninspected:
                    return claim.speaker_id
                inspected_target_id = getattr(claim, "inspected_target_id", None)
                if inspected_target_id in uninspected:
                    return inspected_target_id
            mentioned_counts: dict[int, int] = {}
            for speech in context.structured.recent_public_speeches[-8:]:
                for seat_no in speech.mentioned_seat_nos:
                    player_id = seat_no - 1
                    if player_id in uninspected:
                        mentioned_counts[player_id] = mentioned_counts.get(player_id, 0) + 1
            if mentioned_counts:
                return max(mentioned_counts, key=lambda player_id: (mentioned_counts[player_id], -uninspected.index(player_id)))
        return uninspected[0]

    def _parse_wolf_target_id(self, text: str) -> int | None:
        """从女巫夜间可见信息中提取规则引擎给出的狼刀目标。"""
        patterns = [
            r"狼人目标是\s*player_id=(\d+)",
            r"今晚狼人刀口是\s*player_id=(\d+)",
            r"今晚狼人刀口是\s*玩家(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = int(match.group(1))
            if "玩家" in pattern:
                return value - 1
            return value
        return None

    def _build_fallback_speech(self, context: AIContext, scored_targets: list[int]) -> str:
        """给本地模式一个更像真人局的发言。"""
        hot = f"{(scored_targets[0] + 1)}号" if scored_targets else "前置位"
        second = f"{(scored_targets[1] + 1)}号" if len(scored_targets) > 1 else hot
        sharp = "你这发言太飘了" if "情绪" in context.persona_style or "铁腕" in context.persona_style else "这轮逻辑不顺"
        speech_count = len(context.structured.recent_public_speeches) if context.structured else 0
        if context.phase == "last_words":
            return self._fallback_last_words(context, scored_targets)
        if context.phase == "exile_pk_speech":
            return self._fallback_exile_pk_speech(context, scored_targets)
        seer_chain = self._seer_chain_speech(context)
        if seer_chain:
            return seer_chain
        seer_result = self._latest_seer_result(context)
        if context.role == RoleName.SEER and seer_result:
            target_seat, result = seer_result
            if result == "狼人":
                variants = [
                    f"我今天直接点{target_seat}号进核心轮次。昨晚我验到这里是狼人，所以这轮别散票，先看谁还在替{target_seat}号缓冲。",
                    f"{target_seat}号我不会放。我的夜里信息指向这里是狼人，今天重点不是我像不像预言家，是谁在给{target_seat}号找退路。",
                ]
                return variants[context.player_id % len(variants)]
            variants = [
                f"{target_seat}号我先放一轮，这张是我夜里拿到的好人信息。今天我更想从打{target_seat}号最急的人里找狼坑。",
                f"我报清楚，{target_seat}号偏好人信息在我这里成立。今天别把票浪费在{target_seat}号身上，先盘谁借这个位置做抗推。",
            ]
            return variants[context.player_id % len(variants)]
        fake_claim = self._fallback_wolf_fake_seer_claim(context, scored_targets)
        if fake_claim:
            return fake_claim
        claim_response = self._fallback_claim_response(context, scored_targets)
        if claim_response:
            return claim_response
        evidence = self._pick_public_speech_evidence(context, scored_targets)
        if speech_count <= 1 or evidence is None:
            return self._fallback_opening_speech(context, scored_targets)
        if evidence:
            hot = f"{evidence.speaker_seat_no}号"
            mentioned = self._preferred_alive_mentioned_seat(context, evidence.mentioned_seat_nos, second)
            paraphrase = self._paraphrase_evidence(evidence.content, hot, mentioned)
            angle = self._speech_attack_angle(context, evidence, mentioned)
            if context.role == RoleName.WEREWOLF:
                paraphrase_pressure = (
                    f"{paraphrase}，落点还是不够干净，像是在给后置位递刀"
                    if "但" not in paraphrase
                    else f"{paraphrase}；这不像单纯找狼，更像在给后置位递刀"
                )
                variants = [
                    f"{hot}我暂时不摘。{angle}，这类牌白天很容易把票带歪；{mentioned}先放桌上听回合，不急着替谁定身份。",
                    f"我这轮看{hot}多一点。{paraphrase_pressure}。今天谁顺着这个方向冲，我会一起记票型和站边。",
                    f"{hot}这个位置可以进票池。不是因为他说错一句，而是{angle}；{mentioned}如果被带成公共靶，我反而要回头看带节奏的人。",
                    f"我不想让场子只围着{mentioned}转。{hot}这段更像在铺路，先给态度再等别人补逻辑，这种发言比单纯站错边更危险。",
                ]
                return variants[context.player_id % len(variants)]
            if context.role == RoleName.SEER:
                variants = [
                    f"{hot}这轮我不放过。{angle}，这个位置如果不解释清楚，我晚上会优先考虑验这里。",
                    f"{hot}对{mentioned}的处理太快了。真找狼应该先补前因后果，不是先抢定义；后面谁替{hot}圆，我一起看。",
                    f"我会把{hot}放进验人视野。你现在不是单点发言问题，而是一直在试探场上谁愿意跟。",
                ]
                return variants[context.player_id % len(variants)]
            variants = [
                f"{hot}这里有个具体问题：{angle}。今天我先听你补一段心路，不补的话，我投票会往你这边靠。",
                f"{mentioned}我不急着定死。真正让我不舒服的是{hot}的切入方式，像在先制造靶子，再等别人补票。",
                f"{hot}这张我先挂听。你这轮有身份信息量，但现在只够形成疑点，不够直接归票。",
                f"我这轮看两个人：{hot}和{mentioned}。前者负责起势，后者被推到台面，谁急着把这个关系讲死谁更像狼。",
            ]
            return variants[context.player_id % len(variants)]
        if context.role == RoleName.WEREWOLF:
            lines = [
                f"{hot}这轮要进视野，不是因为一句话，而是他一直在抢安全定义。{second}如果继续顺这个票口，我会把两张一起记。",
                f"{hot}最大的问题不是凶，是给结论时很满，轮到解释收益又开始往回收，这不像真有视角，更像怕后面圆不回来。",
                f"先别急着把{hot}放掉。这个位置看着在找狼，其实每次落点都卡在安全边，我白天会把压力往这里推。",
            ]
            return lines[(context.player_id + context.day) % len(lines)]
        if context.role == RoleName.SEER:
            lines = [
                f"{hot}得重点看，他的站边和票意不在一条线上，像是在跟场面找安全位。{second}如果替他补逻辑，我会一起盘联动。",
                f"{hot}这轮看着完整，其实关键矛盾没接，只是在把公共结论换个说法。这不像有视角，像顺势做结论。",
                f"我不会因为一句态度打死{hot}，但他该解释的改口没有解释；后面再不把心路摊开，我会往验人视野里放。",
            ]
            return lines[(context.player_id + context.day) % len(lines)]
        if context.role in {RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER, RoleName.GUARD}:
            lines = [
                f"{hot}这里我放不下，不是没逻辑，而是一直借公共点推进，自己没新增信息。这种发言我很难当成真找狼。",
                f"{hot}这轮让我不舒服的点很具体：先把人打得很死，被反压后又说别急着定，这就是在留后路。",
                f"今天如果真要出票，我会看{hot}，再看{second}。一个像在抢节奏，一个像在顺着公共结论补刀，都不像正经好人找人。",
            ]
            return lines[(context.player_id + context.day) % len(lines)]
        return f"{hot}的发言和轮次理解明显不顺，后面票型再脏一点我就直接打死。"

    def _fallback_exile_pk_speech(self, context: AIContext, scored_targets: list[int]) -> str:
        """放逐 PK 必须直接辩护和打对手，不能复读白天发言模板。"""
        opponents = context.allowed_target_ids[:]
        opponent_id = opponents[0] if opponents else (scored_targets[0] if scored_targets else None)
        opponent = f"{opponent_id + 1}号" if opponent_id is not None else "对跳位"
        self_seat = f"{context.player_id + 1}号"
        lane = (context.player_id + context.day) % 4
        if context.role == RoleName.WEREWOLF:
            variants = [
                f"PK我直接打{opponent}。我今天被推上来，是因为有人想把票口从真正的发言漏洞上挪开；{opponent}一直只顺公共结论，不敢讲自己票从哪来。",
                f"这轮 PK 我不认自己该出。{opponent}的问题是白天一直借别人观点补刀，真到自己解释时没有新信息，我更像被拿来挡票的。",
                f"我在 PK 里只说一个点：如果我是狼，不会把票型压力留得这么明显。{opponent}才是在关键轮次把脏票往外推的人。",
                f"今天出我收益很低，出{opponent}才能验票型。{opponent}前面发言一直留退路，像等别人先冲完再选安全边。",
            ]
            return variants[lane]
        if context.role == RoleName.SEER:
            variants = [
                f"PK我不能出。我是信息位视角，今天把我推出去会断验人链；{opponent}的问题是一直绕开我的验人，只拿态度压我。",
                f"我在 PK 里把话说死：先别出{self_seat}。{opponent}如果是好人，应该盘我的验人真假，而不是只借票口把我打掉。",
                f"这轮 PK 出我就是帮狼断信息。{opponent}没解释清楚自己为什么能无视验人链，这一点比我的发言争议更重。",
                f"我留在场上还能继续给信息，{opponent}只能给态度。今天真要二选一，我会要求先处理{opponent}的站边动机。",
            ]
            return variants[lane]
        if context.role in {RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT}:
            variants = [
                f"PK我先保自己。{opponent}把我推上来太急，像是在逼神职边提前交身份；我不拍身份，但今天出我轮次一定亏。",
                f"我不该在这个 PK 位出局。{opponent}的问题不是打我，而是他打我的理由一直换，像根据票口在改说法。",
                f"这轮二选一，我会反打{opponent}。我这里至少有自己的票型判断，{opponent}更多是在顺势把我做成抗推。",
                f"PK阶段我不再绕。出我只能得到一个模糊身份，出{opponent}能看清谁在白天补刀，这才是今天的收益点。",
            ]
            return variants[lane]
        variants = [
            f"PK我先讲清楚，我不是没有观点的抗推位。{opponent}白天一直顺着公共压力走，真正到落票理由时反而最虚。",
            f"这轮别因为我好推就出我。{opponent}的问题更具体：他没有自己的第一视角，只是在别人打完以后补一脚。",
            f"我在 PK 里反打{opponent}。我前面至少给了判断，{opponent}一直在等场上风向，谁安全就站谁。",
            f"今天如果二选一，我希望票先进{opponent}。出我只是顺手抗推，出他能看清前面那些补票的人到底站哪边。",
        ]
        return variants[lane]

    def _fallback_last_words(self, context: AIContext, scored_targets: list[int]) -> str:
        """遗言要收束信息或误导，不复用普通白天模板。"""
        hot_id = scored_targets[0] if scored_targets else None
        hot = f"{hot_id + 1}号" if hot_id is not None else "最急着补票的人"
        second = f"{scored_targets[1] + 1}号" if len(scored_targets) > 1 else "后面跟票的人"
        lane = (context.player_id + context.day) % 4
        if context.role == RoleName.SEER:
            history = self._seer_result_history(context)
            if history:
                chain = "，".join(f"{seat}号{result}" for seat, result in history[-3:])
                wolf_checks = [seat for seat, result in history if result == "狼人"]
                target = f"{wolf_checks[-1]}号" if wolf_checks else hot
                variants = [
                    f"遗言只留信息：我的验人是{chain}。今天不要把票拆散，先处理{target}，再回头看谁一直在替他缓冲。",
                    f"我倒牌以后别再盘我像不像了，验人链是{chain}。票优先压{target}，谁改口保他，明天直接进狼坑。",
                    f"最后交代清楚：{chain}。好人别被情绪票带跑，先验票型，{target}这条线不能放。",
                    f"我死了信息也在桌上：{chain}。如果今天不出{target}，至少把给他找退路的人记住。",
                ]
                return variants[lane]
        if context.role == RoleName.WEREWOLF:
            variants = [
                f"我遗言不求你们立刻认我，只留一句：今天票型里最舒服的是{hot}，别被预言家线一压就全场同票。",
                f"我出局可以，但回头看{hot}和{second}的票。他们一直在等别人先冲，自己只补安全刀。",
                f"这轮如果我是抗推，明天先验{hot}的票型收益。谁今天把我推出去又不敢给第二狼坑，谁最脏。",
                f"我最后只点{hot}。他不是发言最差，是全程最会借公共结论藏自己票意。",
            ]
            return variants[lane]
        if context.role == RoleName.WITCH:
            variants = [
                f"遗言我不拍更多身份，只说轮次：今天别被单线站边绑死，{hot}的票意和发言对不上，明天优先复盘他。",
                f"我走了以后重点看{hot}，他一直在借别人结论推进票口，这种位置不像真好人找狼。",
                f"最后留一个判断：{hot}和{second}里至少要开一张，尤其谁今天补票最晚，谁收益最大。",
                f"别把我这张牌当纯抗推结案。{hot}的站边变化要回看，他的票比他的发言更脏。",
            ]
            return variants[lane]
        if context.role == RoleName.HUNTER:
            variants = [
                f"我这张牌出局不代表枪口没价值。先记{hot}，他一直在躲自己的最终票意，后面别轻易放。",
                f"遗言只留枪口方向：{hot}优先级最高，{second}看跟票。别让他们用我出局把票型洗干净。",
                f"我不撒地图炮，就点{hot}。他今天像是在等场面定完再补一脚，这个收益不该被放过。",
                f"我走以后先盘谁最想让我闭嘴。{hot}这张牌的发言和落票断开，后面必须解释。",
            ]
            return variants[lane]
        if context.role == RoleName.IDIOT:
            variants = [
                f"我遗言不装信息，只留票型：{hot}今天推我太顺，{second}跟得太舒服，这两张别一起放。",
                f"如果我被当抗推位走，明天先看谁从我身上拿收益。{hot}最像借我做局的人。",
                f"别因为我话不重就把这轮翻篇。{hot}一直在把公共怀疑收成票，这个动作要记。",
                f"最后我认下自己视角有限，但{hot}的推进方式不自然，明天先让他解释票从哪来。",
            ]
            return variants[lane]
        variants = [
            f"我普通身份遗言很简单：别只看谁声音大，先看{hot}怎么落票，他这轮收益最大。",
            f"我走可以，但{hot}不能放。他一直在别人定调以后补结论，像等安全边。",
            f"最后留两个位置：{hot}主看，{second}次看。尤其今天谁补我票，明天先盘谁。",
            f"我没有夜里信息，只能留发言和票型。{hot}这张牌如果继续没人打，狼队太舒服。",
        ]
        return variants[lane]

    def _fallback_opening_speech(self, context: AIContext, scored_targets: list[int]) -> str:
        """证据不足时给身份驱动的自然发言，避免硬接上一位。"""
        hot = f"{(scored_targets[0] + 1)}号" if scored_targets else "后置位"
        second = f"{(scored_targets[1] + 1)}号" if len(scored_targets) > 1 else hot
        lane = (context.player_id + context.day) % 4
        early_round = self._early_round_phrase(context)
        if context.role == RoleName.WEREWOLF:
            variants = [
                f"我这轮不抢着打死谁，先把关注放在{hot}和{second}。谁急着把{early_round}空信息聊成定论，我白天就顺着那条线往下压。",
                f"前面信息还薄，我先给一个观察标准：别只看谁声音大，要看谁最早把票口收窄。{hot}这个位置我会重点听后续解释。",
                f"我今天会装不急，但票不会散。{hot}如果继续只给态度不给原因，就很适合进今天票池；{second}先留作对照。",
                f"{early_round}别被情绪带着跑。我先盯收口位，尤其是{hot}这种容易被场子认成带队身份的人，后面他说不出具体链路我就打。",
            ]
            return variants[lane]

        if context.role == RoleName.SEER:
            variants = [
                f"我现在先不给空结论。今天重点听{hot}和{second}怎么处理站边，谁把话说满又不给心路，我晚上会优先考虑验那里。",
                f"{self._round_observation_phrase(context)}我更看重发言顺序和改口点。{hot}先放进观察位，不是定狼，是看他后面有没有真实判断。",
                f"我会把票和验人分开看：票上先压发言变形位，夜里再处理最像信息核心的位置。{hot}这张牌后面必须给我具体观点。",
                f"现在没有票型，我不会硬归。后置位如果只重复前面结论、不自己落点，我会比单纯说错话更重地看。",
            ]
            return variants[lane]
        if context.role == RoleName.WITCH:
            variants = [
                f"我这轮先按轮次收益看人，不会因为一句狠话就跟票。{hot}和{second}后面谁急着收票，谁更值得进我的视野。",
                f"{early_round}别把票打成情绪票。我要听谁能说清楚为什么这个人今天必须出，尤其{hot}这种位置别只给态度。",
                f"我不喜欢现在就把人钉死，药和票都一样，乱交就是送轮次。{hot}后面如果继续空压，我会回头看他的身份压力。",
                f"今天先找做局的人，不找最吵的人。{hot}如果能把逻辑讲完整我可以放，讲不完整就进票池。",
            ]
            return variants[lane]
        if context.role == RoleName.HUNTER:
            variants = [
                f"我这轮不怕站出来对线，但不想把枪口和身份压力浪费在空情绪上。{hot}先给压力，后面解释不清我会一直盯。",
                f"{early_round}我看谁敢落票也敢承担后果。{hot}和{second}里如果有人只带节奏不接责任，我会重打。",
                f"我不吃那种安全发言。今天谁都可以保留，但你保留完必须有站边和票意，{hot}后面别只说再看看。",
                f"现在我先压一个标准：发言可以错，但不能滑。{hot}如果一直绕开自己的投票倾向，我会直接点。",
            ]
            return variants[lane]
        if context.role == RoleName.IDIOT:
            variants = [
                f"我这轮先听谁在制造假共识。{hot}和{second}不用急着定死，但谁想把他们快速打成公共靶，我会更先看谁的身份动机。",
                f"{early_round}我不怕被点，但我怕好人跟着空节奏跑。现在先把{hot}放观察位，后面看票型有没有人补刀。",
                f"我不会装自己有信息，平民边能做的就是看谁说话前后不一致。{hot}后面如果改口，我会记这条线。",
                f"今天先别急着一锤定音。{hot}要聊清楚自己的票意，{second}要解释为什么站边或不站边，不然两边都进视野。",
            ]
            return variants[lane]
        variants = [
            f"我就是普通身份视角，先不装有信息。今天重点看{hot}和{second}谁先把空信息聊成定论，{early_round}这种收口最容易出狼。",
            f"现在还没票型，我先给标准：谁只会复述别人、自己不落点，谁就比说错一句话更可疑。{hot}先放观察位。",
            f"我不急着站死任何人，但今天必须有人进票池。{hot}后面要给明确态度，{second}如果顺势补刀我也一起看。",
            f"{early_round}我看发言的新增量。谁只是把场上公共话题换个说法再讲一遍，我会往狼坑里放，{hot}先听后续。",
        ]
        return variants[lane]

    def _early_round_phrase(self, context: AIContext) -> str:
        """按天数生成轮次措辞，避免中后期仍说第一天。"""
        if context.day <= 1:
            return "第一天"
        if context.day == 2:
            return "第二天"
        return f"第{context.day}天"

    def _round_observation_phrase(self, context: AIContext) -> str:
        """按当前天数生成观察口径，避免中后局还说第一轮。"""
        if context.day <= 1:
            return "第一轮"
        return "这轮"

    def _round_risk_phrase(self, context: AIContext) -> str:
        """按当前天数生成风险提醒，避免中后局复用第一天模板。"""
        if context.day <= 1:
            return "第一天最怕好人被假节奏带散"
        return "这轮最怕好人被旧站边绑死"

    def _seer_chain_speech(self, context: AIContext) -> str | None:
        """预言家有多晚验人时，白天必须把信息链转成票意。"""
        if context.role != RoleName.SEER or context.phase not in {"day_speech", "last_words"}:
            return None
        history = self._seer_result_history(context)
        if len(history) < 2:
            return None
        wolf_checks = [seat for seat, result in history if result == "狼人"]
        good_checks = [seat for seat, result in history if result == "好人"]
        chain = "，".join(f"{seat}号{result}" for seat, result in history[-3:])
        if wolf_checks:
            target = wolf_checks[-1]
            shield = f"{good_checks[-1]}号" if good_checks else "我的金水"
            variants = [
                f"我把验人链报完整：{chain}。今天先围绕{target}号出票，{shield}暂时别动；谁还在给{target}号留后路，我直接进第二狼坑。",
                f"我的信息不是单点：{chain}。这轮票口别散，{target}号优先处理，后面再盘谁借{shield}做抗推。",
            ]
            return variants[context.player_id % len(variants)]
        anchor = good_checks[-1]
        variants = [
            f"我目前验人链是：{chain}。没有查杀时别硬归神职线，先从打{anchor}号最急的人里找狼。",
            f"我报清楚两晚信息：{chain}。{anchor}号先放下，今天重点看谁一直想把好人信息做成抗推。",
        ]
        return variants[context.player_id % len(variants)]

    def _fallback_claim_response(self, context: AIContext, scored_targets: list[int]) -> str | None:
        """遇到公开身份宣称时，按身份/阵营生成站边或对冲，而不是模板点评。"""
        if context.structured is None or not context.structured.public_claims:
            return None
        claim = context.structured.public_claims[-1]
        if claim.speaker_id == context.player_id:
            return None
        claimer = f"{claim.speaker_seat_no}号"
        pressure = self._claim_pressure_seat(context, claim, scored_targets)
        side = self._claim_side_lane(context, claim.speaker_id)
        if claim.claimed_role == RoleName.SEER:
            if context.role == RoleName.SEER:
                variants = [
                    f"{claimer}这个预言家我不认。我自己有预言家视角，今天先别被他验人带跑，重点看谁急着替{claimer}冲票。",
                    f"{claimer}如果是真预，警徽流和验人链路不该这么散。我这里会跟他对冲，今天先看站边{claimer}最用力的位置。",
                ]
                return self._with_claim_response_voice(context, variants[context.player_id % len(variants)])
            if context.role == RoleName.WEREWOLF:
                wolf_teammate_ids = {teammate.player_id for teammate in context.structured.wolf_teammates}
                if claim.speaker_id in wolf_teammate_ids:
                    variants = [
                        f"{claimer}这张我先给真预面。验人链至少敢落地，今天更该看谁第一时间反打{claimer}，{pressure}的态度尤其要摊开。",
                        f"我会偏站{claimer}，但不跟着无脑冲。今天重点看{pressure}怎么处理验人结果，谁装中立谁更像在躲票。",
                        f"{claimer}起跳后别把场面聊散。我先顺着他的验人线压一轮，{pressure}如果只喊不认不给理由，我会优先进票。",
                    f"我不替{claimer}兜底，但他敢报验人就有信息量。现在更要看其他人的站边速度，尤其{pressure}有没有借机倒打。",
                    ]
                    return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
                variants = [
                    f"{claimer}先别急着打死，真预和悍跳都可能这么聊。我更想看{pressure}怎么站边，谁借预言家牌硬收票，谁更脏。",
                    f"我不直接认{claimer}，但也不现在冲他。今天最危险的是借一个身份把票打死，所以我先压{pressure}的站边速度。",
                    f"{claimer}这张牌可以留半轮。要是全场马上围着他转，狼就藏在顺势补票里；我今天先看谁拿他当挡箭牌。",
                    f"{claimer}出来以后，桌面很容易变成单线站边。我暂时不跟死，重点看{pressure}有没有借这件事强行收口。",
                    f"{claimer}真假先放桌面，我更在意谁马上把票往外推。{pressure}如果只顺着公共话题说话，不给自己判断，我会先打这里。",
                ]
                return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
            if context.role == RoleName.WITCH:
                variants = [
                    f"{claimer}的预言家身份我先不全认，但药瓶和票一样都讲轮次收益。今天先让他把后续验人路线讲清楚，我票更想压{pressure}。",
                    f"{claimer}现在是核心信息位，真假要靠发言链和站边看。谁急着把{claimer}一脚踢死，我会优先怀疑谁在逼药瓶节奏。",
                    f"我不会现在拍身份救{claimer}，但这轮也别把核心位当普通焦点打。先听{pressure}怎么解释和他的关系线。",
                    f"{claimer}先留验证空间。药瓶最怕被假节奏骗轮次，我会看{pressure}是不是在躲自己的发言责任。",
                ]
                return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
            if context.role == RoleName.HUNTER:
                variants = [
                    f"{claimer}的预言家身份我先不全认，但我的枪口不会跟着情绪乱转。今天先让他讲验人路线，我票更想压{pressure}。",
                    f"{claimer}现在是核心信息位，真假要靠发言链和站边看。谁急着把{claimer}一脚踢死，谁先进我枪口视野。",
                    f"我不会现在拍身份救{claimer}，但这轮也别把核心位当普通焦点打。先听{pressure}怎么解释和他的关系线。",
                    f"{claimer}先留验证空间。猎人牌最怕好人被假节奏带散，我会看{pressure}是不是在躲自己的发言责任。",
                ]
                return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
            if context.role == RoleName.IDIOT:
                variants = [
                    f"{claimer}的预言家身份我先不全认，但我不怕被点，也不怕被拿去做抗推。今天先让他讲验人路线，我票更想压{pressure}。",
                    f"{claimer}现在是核心信息位，真假要靠发言链和站边看。谁急着把{claimer}一脚踢死，谁更像在找抗推位。",
                    f"我不会现在拍身份救{claimer}，但这轮也别把核心位当普通焦点打。先听{pressure}怎么解释和他的关系线。",
                    f"{claimer}先留验证空间。我不怕被推，怕的是好人跟着假共识走，{pressure}必须交清自己的发言责任。",
                ]
                return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
            variants = [
                f"{claimer}跳预以后，今天不是简单信不信的问题。我要看他验人能不能闭环，也看{pressure}有没有借机冲票。",
                f"我先给{claimer}半个回合，不直接认。平民视角最怕假预带票，也怕真预被狼冲，今天我会盯站边速度。",
                f"{claimer}这张牌先放在桌面中心。谁只喊认或不认、不给理由，我会比{claimer}本人更先打。",
                f"{claimer}这件事先不要被一句话定死。平民能做的是看谁在制造票坑，{pressure}要给明确态度。",
                f"我不急着站死{claimer}。今天先听链路能不能自洽，再看谁把这个话题当工具去压别人。",
                f"普通好人视角没夜里信息，我只看动作。{pressure}如果围着{claimer}绕半天不给票意，我会先打这里。",
                f"{claimer}出来以后我不跟喊口号。今天每个人都要交站边原因，尤其{pressure}不能只说观望。",
                f"我暂时不把票交给{claimer}，也不反手打死他。先看{pressure}的反应是不是为了躲自己的狼坑。",
                f"这轮信息核心是{claimer}，但出票不能只靠身份。{pressure}的发言如果不落到票，我会比验人本身更重看。",
                f"我先按发言兑现看人：{claimer}要给链路，{pressure}要给态度，谁含糊谁就有狼面。",
            ]
            return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
        if claim.claimed_role == RoleName.WITCH:
            if context.role == RoleName.WEREWOLF:
                variants = [
                    f"{claimer}这个女巫我不直接认。真女巫要讲清药瓶收益，不是拍身份逼全场让票；今天先把{claimer}放进票型压力。",
                    f"{claimer}跳女巫以后不能自动免推。谁借这个身份把票口挪开，我会一起看，今天我更想压{claimer}的用药逻辑。",
                ]
                return self._with_claim_response_voice(context, variants[context.player_id % len(variants)])
            variants = [
                f"{claimer}跳女巫先别出。女巫牌有药瓶信息，今天更该看谁一直推{claimer}，{pressure}的票意必须解释。",
                f"女巫牌先留轮次。{claimer}如果是假，后面会露用药矛盾；现在急着冲女巫的人，收益更像狼。",
                f"{claimer}这身份我先给一天空间。今天票别落在女巫身上，优先压把女巫做抗推的{pressure}。",
            ]
            return self._with_claim_response_voice(context, variants[self._claim_response_lane(context, variants)])
        return None

    def _claim_pressure_seat(self, context: AIContext, claim, scored_targets: list[int]) -> str:
        """身份宣称回应里的压力位必须是宣称者之外的真实外置位。"""
        excluded = {claim.speaker_id, context.player_id}
        inspected_target_id = getattr(claim, "inspected_target_id", None)
        if inspected_target_id is not None and inspected_target_id not in excluded:
            return f"{inspected_target_id + 1}号"
        for target_id in scored_targets:
            if target_id not in excluded:
                return f"{target_id + 1}号"
        if context.structured:
            for player in context.structured.public_players:
                if player.alive and player.player_id not in excluded:
                    return f"{player.seat_no}号"
        return "其他位置"

    def _fallback_wolf_fake_seer_claim(self, context: AIContext, scored_targets: list[int]) -> str | None:
        """狼人 fallback 有条件悍跳预言家，制造真假预对冲，而不是永远打模糊发言。"""
        if context.role != RoleName.WEREWOLF or context.phase != "day_speech" or context.structured is None:
            return None
        if any(claim.speaker_id == context.player_id for claim in context.structured.public_claims):
            return None
        existing_seer_claims = [claim for claim in context.structured.public_claims if claim.claimed_role == RoleName.SEER]
        wolf_teammate_ids = {teammate.player_id for teammate in context.structured.wolf_teammates}
        wolf_teammate_already_jumped = any(claim.speaker_id in wolf_teammate_ids for claim in existing_seer_claims)
        if wolf_teammate_already_jumped:
            return None
        lane = (context.player_id + context.day) % 4
        wants_jump = "悍跳" in context.strategy_style or "冲票" in context.strategy_style
        if not wants_jump and not existing_seer_claims:
            return None
        target_id, result = self._fake_seer_inspection(context, scored_targets, existing_seer_claims)
        if target_id is None:
            return None
        target_seat = target_id + 1
        if existing_seer_claims:
            claimer = existing_seer_claims[-1].speaker_seat_no
            variants = [
                f"我对跳预言家，{target_seat}号{result}。{claimer}号那套验人链不闭合，今天别被他的定义带跑，重点看谁站边他站得最急。",
                f"我也把身份拍了：预言家，昨晚验{target_seat}号是{result}。{claimer}号如果是真预，不该把票口收得那么急，我今天和他对冲。",
                f"我这里是真预视角，{target_seat}号给{result}。{claimer}号像悍跳抢节奏，今天重点不是谁声音大，是谁的验人能接上发言链。",
                f"{claimer}号我不认，我跳预言家，夜里验到{target_seat}号{result}。今天票别散，把两个预言家的站边和验人逻辑盘清楚。",
            ]
            return variants[lane]
        variants = [
            f"我直接跳预言家，昨晚验{target_seat}号是{result}。今天先别散票，谁急着给{target_seat}号找退路，谁就进我的狼坑。",
            f"我拍身份带节奏：预言家，{target_seat}号{result}。{self._early_round_phrase(context)}不怕对跳，怕的是有人听完验人还继续装中立。",
            f"预言家牌出来报信息，夜里摸了{target_seat}号，结果是{result}。今天先围绕这条线打，后面谁改口我会记票型。",
            f"我起跳预言家，{target_seat}号给{result}。这轮我要的不是全场马上认我，是每个人把站边理由说清楚。",
        ]
        return variants[lane]

    def _fake_seer_inspection(self, context: AIContext, scored_targets: list[int], existing_seer_claims: list) -> tuple[int | None, str]:
        """给狼人伪预言家选择一个不会暴露队友的公开验人口径。"""
        if context.structured is None:
            return None, "好人"
        teammate_ids = {teammate.player_id for teammate in context.structured.wolf_teammates}
        alive_non_teammates = [
            player.player_id
            for player in context.structured.public_players
            if player.alive and player.player_id != context.player_id and player.player_id not in teammate_ids
        ]
        if not alive_non_teammates:
            return None, "好人"
        if existing_seer_claims:
            latest = existing_seer_claims[-1]
            inspected_target_id = getattr(latest, "inspected_target_id", None)
            inspected_result = getattr(latest, "inspected_result", None)
            if inspected_target_id in alive_non_teammates:
                return inspected_target_id, "好人" if inspected_result == "狼人" else "狼人"
        for target_id in scored_targets:
            if target_id in alive_non_teammates:
                return target_id, "狼人"
        return alive_non_teammates[(context.player_id + context.day) % len(alive_non_teammates)], "好人"

    def _claim_side_lane(self, context: AIContext, claimer_id: int) -> int:
        """按号位和身份给公开身份宣称生成不同站边角度。"""
        return context.player_id + claimer_id + context.day

    def _claim_response_lane(self, context: AIContext, variants: list[str]) -> int:
        """身份宣称回应按已回应人数错开，避免连续玩家复读同一模板。"""
        claim_response_count = 0
        if context.structured:
            for speech in context.structured.recent_public_speeches:
                if speech.speaker_id == context.player_id:
                    continue
                if any(token in speech.content for token in ["预言家", "验人", "真预", "悍跳", "站边", "验证空间", "跳预"]):
                    claim_response_count += 1
        persona_offset = sum(ord(char) for char in (context.persona_style or "")) % max(1, len(variants))
        return (context.player_id * 2 + context.day + claim_response_count + persona_offset) % len(variants)

    def _with_claim_response_voice(self, context: AIContext, content: str) -> str:
        """给身份回应加不同分析入口，避免多人同一句式站边。"""
        entrances = [
            "我按票型说，",
            "我按发言新增量看，",
            "反应速度这块，",
            "我按轮次收益盘，",
            "我只抓站边兑现，",
            "补票动作上，",
            "我从验人链反推，",
            "我按谁最想收口来看，",
        ]
        index = (context.player_id * 3 + context.day + len(context.persona_style or "")) % len(entrances)
        entrance = entrances[index]
        if content.startswith(entrance):
            rendered = content
        else:
            content = re.sub(r"^我先", "", content)
            rendered = entrance + content
        if context.phase in {"day_speech", "exile_pk_speech", "last_words"} and not re.search(r"今天|这轮", rendered):
            rendered = rendered.replace("，", "，这轮", 1) if "，" in rendered else f"这轮{rendered}"
        return rendered

    def _pick_public_speech_evidence(self, context: AIContext, scored_targets: list[int]):
        """选一条可引用的公开发言证据。"""
        if context.structured is None:
            return None
        alive_ids = {player.player_id for player in context.structured.public_players if player.alive}
        alive_seats = {player.seat_no for player in context.structured.public_players if player.alive}
        preferred_seats = {target_id + 1 for target_id in scored_targets[:4] if target_id in alive_ids}
        candidates = [
            item
            for item in reversed(context.structured.recent_public_speeches)
            if item.speaker_seat_no != context.player_id + 1
            and item.speaker_id in alive_ids
            and item.content.strip()
            and "“" not in item.content
            and "”" not in item.content
        ]
        preferred = [
            item
            for item in candidates
            if not self._looks_like_fallback_echo(item.content)
            and (
                item.speaker_seat_no in preferred_seats
                or any(seat in preferred_seats for seat in item.mentioned_seat_nos if seat in alive_seats)
            )
        ]
        pool = preferred or [item for item in candidates if not self._looks_like_fallback_echo(item.content)]
        return pool[context.player_id % len(pool)] if pool else None

    def _preferred_alive_mentioned_seat(self, context: AIContext, mentioned_seat_nos: list[int], fallback_seat: str) -> str:
        """从被点名对象里挑一个当前仍存活的号位。"""
        if context.structured is None:
            return fallback_seat
        alive_seats = {
            player.seat_no
            for player in context.structured.public_players
            if player.alive
        }
        for seat_no in mentioned_seat_nos:
            if seat_no in alive_seats:
                return f"{seat_no}号"
        return fallback_seat

    def _looks_like_fallback_echo(self, content: str) -> bool:
        """避免 fallback 继续引用 fallback，形成套娃发言。"""
        echo_markers = [
            "这段我会先抓住",
            "这里有个具体问题",
            "刚才这句",
            "我不急着跟",
            "推得太顺",
            "我会防这里",
            "不完全同频",
            "我先不跟死",
            "发言我记一笔",
            "这轮我把",
            "这个位置可以进票池",
            "暂时不摘",
            "这张我先挂听",
            "真正让我不舒服",
            "不想让场子只围着",
        ]
        return any(marker in content for marker in echo_markers) or "“" in content or "”" in content

    def _latest_seer_result(self, context: AIContext) -> tuple[int, str] | None:
        """从预言家私有记忆中读取最近一次查验。"""
        history = self._seer_result_history(context)
        return history[-1] if history else None

    def _seer_result_history(self, context: AIContext) -> list[tuple[int, str]]:
        """读取预言家的全部历史查验结果，返回 seat_no/result。"""
        if context.structured is None:
            return []
        if context.structured.seer_inspections:
            return [
                (fact.target_seat_no, fact.result)
                for fact in sorted(context.structured.seer_inspections, key=lambda fact: (fact.night_id, fact.day))
            ]
        results: list[tuple[int, str]] = []
        for item in reversed(context.structured.private_observations):
            target_seat = item.data.get("target_seat_no")
            result = item.data.get("result")
            if isinstance(target_seat, int) and result in {"狼人", "好人"}:
                results.append((target_seat, result))
                continue
            if "夜查验" not in item.content:
                continue
            match = re.search(r"（(\d+)号）\s*->\s*(狼人|好人)", item.content)
            if match:
                results.append((int(match.group(1)), match.group(2)))
        return list(reversed(results))

    def _seer_inspection_history(self, context: AIContext) -> list[int]:
        """读取预言家已验过的 player_id。"""
        if context.structured is None:
            return []
        if context.structured.seer_inspections:
            return [fact.target_id for fact in context.structured.seer_inspections]
        inspected: list[int] = []
        for item in context.structured.private_observations:
            target_id = item.data.get("target_id")
            if isinstance(target_id, int) and target_id not in inspected:
                inspected.append(target_id)
                continue
            target_seat = item.data.get("target_seat_no")
            if isinstance(target_seat, int) and target_seat - 1 not in inspected:
                inspected.append(target_seat - 1)
        return inspected

    def _short_quote(self, content: str, limit: int = 34) -> str:
        """截取一段桌面可引用的话。"""
        content = re.sub(r"\s+", "", content).rstrip("。！？!?，,；;")
        if len(content) <= limit:
            return f"“{content}”"
        return f"“{content[:limit]}...”"

    def _paraphrase_evidence(self, content: str, hot: str, mentioned: str) -> str:
        """把公开证据转成转述，避免多人重复逐字引用同一句话。"""
        if "站" in content or "保" in content:
            return f"{hot}在{mentioned}身上给倾向给得太早"
        if "链路" in content or "逻辑" in content:
            return f"{hot}打{mentioned}时一直强调链路，但自己落点也没完全闭合"
        if "预言家" in content or "金水" in content or "查杀" in content:
            return f"{hot}已经把身份信息摆到桌面上"
        return f"{hot}点{mentioned}有态度，但缺少能直接出票的硬理由"

    def _speech_attack_angle(self, context: AIContext, evidence, mentioned: str) -> str:
        """按玩家号位拆分发言切入角度，降低兜底发言同质化。"""
        seat = evidence.speaker_seat_no
        angles = [
            f"{seat}号先给结论再找理由，顺序不自然",
            f"{seat}号只讲谁像问题，没有讲自己准备怎么投",
            f"{seat}号把{mentioned}推上台面，但没有解释收益从哪里来",
            f"{seat}号话里留了退路，像怕后面票型反打",
            f"{seat}号一直借公共话题发力，自己的新增视角太少",
        ]
        if context.role == RoleName.WEREWOLF:
            angles.extend(
                [
                    f"{seat}号太容易成为好人带队点，白天先压住他的可信度",
                    f"{seat}号现在像信息位外溢，不能让他舒服收束场面",
                ]
            )
        return angles[(context.player_id + seat + context.day) % len(angles)]

    def _build_fallback_wolf_chat(
        self,
        context: AIContext,
        scored_targets: list[int],
        *,
        selected_target_id: int | None = None,
    ) -> str:
        """给本地模式生成狼人夜聊。"""
        ranked_targets = [selected_target_id] if selected_target_id is not None else []
        ranked_targets.extend(target_id for target_id in scored_targets if target_id != selected_target_id)
        hot = f"{(ranked_targets[0] + 1)}号" if ranked_targets else "高价值位"
        second = f"{(ranked_targets[1] + 1)}号" if len(ranked_targets) > 1 else hot
        third = f"{(ranked_targets[2] + 1)}号" if len(ranked_targets) > 2 else second
        target_id = ranked_targets[0] if ranked_targets else None
        primary_reason = self._wolf_kill_reason(context, target_id, angle=0)
        pressure_reason = self._wolf_kill_reason(context, target_id, angle=1)
        cover_reason = self._wolf_kill_reason(context, target_id, angle=2)
        counter_reason = self._wolf_kill_reason(context, target_id, angle=3)
        round_id = 1
        current_plan_seat = hot
        round_id = self._current_wolf_chat_round(context)
        if context.structured and context.structured.wolf_chat_records:
            round_id = max(round_id, context.structured.wolf_chat_records[-1].round_id)
            last_target = context.structured.wolf_chat_records[-1].proposed_target_seat_no
            if last_target is not None and selected_target_id is None:
                current_plan_seat = f"{last_target}号"
        lane = self._wolf_chat_lane(context)
        night_variant = 0
        if context.structured is not None:
            night_variant = max(0, context.structured.night_id - 1)
        shifted_lane = (lane + night_variant) % 4
        if round_id >= 2:
            variants = {
                0: f"第二轮我收口，{hot}不换。{primary_reason}，明天别主动解释死因，先让好人自己盘错方向。",
                1: f"我补白天口径：{current_plan_seat}死了以后别集体踩{second}，太齐会脏。一个人压{second}，一个人回头看{third}，票型分开。",
                2: f"我不建议临时改{third}，改刀收益不如{hot}稳定。{cover_reason}，而且明天能把{second}留成可推位置。",
                3: f"{current_plan_seat}这刀落下去以后，我白天会轻踩{second}，不硬冲。谁追夜里收益，我们就说他在用死讯倒推带节奏。",
            }
            if night_variant % 2 == 1:
                variants[0] = f"这一夜我换个收法，还是落{hot}。{pressure_reason}，这刀收益在于拆带队点；明天我不提死因，只看谁主动把票带回{second}。"
                variants[1] = f"我负责拆口径：{current_plan_seat}死后我们暂时绕开{third}，让好人自己争真假信息。票型上我会把压力丢给{second}。"
                variants[2] = f"临时改{third}风险太高，{hot}这刀更能拆归票。{cover_reason}，白天我们分开站，不要一起解释收益。"
                variants[3] = f"{current_plan_seat}落刀后的收益是制造票型分歧。我白天会保留，不急着踩{second}，谁把死讯盘得太准，谁反而像有夜里视角。"
            return variants[shifted_lane]
        if target_id is not None and primary_reason:
            variants = {
                0: f"我先提主刀：今晚动{hot}。{primary_reason}。{second}先别急着杀，留着白天能做票坑。",
                1: f"我给备刀：今晚也可以动{hot}。{pressure_reason}，这刀能拆好人白天的收束点；如果队友坚持前一个刀口，我也能接。",
                2: f"我站{hot}，但明天口径要散。{cover_reason}。我可以去轻踩{third}，别四个人都围着同一张牌发力。",
                3: f"{hot}收益够，不用贪换刀。{counter_reason}。如果明天有人保{hot}遗留视角，就顺势把压力转给{second}。",
            }
            if night_variant % 2 == 1:
                variants[0] = f"这夜我先开刀口：{hot}。{primary_reason}，这刀收益够；白天不要四个人都顺这个死讯打。"
                variants[1] = f"我给另一个收益角度，{hot}可以动。{pressure_reason}，明天把{second}留成明线压力位。"
                variants[2] = f"我认{hot}这刀，不过口径要错开。{cover_reason}，我白天会绕到{third}身上试反应。"
                variants[3] = f"{hot}这刀收益够干净。{counter_reason}，如果有人硬保死者遗留视角，我们再顺势打{second}。"
            return variants[shifted_lane]
        lines = [
            f"前面的刀口思路可以延展，我偏向动{hot}。这位置白天说话最像有真视角，再让他多活一轮容易把我们票型盘出来。",
            f"如果今晚统一刀，我站{hot}。{second}也能刀，但{hot}更像明天会带队的人，收益更直接。",
            f"我补一个理由，先杀{hot}不是因为他最跳，是因为他活到白天最容易把场子收住。我们今晚要的是拆信息位，不是随便找个民走。",
        ]
        return lines[lane % len(lines)]

    def _wolf_chat_lane(self, context: AIContext) -> int:
        """狼队 fallback 发言分工：主刀、备刀、白天口径、反改刀。"""
        if context.structured and context.structured.wolf_chat_records:
            prompt_round = 1
            round_match = re.search(r"第(\d+)轮", context.prompt)
            if round_match:
                prompt_round = int(round_match.group(1))
            same_round = [
                record
                for record in context.structured.wolf_chat_records
                if record.round_id == prompt_round
            ]
            return min(len(same_round), 3)
        # 新一轮第一位固定主刀，后续按已有发言数切分，避免少狼局同轮复读。
        return 0

    def _wolf_kill_reason(self, context: AIContext, target_id: int | None, angle: int = 0) -> str:
        """基于公开证据生成狼队刀口收益说明。"""
        if context.structured is None or target_id is None:
            return ""
        target_seat = target_id + 1
        for claim in reversed(context.structured.public_claims):
            if claim.speaker_id == target_id:
                reasons = [
                    f"{target_seat}号已经公开往{claim.claimed_role.value}方向聊，哪怕是诈身份，明天也容易带队归票",
                    f"{target_seat}号身份压力最大，留着会逼我们白天一直解释站边",
                    f"{target_seat}号一死，好人会先争他真假，我们反而能藏住狼队票型",
                    f"{target_seat}号活着容易把验人和票型串起来，今晚处理比白天硬推省成本",
                ]
                return reasons[angle % len(reasons)]
        for speech in reversed(context.structured.recent_public_speeches):
            if speech.speaker_id == target_id and not self._looks_like_fallback_echo(speech.content):
                reasons = [
                    f"{target_seat}号白天存在感高，留着的信息压力会继续压我们狼坑",
                    f"{target_seat}号能把散点发言收成主线，明天容易带出一波归票",
                    f"{target_seat}号死后他的怀疑对象还能被我们借用，白天有转移空间",
                    f"{target_seat}号如果继续活着，会逼低位狼接更多解释，风险太高",
                ]
                return reasons[angle % len(reasons)]
        mentions = [
            speech.speaker_seat_no
            for speech in context.structured.recent_public_speeches
            if target_seat in speech.mentioned_seat_nos
        ]
        if len(mentions) >= 2:
            reasons = [
                f"{target_seat}号已经被{len(mentions)}个人反复放进讨论中心，刀掉的收益是打断明天围绕他的站边链",
                f"{target_seat}号处在讨论中心，死讯会让好人围绕旧矛盾内耗",
                f"{target_seat}号的关系线太多，今晚处理能减少明天能被盘出的联动",
                f"{target_seat}号已经形成公共锚点，留着会让好人更容易统一方向",
            ]
            return reasons[angle % len(reasons)]
        fallback_reasons = [
            f"{target_seat}号留到明天会继续产出信息和归票压力，今晚处理收益更稳",
            f"{target_seat}号位置不低，刀掉以后明天可以把焦点转给旁边的人",
            f"{target_seat}号不像天然抗推位，白天硬推成本高，夜里处理更干净",
            f"{target_seat}号能活到后面会压缩狼队发言空间，今晚先拆掉比较稳",
        ]
        return fallback_reasons[angle % len(fallback_reasons)]
