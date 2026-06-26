"""OpenAI Agents SDK 封装。"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from app.agents.prompts import build_player_instructions
from app.core.config import settings
from app.engine.models import AgentDecision, RoleName

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

        user_prompt = f"""
你是 {context.player_id + 1} 号位玩家。
你的身份是：{context.role.value}。
你能看到下面这些信息：
- 你的公开桌面信息（包括当前存活情况、最近发言、最近票型、最近系统播报）
- 你的私有信息（如果你是神职或狼人，这里会给你身份相关结果）
- 当前可选目标列表

当前阶段：{context.phase}
第 {context.day} 天

公开局面：
{context.visible_state}

可选目标 ID：
{context.allowed_target_ids}

你的任务：
{context.prompt}

请严格返回 JSON 对象。
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
            result = await Runner.run(agent, user_prompt, run_config=self.run_config)
            raw_output = result.final_output
            if isinstance(raw_output, AgentDecision):
                return raw_output
            if isinstance(raw_output, dict):
                return AgentDecision.model_validate(raw_output)
            if isinstance(raw_output, str):
                return AgentDecision.model_validate(json.loads(raw_output))
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

    def _fallback_decision(self, context: AIContext) -> AgentDecision:
        """未配置 OpenAI 时的本地兜底逻辑。"""
        scored_targets = self._score_targets(context)
        target_id: Optional[int] = scored_targets[0] if scored_targets else None

        if context.phase in {"day_speech", "campaign_speech", "pk_campaign_speech", "last_words"}:
            content = self._build_fallback_speech(context, scored_targets)
            return AgentDecision(
                action="speak",
                target_id=None,
                content=content,
                reason=f"未启用 OpenAI，按 {context.player_id + 1} 号位的公开信息、身份目标与私有上下文生成兜底发言。",
            )
        if context.phase in {"day_vote", "sheriff_vote", "sheriff_pk_vote"}:
            return AgentDecision(
                action="vote",
                target_id=target_id,
                content="",
                reason=f"未启用 OpenAI，依据 {context.player_id + 1} 号位的公开信息与身份目标对候选目标排序后投票。",
            )
        if context.phase == "wolf_chat":
            return AgentDecision(
                action="night_action",
                target_id=target_id,
                content=self._build_fallback_wolf_chat(context, scored_targets),
                reason=f"未启用 OpenAI，依据 {context.player_id + 1} 号位狼人可见局势与刀口收益生成夜聊。",
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
            match = re.match(r"(\d+):\s*(玩家\d+)\s*-\s*(存活|死亡)(.*)", line)
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
                if target_id in {0, 1, 2, 3}:
                    scores[target_id] += 1
        if context.role == RoleName.SEER and context.phase == "night_action":
            for target_id in scores:
                scores[target_id] += 1
        if context.phase in {"sheriff_vote", "sheriff_pk_vote"}:
            for target_id in scores:
                scores[target_id] += 2
        if context.role == RoleName.WITCH and context.phase == "night_action":
            for target_id in scores:
                scores[target_id] += 1

        return sorted(scores, key=lambda item: (-scores[item], item))

    def _build_fallback_speech(self, context: AIContext, scored_targets: list[int]) -> str:
        """给本地模式一个更像真人局的发言。"""
        hot = f"{(scored_targets[0] + 1)}号" if scored_targets else "前置位"
        second = f"{(scored_targets[1] + 1)}号" if len(scored_targets) > 1 else hot
        sharp = "你这发言太飘了" if "情绪" in context.persona_style or "铁腕" in context.persona_style else "这轮逻辑不顺"
        if context.role == RoleName.WEREWOLF:
            lines = [
                f"我先压{hot}，你前面那句站边来得太快，后面又没把理由补实，像先抢位置再回头找补。{second}要是再顺着你补票，我就一起记。",
                f"{hot}这轮最大的问题不是凶，是你把话说满以后又开始往回收，这不像真有视角，更像怕自己后面圆不回来。",
                f"先别急着把{hot}放掉，他前面踩人踩得很响，真到落点时又只剩态度，这种发言很像在给自己做白。",
            ]
            return random.choice(lines)
        if context.role == RoleName.SEER:
            lines = [
                f"我先看{hot}，你前后两次站边不在一条线上，像是在跟着场面找安全位。{second}如果继续替你补逻辑，我会一起盘联动。",
                f"{hot}这轮发言看着完整，其实关键矛盾没接，你只是在重复别人已经点过的东西，这不像有信息位，像在顺势做结论。",
                f"我不会因为一句态度就打死{hot}，但你这轮该解释的改口没解释，后面再不把心路摊开，我一定往重里点。",
            ]
            return random.choice(lines)
        if context.role in {RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER, RoleName.GUARD}:
            lines = [
                f"我先点{hot}，你不是没逻辑，是逻辑全借前置位的，自己没新增信息，这种发言我很难当成真找狼。",
                f"{hot}这轮让我不舒服的点很具体：你前面把人打得很死，后面别人一反压你又开始说别急着定，这就是在留后路。",
                f"今天如果真要出票，我先看{hot}，再看{second}。一个像在抢节奏，一个像在顺着公共结论补刀，都不像正经好人找人。",
            ]
            return random.choice(lines)
        return f"我先点一下{hot}，他的发言和轮次理解明显不顺，后面票型再脏一点我就直接打死。"

    def _build_fallback_wolf_chat(self, context: AIContext, scored_targets: list[int]) -> str:
        """给本地模式生成狼人夜聊。"""
        hot = f"{(scored_targets[0] + 1)}号" if scored_targets else "高价值位"
        second = f"{(scored_targets[1] + 1)}号" if len(scored_targets) > 1 else hot
        lines = [
            f"我能接前面的刀口思路，我也偏向先动{hot}。这位置白天说话最像有真视角，再让他多活一轮容易把我们票型盘出来。",
            f"如果今晚统一刀，我先站{hot}。{second}也能刀，但{hot}更像明天会带队的人，收益更直接。",
            f"我补一个理由，先杀{hot}不是因为他最跳，是因为他活到白天最容易把场子收住。我们今晚要的是拆信息位，不是随便找个民走。",
        ]
        return random.choice(lines)
