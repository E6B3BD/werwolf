"""Agent 决策管线。

目标是把 Observe -> Memory -> Think -> Action 的边界集中到一处，
避免每个游戏阶段各自拼 prompt。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.agents.advisor import advise
from app.agents.action_space import ActionKind, ActionSpace
from app.agents.runtime import _render_visible_event, _render_visible_message
from app.agents.runtime import AIContext, OpenAIAgentRuntime
from app.engine.models import AgentDecision, PlayerAgentState, PlayerState


@dataclass(slots=True)
class DecisionRequest:
    """一次 Agent 决策请求。"""

    player: PlayerState
    context: AIContext
    action_space: ActionSpace
    preferred_action: ActionKind
    stage_goal: str


class DecisionPipeline:
    """统一 Agent 决策管线。"""

    def __init__(self, runtime: OpenAIAgentRuntime) -> None:
        self.runtime = runtime

    async def decide(self, request: DecisionRequest, agent_state: PlayerAgentState | None = None) -> AgentDecision:
        """执行一次决策。"""
        request.context.prompt = self._build_prompt(request, agent_state)
        decision = await self.runtime.decide(request.context)
        decision.action_type = decision.action_type or request.preferred_action
        if agent_state:
            if decision.reason and request.context.phase != "wolf_chat":
                agent_state.last_internal_plan = self._compact(
                    f"{agent_state.last_internal_plan}；{request.context.phase}: {decision.reason}",
                    limit=520,
                )
            if decision.target_id is not None and request.context.phase != "wolf_chat":
                agent_state.current_focus = f"{decision.target_id + 1}号"
            agent_state.memory_version += 1
        return decision

    def _build_prompt(self, request: DecisionRequest, agent_state: PlayerAgentState | None) -> str:
        """构造阶段任务提示。"""
        state_text = "无持续状态。"
        if agent_state:
            state_text = (
                f"你的长期私有摘要：{agent_state.private_summary or '无'}\n"
                f"你的公开立场摘要：{agent_state.last_public_position or '无'}\n"
                f"你当前关注：{agent_state.current_focus or '无'}\n"
                f"你上次内部计划：{agent_state.last_internal_plan or '无'}"
            )
        evidence_text = self._build_evidence_text(request.context)
        return "\n".join(
            [
                request.stage_goal,
                "可选动作空间：",
                request.action_space.describe(),
                "可引用公开证据：",
                evidence_text,
                "你的持续状态：",
                state_text,
                "输出要求：先在 reason 中保留内部判断，再在 content 中只写桌上会说出口的话。content 要么引用具体号位/发言矛盾/票型证据，要么在证据不足时给出本轮观察框架、保留位和投票倾向；禁止为了引用而机械复述上一位。",
            ]
        )

    def _build_evidence_text(self, context: AIContext) -> str:
        """把结构化证据转成人类可读摘要，避免只给一坨 JSON。"""
        structured = context.structured
        if structured is None:
            return "无结构化公开证据。"
        lines: list[str] = []
        if structured.new_visible_events or structured.new_visible_messages:
            lines.append("本次新增可见信息：")
            for event in structured.new_visible_events[-6:]:
                lines.append(f"- {_render_visible_event(event)}")
            for message in structured.new_visible_messages[-8:]:
                lines.append(f"- {_render_visible_message(message)}")
        if structured.recent_public_speeches:
            lines.append("最近公开发言：")
            for item in structured.recent_public_speeches[-6:]:
                mentions = "、".join(f"{seat}号" for seat in item.mentioned_seat_nos) or "未点具体号位"
                tags = "、".join(item.stance_keywords) or "无明显关键词"
                lines.append(
                    f"- 第{item.day}天 {item.speaker_seat_no}号({item.speech_type}) 点到[{mentions}] 关键词[{tags}]：{item.content[:120]}"
                )
        if structured.recent_votes:
            lines.append("最近票型：")
            for vote in structured.recent_votes[-8:]:
                lines.append(
                    f"- 第{vote.day}天 {vote.voter_seat_no}号投{vote.target_seat_no}号({vote.vote_type}/{vote.vote_round or '常规'})"
                )
        if structured.public_claims:
            lines.append("公开身份宣称：")
            for claim in structured.public_claims[-6:]:
                lines.append(f"- 第{claim.day}天 {claim.speaker_seat_no}号声称{claim.claimed_role.value}：{claim.source_text[:80]}")
        advice = advise(structured)
        lines.append("基于你可见信息的怀疑排序：")
        lines.append(advice.render(limit=4))
        return "\n".join(lines) if lines else "暂无发言、票型或身份宣称证据。"

    def _compact(self, text: str, limit: int) -> str:
        """限制长期状态长度。"""
        text = text.strip("； \n")
        if len(text) <= limit:
            return text
        return text[-limit:]
