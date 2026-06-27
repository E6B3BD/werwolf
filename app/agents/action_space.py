"""Agent 可选动作空间。

这一层只枚举“此刻能表达哪些意图”，不做最终规则结算。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ActionKind = Literal[
    "speak",
    "vote",
    "wolf_chat",
    "wolf_confirm",
    "night_action",
    "guard",
    "inspect",
    "witch_action",
    "campaign_speech",
    "pk_campaign_speech",
    "exile_pk_speech",
    "sheriff_vote",
    "sheriff_pk_vote",
    "exile_pk_vote",
    "choose_speech_order",
    "last_words",
    "hunter_shot",
    "badge_transfer",
    "skip",
]


@dataclass(frozen=True, slots=True)
class ActionOption:
    """单个 Agent 可选动作。"""

    kind: ActionKind
    label: str
    target_ids: list[int] = field(default_factory=list)
    required: bool = False
    guidance: str = ""


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """当前阶段的可选动作集合。"""

    phase: str
    actor_id: int
    options: list[ActionOption]

    def first_targets(self, kind: ActionKind) -> list[int]:
        for option in self.options:
            if option.kind == kind:
                return option.target_ids
        return []

    def describe(self) -> str:
        if not self.options:
            return "当前没有可执行动作。"
        lines = []
        for index, option in enumerate(self.options, start=1):
            targets = ", ".join(str(target_id) for target_id in option.target_ids) or "无目标"
            lines.append(f"{index}. {option.kind}: {option.label}; targets=[{targets}]; {option.guidance}")
        return "\n".join(lines)
