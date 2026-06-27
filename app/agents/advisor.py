"""基于玩家可见视角的轻量策略辅助。

Advisor 不参与规则裁定，只把 AgentVisibleContext 中已经可见的证据
压成怀疑排序，帮助发言和投票更像围绕同一局游戏推进。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.engine.models import AgentVisibleContext, Camp, RoleName


@dataclass(frozen=True, slots=True)
class Suspicion:
    """单个可疑对象评分。"""

    player_id: int
    seat_no: int
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Advice:
    """当前玩家视角下的策略建议。"""

    suspicions: list[Suspicion]
    recommended_target_id: int | None = None

    def render(self, limit: int = 4) -> str:
        if not self.suspicions:
            return "暂无足够证据形成稳定怀疑对象。"
        lines = []
        for item in self.suspicions[:limit]:
            reason = "；".join(item.reasons[:2]) or "公开信息压力较高"
            lines.append(f"- {item.seat_no}号：怀疑值{item.score:.1f}，{reason}")
        if self.recommended_target_id is not None:
            lines.append(f"建议重点施压/投票：{self.recommended_target_id + 1}号。")
        return "\n".join(lines)


def advise(context: AgentVisibleContext) -> Advice:
    """从单个玩家可见上下文生成怀疑排序。"""
    alive_ids = {player.player_id for player in context.public_players if player.alive}
    scores: dict[int, float] = {
        player_id: 0.0
        for player_id in alive_ids
        if player_id != context.self_player.player_id
    }
    reasons: dict[int, list[str]] = {player_id: [] for player_id in scores}

    teammate_ids = {teammate.player_id for teammate in context.wolf_teammates}
    for teammate_id in teammate_ids:
        scores.pop(teammate_id, None)
        reasons.pop(teammate_id, None)

    _score_known_roles(context, scores, reasons)
    _score_claims(context, scores, reasons)
    _score_speeches(context, scores, reasons)
    _score_votes(context, scores, reasons)

    suspicions = [
        Suspicion(
            player_id=player_id,
            seat_no=player_id + 1,
            score=score,
            reasons=reasons.get(player_id, [])[:4],
        )
        for player_id, score in scores.items()
        if score > 0
    ]
    suspicions.sort(key=lambda item: (-item.score, item.seat_no))
    recommended = suspicions[0].player_id if suspicions else None
    return Advice(suspicions=suspicions, recommended_target_id=recommended)


def _add(scores: dict[int, float], reasons: dict[int, list[str]], player_id: int, score: float, reason: str) -> None:
    if player_id not in scores:
        return
    scores[player_id] += score
    if reason not in reasons[player_id]:
        reasons[player_id].append(reason)


def _score_known_roles(context: AgentVisibleContext, scores: dict[int, float], reasons: dict[int, list[str]]) -> None:
    self_role = context.self_player.role
    if self_role == RoleName.SEER:
        for player_id, role in context.known_role_map.items():
            if player_id == context.self_player.player_id:
                continue
            if role == RoleName.WEREWOLF:
                _add(scores, reasons, player_id, 8.0, "你的查验/已知信息指向这里是狼人")
    if context.self_player.camp == Camp.WEREWOLF:
        for claim in context.public_claims:
            threat = {
                RoleName.SEER: 6.0,
                RoleName.WITCH: 4.0,
                RoleName.HUNTER: 2.0,
                RoleName.IDIOT: 1.0,
                RoleName.VILLAGER: 0.5,
            }.get(claim.claimed_role, 0.0)
            if threat:
                _add(scores, reasons, claim.speaker_id, threat, f"{claim.speaker_seat_no}号公开声称{claim.claimed_role.value}，对狼队威胁高")


def _score_claims(context: AgentVisibleContext, scores: dict[int, float], reasons: dict[int, list[str]]) -> None:
    seen_roles: dict[RoleName, list[int]] = {}
    for claim in context.public_claims:
        seen_roles.setdefault(claim.claimed_role, []).append(claim.speaker_id)
        if claim.speaker_id in scores and claim.claimed_role == RoleName.SEER:
            _add(scores, reasons, claim.speaker_id, 1.5, "预言家宣称需要接受站边和验人链路压力")
    for role, claimers in seen_roles.items():
        if role == RoleName.SEER and len(set(claimers)) >= 2:
            for player_id in set(claimers):
                _add(scores, reasons, player_id, 2.0, "出现预言家对跳，需要用发言链路分辨真假")


def _score_speeches(context: AgentVisibleContext, scores: dict[int, float], reasons: dict[int, list[str]]) -> None:
    mention_count: dict[int, int] = {}
    for speech in context.recent_public_speeches:
        if speech.speaker_id in scores:
            if len(speech.mentioned_seat_nos) == 0:
                _add(scores, reasons, speech.speaker_id, 0.7, "发言没有落到具体号位")
            if len(speech.mentioned_seat_nos) >= 3:
                _add(scores, reasons, speech.speaker_id, 0.8, "点人过散，容易是在铺多条退路")
            risky_tokens = {"预言家", "站边", "归票", "查杀", "金水", "保", "打", "票"}
            if any(token in risky_tokens for token in speech.stance_keywords):
                _add(scores, reasons, speech.speaker_id, 0.8, "发言涉及身份/归票，需要后续解释闭环")
        for seat in speech.mentioned_seat_nos:
            player_id = seat - 1
            if player_id in scores:
                mention_count[player_id] = mention_count.get(player_id, 0) + 1
    for player_id, count in mention_count.items():
        if count >= 2:
            _add(scores, reasons, player_id, min(2.5, count * 0.8), f"被{count}条公开发言反复点到")


def _score_votes(context: AgentVisibleContext, scores: dict[int, float], reasons: dict[int, list[str]]) -> None:
    vote_targets: dict[int, int] = {}
    for vote in context.recent_votes:
        vote_targets[vote.target_id] = vote_targets.get(vote.target_id, 0) + 1
        if vote.voter_id in scores and vote.target_id == context.self_player.player_id:
            _add(scores, reasons, vote.voter_id, 1.0, "该位置把票压到你身上，需要回应其动机")
    for player_id, count in vote_targets.items():
        if count >= 2:
            _add(scores, reasons, player_id, min(2.0, count * 0.5), f"票型上已有{count}票压力")
