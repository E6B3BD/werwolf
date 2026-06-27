"""Agent 对局质量评估。

这一层只做可玩性/文本质量检查，不参与规则裁定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.engine.models import Camp, Phase, RoleName


AI_LEAK_TOKENS = ["target_id", "结构化上下文", "根据提示词", "系统提示", "候选列表", "我先接", "我先接入", "作为AI", "JSON"]
PRIVATE_LEAK_TOKENS = ["狼人夜聊", "狼队友", "刀口", "今晚先刀", "夜谈结束"]
WOLF_HISTORY_SUMMARY_LEAK_TOKENS = ["过往夜晚复盘", "只保留策略教训", "不复述旧夜具体刀口", "上夜狼队", "旧夜"]
CONCRETE_REASON_TOKENS = ["因为", "所以", "但", "逻辑", "票", "发言", "站边", "归票", "身份", "链路", "理由", "收益"]
WOLF_KILL_REASON_TOKENS = ["收益", "带队", "归票", "预言家", "票型", "信息", "拆", "联动", "转移", "口径", "分散"]
AWKWARD_TABLE_TOKENS = ["外置位"]
CHAIN_COMMENTARY_PATTERNS = [
    ("链式接话", re.compile(r"我(?:先|来)?(?:接|接一下|接入)\s*\d{1,2}号")),
    ("链式抓句", re.compile(r"我(?:先|来)?(?:抓|看|说)\s*\d{1,2}号[^。！？!?]{0,12}(?:这句|刚才|这段)")),
    ("连续点评", re.compile(r"\d{1,2}号[^。！？!?]{0,12}(?:这句|刚才那句|这段)[^。！？!?]{0,28}(?:我先|我会|继续盯|没问题)")),
    ("套话抬轿", re.compile(r"\d{1,2}号[^。！？!?]{0,18}(?:这句|刚才那句|这段)[^。！？!?]{0,24}(?:没问题|我先认|偏好人)")),
]
AWKWARD_REPEAT_PATTERNS = [
    ("重复转折", re.compile(r"但[^。！？!?]{0,18}但")),
    ("重复让步", re.compile(r"先[^。！？!?]{0,18}先")),
    ("重复观察", re.compile(r"盯[^。！？!?]{0,18}盯")),
]


@dataclass(slots=True)
class PlayabilityReport:
    """一段样局的可玩性评估结果。"""

    findings: list[str] = field(default_factory=list)
    completed: bool = False
    winner: str = ""
    phases_covered: list[str] = field(default_factory=list)
    day_speech_count: int = 0
    unique_speech_ratio: float = 0.0
    concrete_speech_count: int = 0
    wolf_chat_count: int = 0
    vote_audit_count: int = 0
    max_vote_share: float = 0.0
    night_count: int = 0
    wolf_chat_night_count: int = 0
    wolf_chat_roleplay_variety: int = 0
    wolf_chat_stance_variety: int = 0
    claim_response_count: int = 0
    seer_counterclaim_count: int = 0
    vote_intent_speech_count: int = 0
    day_angle_variety: int = 0
    role_strategy_signal_count: int = 0
    role_strategy_roles: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings


def evaluate_playability(game, *, require_counterclaim: bool = True) -> PlayabilityReport:
    """对一段完整样局做内容质量检查。"""
    report = PlayabilityReport()
    report.completed = game.phase == Phase.GAME_OVER
    report.winner = game.winner or ""
    report.phases_covered = sorted({event.phase for event in game.events} | {message.phase for message in game.message_log})
    day_speeches = [speech for speech in game.speeches if speech.speech_type == "day"]
    report.day_speech_count = len(day_speeches)
    if day_speeches:
        report.unique_speech_ratio = len({speech.content for speech in day_speeches}) / len(day_speeches)
        if report.unique_speech_ratio < 0.65:
            report.findings.append(f"白天发言重复度过高：unique_ratio={report.unique_speech_ratio:.2f}")
        report.concrete_speech_count = sum(
            1
            for speech in day_speeches
            if re.search(r"\d+号", speech.content) and any(token in speech.content for token in CONCRETE_REASON_TOKENS)
        )
        if report.concrete_speech_count < max(2, len(day_speeches) // 3):
            report.findings.append(f"具体推理发言不足：{report.concrete_speech_count}/{len(day_speeches)}")
        for speech in day_speeches:
            nested_template = "这段我会先抓住：“" in speech.content and "这段我会先抓住：“" in speech.content[8:]
            if "“" in speech.content or "”" in speech.content or nested_template:
                report.findings.append(f"白天发言出现套娃引用：{speech.player_name}: {speech.content}")
            for label, pattern in AWKWARD_REPEAT_PATTERNS:
                if pattern.search(speech.content):
                    report.findings.append(f"白天发言出现{label}：{speech.player_name}: {speech.content}")
                    break
            if any(token in speech.content for token in AI_LEAK_TOKENS):
                report.findings.append(f"白天发言有AI字段泄漏：{speech.player_name}: {speech.content}")
            if any(token in speech.content for token in PRIVATE_LEAK_TOKENS):
                report.findings.append(f"白天发言泄漏私密信息：{speech.player_name}: {speech.content}")
            if any(token in speech.content for token in WOLF_HISTORY_SUMMARY_LEAK_TOKENS):
                report.findings.append(f"白天发言复述狼队历史摘要：{speech.player_name}: {speech.content}")
            if any(token in speech.content for token in AWKWARD_TABLE_TOKENS):
                report.findings.append(f"白天发言有机械桌游术语：{speech.player_name}: {speech.content}")
            for label, pattern in CHAIN_COMMENTARY_PATTERNS:
                if pattern.search(speech.content):
                    report.findings.append(f"白天发言出现{label}：{speech.player_name}: {speech.content}")
                    break
        report.claim_response_count = _count_claim_responses(day_speeches)
        report.seer_counterclaim_count = _count_seer_counterclaims(day_speeches)
        report.vote_intent_speech_count = sum(
            1 for speech in day_speeches if any(token in speech.content for token in ["投", "票", "归", "压", "出"])
        )
        report.day_angle_variety = len({_classify_day_angle(speech.content) for speech in day_speeches})
        strategy_roles = _role_strategy_roles(game, day_speeches)
        report.role_strategy_roles = sorted(role.value for role in strategy_roles)
        report.role_strategy_signal_count = len(strategy_roles)
        seer_claim_has_response_window = _seer_claim_followup_count(day_speeches) >= 2
        if seer_claim_has_response_window and any("预言家" in speech.content for speech in day_speeches) and report.claim_response_count < 2:
            report.findings.append(f"白天身份宣称回应不足：claim_response_count={report.claim_response_count}")
        if (
            require_counterclaim
            and len(day_speeches) >= 8
            and seer_claim_has_response_window
            and _has_seer_claim(day_speeches)
            and report.seer_counterclaim_count == 0
        ):
            report.findings.append("有预言家宣称但缺少对跳/对冲，身份博弈不足")
        if len(day_speeches) >= 6 and report.vote_intent_speech_count < max(2, len(day_speeches) // 4):
            report.findings.append(f"白天投票意向不足：{report.vote_intent_speech_count}/{len(day_speeches)}")
        if len(day_speeches) >= 6 and report.day_angle_variety < 4:
            report.findings.append(f"白天发言角度过少：day_angle_variety={report.day_angle_variety}")
        table_roles = {
            game.players[speech.player_id].role
            for speech in day_speeches
            if 0 <= speech.player_id < len(game.players)
        }
        required_strategy_roles = min(4, len(table_roles))
        if len(day_speeches) >= 8 and required_strategy_roles >= 3 and report.role_strategy_signal_count < required_strategy_roles:
            report.findings.append(
                "角色打法链路不足："
                f"role_strategy_signal_count={report.role_strategy_signal_count}, roles={report.role_strategy_roles}"
            )
        repeated_claim_templates = _repeated_claim_response_templates(day_speeches)
        if repeated_claim_templates:
            report.findings.append(f"身份宣称回应模板重复：{repeated_claim_templates}")
    else:
        report.findings.append("没有白天发言样本")

    last_words = [speech for speech in game.speeches if speech.speech_type == "last_words"]
    if last_words:
        day_contents = {speech.content for speech in day_speeches}
        for speech in last_words:
            if speech.content in day_contents:
                report.findings.append(f"遗言复用白天发言：{speech.player_name}: {speech.content}")
            if _looks_like_day_claim_response_template(speech.content):
                report.findings.append(f"遗言像普通白天模板：{speech.player_name}: {speech.content}")

    current_night_records = game._current_wolf_chat_records()
    all_wolf_records = game.wolf_chat_records
    report.wolf_chat_count = len(all_wolf_records)
    report.night_count = len({summary.night_id for summary in game.night_summaries if summary.night_id is not None})
    report.wolf_chat_night_count = len({record.night_id for record in all_wolf_records})
    if all_wolf_records:
        wolf_unique_ratio = len({record.content for record in all_wolf_records}) / len(all_wolf_records)
        if wolf_unique_ratio < 0.75:
            report.findings.append(f"狼聊重复度过高：unique_ratio={wolf_unique_ratio:.2f}")
        per_night_round_texts: dict[tuple[int, int], set[str]] = {}
        per_night_openings: dict[int, set[str]] = {}
        per_night_stances: dict[int, set[str]] = {}
        first_round_targets_by_night: dict[int, list[int]] = {}
        for record in all_wolf_records:
            per_night_round_texts.setdefault((record.night_id, record.round_id), set()).add(record.content)
            opening = re.split(r"[。！？!?]", record.content.strip(), maxsplit=1)[0]
            per_night_openings.setdefault(record.night_id, set()).add(_normalize_wolf_chat_opening(opening))
            if record.stance_to_previous:
                per_night_stances.setdefault(record.night_id, set()).add(record.stance_to_previous)
            if record.round_id == 1 and record.proposed_target_seat_no is not None:
                first_round_targets_by_night.setdefault(record.night_id, []).append(record.proposed_target_seat_no)
            if any(token in record.content for token in WOLF_HISTORY_SUMMARY_LEAK_TOKENS):
                report.findings.append(f"狼聊疑似复述历史摘要模板：{record.content}")
            if record.night_id > 1 and record.round_id == 1 and any(
                token in record.content
                for token in ["第1天夜晚#", "第2天夜晚#", "最终刀口", "来源proposal_vote", "来源engine_default"]
            ):
                report.findings.append(f"第{record.night_id}夜狼聊疑似复述历史摘要：{record.content}")
            if record.proposed_target_seat_no is None:
                report.findings.append(f"狼聊缺少建议刀口：{record.content}")
            elif f"{record.proposed_target_seat_no}号" not in record.content:
                report.findings.append(f"狼聊文本目标与结构化目标不一致：{record.content} / {record.proposed_target_seat_no}号")
            elif _wolf_chat_conflicting_commitment(record.content, record.proposed_target_seat_no):
                report.findings.append(f"狼聊文本存在冲突收口目标：{record.content} / {record.proposed_target_seat_no}号")
            elif _wolf_chat_negates_target(record.content, record.proposed_target_seat_no):
                report.findings.append(f"狼聊文本否定了结构化刀口：{record.content} / {record.proposed_target_seat_no}号")
            for label, pattern in AWKWARD_REPEAT_PATTERNS:
                if pattern.search(record.content):
                    report.findings.append(f"狼聊出现{label}：{record.content}")
                    break
            if any(token in record.content for token in AWKWARD_TABLE_TOKENS):
                report.findings.append(f"狼聊有机械桌游术语：{record.content}")
            if not any(token in record.content for token in WOLF_KILL_REASON_TOKENS):
                report.findings.append(f"狼聊缺少刀口收益说明：{record.content}")
        for (night_id, round_id), texts in per_night_round_texts.items():
            round_count = sum(1 for record in all_wolf_records if record.night_id == night_id and record.round_id == round_id)
            if round_count > 1 and len(texts) == 1:
                report.findings.append(f"第{night_id}夜第{round_id}轮狼聊整轮复读")
        if per_night_openings:
            report.wolf_chat_roleplay_variety = min(len(openings) for openings in per_night_openings.values())
            for night_id, openings in per_night_openings.items():
                night_records = [record for record in all_wolf_records if record.night_id == night_id]
                if len(night_records) >= 4 and len(openings) < 3:
                    report.findings.append(f"第{night_id}夜狼聊分工不足，开场理由过于重复：variety={len(openings)}")
        if per_night_stances:
            report.wolf_chat_stance_variety = min(len(stances) for stances in per_night_stances.values())
            long_nights = {
                night_id: stances
                for night_id, stances in per_night_stances.items()
                if sum(1 for record in all_wolf_records if record.night_id == night_id) >= 4
            }
            for night_id, stances in long_nights.items():
                if "proposal" not in stances or not ({"support", "final_confirm"} & stances):
                    report.findings.append(f"第{night_id}夜狼聊缺少提案到收口的结构化链路：stances={sorted(stances)}")
        for night_id, target_seats in first_round_targets_by_night.items():
            if len(target_seats) >= 3:
                top_support = max(target_seats.count(seat) for seat in set(target_seats))
                if top_support < 2:
                    report.findings.append(f"第{night_id}夜狼聊缺少同轮协商共识：targets={target_seats}")
                if (
                    night_id == 1
                    and
                    len(target_seats) >= 4
                    and len(set(target_seats)) == 1
                    and not _night_has_forced_power_target(game, night_id, target_seats[0] - 1)
                ):
                    report.findings.append(f"第{night_id}夜狼聊首轮无备刀分歧：targets={target_seats}")
    if game.phase == Phase.WOLF_CHAT and current_night_records:
        night_ids = {record.night_id for record in current_night_records}
        if night_ids != {game.night_id}:
            report.findings.append(f"当前夜狼聊混入其他夜记录：{night_ids}")

    public_snapshot = game.to_snapshot()
    report.findings.extend(_snapshot_actionability_findings(public_snapshot))
    if game.human_player.camp != Camp.WEREWOLF:
        if public_snapshot.wolf_chat_records:
            report.findings.append("非狼人快照泄漏狼聊记录")
        if any(event.phase == "wolf_chat" for event in public_snapshot.events):
            report.findings.append("非狼人快照泄漏狼聊系统播报")

    vote_audits = [audit for audit in game.decision_audits if audit.action in {"vote", "exile_pk_vote"}]
    report.vote_audit_count = len(vote_audits)
    if vote_audits and not any("公开" in audit.reason or "发言" in audit.reason or "票" in audit.reason for audit in vote_audits):
        report.findings.append("投票审计缺少公开证据依据")
    votes_by_day: dict[int, list] = {}
    for vote in game.votes:
        if vote.vote_type == "exile" and vote.vote_round == f"day_{vote.day}_exile":
            votes_by_day.setdefault(vote.day, []).append(vote)
    for day, day_votes in votes_by_day.items():
        target_counts: dict[int, int] = {}
        for vote in day_votes:
            target_counts[vote.target_id] = target_counts.get(vote.target_id, 0) + 1
        day_share = max(target_counts.values()) / len(day_votes)
        report.max_vote_share = max(report.max_vote_share, day_share)
        top_target = max(target_counts, key=lambda target_id: target_counts[target_id])
        if len(day_votes) >= 6 and day_share > 0.72 and not _day_has_seer_wolf_anchor(game, day, top_target):
            report.findings.append(f"第{day}天投票过度集中，缺少自然分歧：max_share={day_share:.2f}")
    return report


def _day_has_seer_wolf_anchor(game, day: int, target_id: int) -> bool:
    """当天若有预言家发言/遗言明确查杀该目标，高集中票属于合理归票。"""
    target_seat = target_id + 1
    patterns = [
        rf"(?:验到|验了|查验|摸了)\s*{target_seat}号[^。！？!?，,；;]{{0,16}}(?:狼人|查杀)",
        rf"{target_seat}号[^。！？!?，,；;]{{0,12}}(?:是|给)?\s*(?:狼人|查杀)",
    ]
    for speech in game.speeches:
        if speech.day != day:
            continue
        if "预言家" not in speech.content and "验" not in speech.content:
            continue
        if any(re.search(pattern, speech.content) for pattern in patterns):
            return True
    return False


def _night_has_forced_power_target(game, night_id: int, target_id: int) -> bool:
    """若前一白天已出现强神职/强信息位锚点，狼队同刀属于合理收束。"""
    if target_id < 0:
        return False
    target_seat = target_id + 1
    relevant_speeches = [speech for speech in game.speeches if speech.day <= max(1, night_id - 1)]
    if not relevant_speeches:
        return False
    direct_claim_patterns = [
        rf"我是预言家",
        rf"我(?:跳|起跳)?预言家",
        rf"我是女巫",
        rf"我是猎人",
    ]
    for speech in relevant_speeches:
        if speech.player_id != target_id:
            continue
        if any(re.search(pattern, speech.content) for pattern in direct_claim_patterns):
            return True
        if any(token in speech.content for token in ["查杀", "金水", "验了", "验到", "验人"]):
            return True
    indirect_anchor_patterns = [
        rf"{target_seat}号[^。！？!?，,；;]{{0,12}}(?:是|给)?\s*(?:狼人|查杀)",
        rf"(?:验到|验了|查验|摸了)\s*{target_seat}号[^。！？!?，,；;]{{0,12}}(?:狼人|查杀|好人|金水)",
        rf"{target_seat}号[^。！？!?，,；;]{{0,12}}(?:像|是)?(?:预言家|女巫|猎人)",
    ]
    if any(any(re.search(pattern, speech.content) for pattern in indirect_anchor_patterns) for speech in relevant_speeches):
        return True
    pressure_mentions = 0
    pressure_tokens = ["投", "票", "出", "压", "归票", "站边", "怀疑", "狼坑", "带队"]
    for speech in relevant_speeches:
        if f"{target_seat}号" not in speech.content:
            continue
        if any(token in speech.content for token in pressure_tokens):
            pressure_mentions += 1
    return pressure_mentions >= 2


def _normalize_wolf_chat_opening(opening: str) -> str:
    """粗归一化狼聊首句，识别多人复用同一理由模板。"""
    opening = re.sub(r"\d+号", "X号", opening)
    opening = re.sub(r"\s+", "", opening)
    for marker in ["我先提主刀", "我偏向今晚动", "我支持把", "我不想分票", "今晚死收益最高"]:
        opening = opening.replace(marker, "主刀")
    return opening[:36]


def _wolf_chat_negates_target(content: str, target_seat_no: int) -> bool:
    """识别“嘴上反对刀 X 号，但结构化目标也是 X 号”的矛盾夜聊。"""
    target = rf"{target_seat_no}号"
    negation = r"(?:先别|暂时别|别急着|不用急着|不要|不建议|不想|不同意|不急着|别)"
    negation_patterns = [
        rf"{negation}[^。！？!?，,；;]{{0,12}}(?:刀|杀|动|奔|压)?{target}",
        rf"{target}[^。！？!?，,；;]{{0,8}}(?:先别|别急|不用急|不急着|不建议|不要)(?:刀|杀|动|奔|压)?",
        rf"不(?:完全)?同意[^。！？!?]{{0,16}}{target}",
    ]
    return any(re.search(pattern, content) for pattern in negation_patterns)


def _wolf_chat_conflicting_commitment(content: str, target_seat_no: int) -> bool:
    """识别“结构化刀 A，但话术收口/不换 B”的矛盾夜聊。"""
    commitment_patterns = [
        r"(?:收口|统一|落点|主刀|刀口)[^。！？!?，,；;]{0,8}(?<!\d)([1-9]|1[0-2])号",
        r"(?<!\d)([1-9]|1[0-2])号[^。！？!?，,；;]{0,4}(?:不换|这刀|落下去)",
    ]
    for pattern in commitment_patterns:
        for raw in re.findall(pattern, content):
            if int(raw) != target_seat_no:
                return True
    return False


def _count_claim_responses(day_speeches) -> int:
    """统计公开身份宣称后，后续发言是否围绕真假/站边/对冲作回应。"""
    response_count = 0
    active_claim_seen = False
    response_tokens = ["真预", "悍跳", "跳预", "预言家", "站边", "验人", "对冲", "认", "不认"]
    for speech in day_speeches:
        content = speech.content
        if active_claim_seen and any(token in content for token in response_tokens):
            response_count += 1
        if _is_seer_claim_text(content):
            active_claim_seen = True
    return response_count


def _repeated_claim_response_templates(day_speeches) -> list[str]:
    """识别预言家宣称后多人复用同一回应模板。"""
    active_claim_seen = False
    buckets: dict[str, int] = {}
    for speech in day_speeches:
        content = speech.content
        if active_claim_seen and any(token in content for token in ["预言家", "真预", "悍跳", "验人", "站边", "验证空间"]):
            bucket = _normalize_claim_response_template(content)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        if _is_seer_claim_text(content):
            active_claim_seen = True
    return [bucket for bucket, count in buckets.items() if count >= 3]


def _normalize_claim_response_template(content: str) -> str:
    """归一化身份宣称回应，号位不同但句式相同也算重复。"""
    content = re.sub(r"\d+号", "X号", content)
    content = re.sub(r"第[一二三四五六七八九十]+天", "第N天", content)
    content = re.sub(r"\s+", "", content)
    return content[:42]


def _looks_like_day_claim_response_template(content: str) -> bool:
    """遗言不应继续使用普通白天身份回应模板。"""
    patterns = [
        r"先留验证空间",
        r"今天先听链路能不能自洽",
        r"跳预以后，今天不是简单信不信的问题",
        r"平民视角最怕假预带票",
        r"这件事先不要被一句话定死",
        r"这轮也别把核心位当普通焦点打",
    ]
    return any(re.search(pattern, content) for pattern in patterns)


def _has_seer_claim(day_speeches) -> bool:
    """是否出现预言家宣称。"""
    return any(_is_seer_claim_text(speech.content) for speech in day_speeches)


def _is_seer_claim_text(content: str) -> bool:
    """统一识别自然语言预言家宣称。"""
    claim_tokens = [
        "我是预言家",
        "我跳预言家",
        "我起跳预言家",
        "我直接跳预言家",
        "我也把身份拍了：预言家",
        "我也把身份拍了:预言家",
        "我拍身份带节奏：预言家",
        "我拍身份带节奏:预言家",
        "我这里是真预视角",
        "预言家牌出来报信息",
        "预言家牌",
        "昨晚我验到",
        "我的夜里信息指向",
        "夜里拿到的好人信息",
        "偏好人信息在我这里成立",
    ]
    return any(token in content for token in claim_tokens)


def _seer_claim_followup_count(day_speeches) -> int:
    """预言家宣称后还剩几条发言；末置位起跳不能强行要求后续回应。"""
    for index, speech in enumerate(day_speeches):
        content = speech.content
        if _is_seer_claim_text(content):
            return len(day_speeches) - index - 1
    return 0


def _count_seer_counterclaims(day_speeches) -> int:
    """统计预言家对跳/对冲场景，作为身份博弈最低门槛。"""
    count = 0
    seer_claim_seen = False
    counter_tokens = [
        "对跳预言家",
        "我也把身份拍了",
        "我也拍预言家",
        "我拍身份",
        "我这里是真预",
        "预言家牌出来报信息",
        "我不认",
        "和他对冲",
        "跳预言家",
        "起跳后",
        "真预面",
        "假预",
    ]
    for speech in day_speeches:
        content = speech.content
        if seer_claim_seen and any(token in content for token in counter_tokens):
            count += 1
        if _is_seer_claim_text(content):
            seer_claim_seen = True
    return count


def _classify_day_angle(content: str) -> str:
    """粗分类白天发言角度，用于发现整桌都在同一种模板里打转。"""
    if any(token in content for token in ["预言家", "验人", "金水", "查杀", "悍跳", "真预", "跳预", "警徽流"]):
        return "claim"
    if any(token in content for token in ["身份", "神职", "平民", "女巫", "猎人", "白痴"]):
        return "role"
    if any(token in content for token in ["改口", "前后", "闭环", "链路", "逻辑", "解释"]):
        return "logic"
    if any(token in content for token in ["站边", "认", "不认", "保"]):
        return "stance"
    if any(token in content for token in ["票", "投", "归票", "补票", "冲票"]):
        return "vote"
    return "pressure"


def _role_strategy_roles(game, day_speeches) -> set[RoleName]:
    """按真实身份检查发言是否体现对应玩法链路，而不是所有人都泛泛点评。"""
    roles: set[RoleName] = set()
    for speech in day_speeches:
        if not (0 <= speech.player_id < len(game.players)):
            continue
        role = game.players[speech.player_id].role
        content = speech.content
        if role == RoleName.SEER and _seer_strategy_signal(content):
            roles.add(role)
        elif role == RoleName.WITCH and _witch_strategy_signal(content):
            roles.add(role)
        elif role == RoleName.HUNTER and _hunter_strategy_signal(content):
            roles.add(role)
        elif role == RoleName.IDIOT and _idiot_strategy_signal(content):
            roles.add(role)
        elif role == RoleName.WEREWOLF and _wolf_strategy_signal(content):
            roles.add(role)
        elif role == RoleName.VILLAGER and _villager_strategy_signal(content):
            roles.add(role)
    return roles


def _seer_strategy_signal(content: str) -> bool:
    return any(token in content for token in ["预言家", "验", "查杀", "金水", "验人链", "夜里信息"])


def _witch_strategy_signal(content: str) -> bool:
    return any(token in content for token in ["女巫", "药", "毒", "救", "药瓶", "轮次"])


def _hunter_strategy_signal(content: str) -> bool:
    return any(token in content for token in ["猎人", "枪", "枪口", "开枪"])


def _idiot_strategy_signal(content: str) -> bool:
    return any(token in content for token in ["白痴", "翻牌", "抗推", "不怕被点", "不怕被推"])


def _wolf_strategy_signal(content: str) -> bool:
    return any(token in content for token in ["预言家", "悍跳", "对跳", "冲票", "收票", "带节奏", "递刀", "票口"])


def _villager_strategy_signal(content: str) -> bool:
    return any(token in content for token in ["普通好人", "普通身份", "平民", "票型", "站边", "补票", "发言"])


def _snapshot_actionability_findings(snapshot) -> list[str]:
    """检查前端是否拿得到当前真人操作所需数据。"""
    findings: list[str] = []
    pending = snapshot.pending_human_action
    if not pending:
        return findings

    target_required_actions = {
        "wolf_chat",
        "night",
        "day_vote",
        "hunter_shot",
        "badge_transfer",
        "sheriff_vote",
    }
    if pending in target_required_actions and not snapshot.human_target_candidates:
        findings.append(f"真人待操作 {pending} 缺少候选目标，前端会卡住")

    if pending == "night" and not snapshot.human_allowed_night_actions:
        findings.append("真人夜晚待操作缺少可选动作")

    if pending == "wolf_chat" and not snapshot.human_is_wolf:
        findings.append("非狼人玩家被标记为狼聊待操作")

    if pending == "hunter_shot" and snapshot.phase != Phase.HUNTER_SHOT:
        findings.append(f"猎人开枪待操作阶段不一致：{snapshot.phase}")

    if pending == "badge_transfer" and snapshot.phase != Phase.BADGE_TRANSFER:
        findings.append(f"警徽移交待操作阶段不一致：{snapshot.phase}")

    if pending == "last_words" and snapshot.phase != Phase.LAST_WORDS:
        findings.append(f"遗言待操作阶段不一致：{snapshot.phase}")

    return findings
