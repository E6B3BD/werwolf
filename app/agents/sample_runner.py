"""本地样局 runner，用于手动/CI 检查 Agent 可玩性。"""
from __future__ import annotations

import asyncio
import argparse
import json
import re

from app.agents.quality import PlayabilityReport, evaluate_playability
from app.agents.runtime import OpenAIAgentRuntime
from app.engine.game import WerwolfGame
from app.engine.models import Camp, HumanNightAction, Phase, PlayerState, RoleName, SpeechRecord


SAMPLE_ROLES = [
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
]


SCRIPT_VARIANTS = [
    {
        # 好人强协同：真预第一天给硬查杀，神民围绕验人链验票型。
        4: "我是预言家，昨晚验了1号是狼人。今天票别散，先看谁还在替1号缓冲。",
        5: "我不拍身份，但1号被查杀以后反应太虚。这个轮次不能让查杀位拖到后面，今天先压1号，明天复盘谁保过他。",
        6: "猎人牌不怕站边，我认这条验人链先走一轮。谁今天强行拆票，谁就进我枪口视野。",
        7: "我不怕被推，但暂时站5号的预言家链，不把票无脑交出去。1号要先回应查杀，后面谁替1号拆票谁就一起进视野。",
        8: "我普通好人先跟验人链验票型，1号如果是狼，后面保他的牌都要回看。",
    },
    {
        # 狼队强误导：悍跳给真预查杀，好人桌面被迫处理真假预。
        1: "我跳预言家，昨晚验5号是狼人。5号如果对跳就是被我验穿，今天别给他喘息空间。",
        4: "我是真预，昨晚验9号是好人。2号这个跳法太急，今天不能只听谁先报身份。",
        5: "我不拍具体身份，但药瓶和票一样都讲轮次收益。2号这条预言家线太急，今天别让单边查杀直接拿走票权。",
        6: "猎人牌不怕被身份线带节奏，我先看谁敢落票还敢承担后果。今天强冲5号的人，都先进我枪口视野。",
        7: "我不怕被点也不怕被抗推。2号和5号都要聊心路，谁拿预言家话题快速做票坑，我就先反看谁。",
        9: "我先不认2号单边预言家。5号和2号都要聊心路，谁急着收票谁更像狼。",
        10: "7号说话有攻击性但不是空喷，真正要看的是谁顺着预言家话题补票。",
    },
    {
        # 混乱分票：真预给狼查杀，但桌面有人怀疑预言家节奏，制造 PK/错票可能。
        4: "我是预言家，昨晚验了3号是狼人，今天票别散，但我也看谁故意把话题打散。",
        5: "我不拍身份，但药瓶轮次不能被假预随便带走。3号要解释，5号也要讲验人心路，今天别无脑同票。",
        6: "猎人牌不怕被点，我今天只看谁敢落票还敢解释后果。3号必须进票型压力。",
        7: "我不怕被点也不怕被抗推。3号我不急着直接打死，5号这个预言家跳法也要验发言质量。",
        11: "我普通好人视角更怕假预言家收票。3号要解释，5号也要讲为什么验这里。",
    },
]

_SAMPLE_SCRIPT_VARIANTS_BY_GAME_ID: dict[str, int] = {}


def build_sample_game(use_live_runtime: bool = False, human_player_id: int = 8, script_variant: int = 0) -> WerwolfGame:
    """构建稳定样局，供质量检查使用。"""
    players = [
        PlayerState(
            id=index,
            name=f"玩家{index + 1}",
            role=role,
            camp=Camp.WEREWOLF if role == RoleName.WEREWOLF else Camp.VILLAGER,
            is_human=index == human_player_id,
        )
        for index, role in enumerate(SAMPLE_ROLES)
    ]
    game = WerwolfGame(player_count=12, human_player_id=human_player_id, day=1, phase=Phase.WOLF_CHAT, players=players)
    runtime = OpenAIAgentRuntime()
    if not use_live_runtime:
        runtime.enabled = False
    game.runtime = runtime
    game.initialize_agent_state()
    _SAMPLE_SCRIPT_VARIANTS_BY_GAME_ID[game.game_id] = script_variant % len(SCRIPT_VARIANTS)
    return game


async def run_sample_game(
    use_live_runtime: bool = False,
    days: int = 1,
    human_player_id: int = 8,
    script_variant: int = 0,
) -> tuple[WerwolfGame, PlayabilityReport]:
    """跑到指定天数的白天放逐后，返回游戏和质量报告。"""
    game = build_sample_game(
        use_live_runtime=use_live_runtime,
        human_player_id=human_player_id,
        script_variant=script_variant,
    )
    completed_days = 0
    completed_day_ids: set[int] = set()
    guard = 0
    while game.phase != Phase.GAME_OVER and completed_days < days and guard < 120:
        guard += 1
        completed_day = await _advance_sample_game_once(game)
        if completed_day and game.day not in completed_day_ids:
            completed_day_ids.add(game.day)
            completed_days += 1
        if not completed_day and game.phase not in {
            Phase.WOLF_CHAT,
            Phase.NIGHT,
            Phase.LAST_WORDS,
            Phase.HUNTER_SHOT,
            Phase.BADGE_TRANSFER,
            Phase.DAY_SPEECH,
            Phase.DAY_VOTE,
            Phase.EXILE_PK_SPEECH,
            Phase.EXILE_PK_VOTE,
            Phase.GAME_OVER,
        }:
            break
    return game, evaluate_playability(game, require_counterclaim=False)


async def run_full_sample_game(
    use_live_runtime: bool = False,
    human_player_id: int = 8,
    script_variant: int = 0,
    max_steps: int = 260,
) -> tuple[WerwolfGame, PlayabilityReport]:
    """跑完整局，用于验证主线流程确实能闭环到胜负。"""
    game = build_sample_game(
        use_live_runtime=use_live_runtime,
        human_player_id=human_player_id,
        script_variant=script_variant,
    )
    for _ in range(max_steps):
        if game.phase == Phase.GAME_OVER:
            break
        await _advance_sample_game_once(game)
    report = evaluate_playability(game, require_counterclaim=True)
    if game.phase != Phase.GAME_OVER:
        report.findings.append(f"完整样局未结束：phase={game.phase.value}, day={game.day}, night_id={game.night_id}")
    if game.phase == Phase.GAME_OVER and not game.winner:
        report.findings.append("完整样局已结束但缺少获胜方")
    return game, report


async def run_playability_matrix(use_live_runtime: bool = False) -> list[tuple[WerwolfGame, PlayabilityReport]]:
    """覆盖多个真人身份和脚本变体，避免单一样局通过但真实体验仍单薄。"""
    cases = [
        (0, 0),  # 狼人真人：验证狼聊、队友隔离和确认刀口
        (4, 0),  # 预言家真人：验证查验私有上下文
        (5, 1),  # 女巫真人：验证刀口可见和用药候选
        (6, 2),  # 猎人真人：验证出局衍生流程
        (7, 1),  # 白痴真人：验证翻牌/投票权路径
        (8, 2),  # 平民真人：验证纯公开信息打法
    ]
    results: list[tuple[WerwolfGame, PlayabilityReport]] = []
    for human_player_id, script_variant in cases:
        game, report = await run_full_sample_game(
            use_live_runtime=use_live_runtime,
            human_player_id=human_player_id,
            script_variant=script_variant,
        )
        _augment_matrix_report(game, report, human_player_id)
        results.append((game, report))
    return results


async def run_balance_matrix(use_live_runtime: bool = False) -> list[tuple[WerwolfGame, PlayabilityReport]]:
    """跑更多固定组合，用于观察 fallback 胜负分布和身份路径稳定性。"""
    results: list[tuple[WerwolfGame, PlayabilityReport]] = []
    for script_variant in range(len(SCRIPT_VARIANTS)):
        for human_player_id in range(len(SAMPLE_ROLES)):
            game, report = await run_full_sample_game(
                use_live_runtime=use_live_runtime,
                human_player_id=human_player_id,
                script_variant=script_variant,
            )
            _augment_matrix_report(game, report, human_player_id)
            results.append((game, report))
    return results


def _augment_matrix_report(game: WerwolfGame, report: PlayabilityReport, human_player_id: int) -> None:
    """矩阵用更硬的可玩性门槛，防止单局指标掩盖身份路径问题。"""
    if not report.completed:
        report.findings.append(f"矩阵样局未闭环：human={human_player_id + 1}号")
    if report.wolf_chat_night_count < 2:
        report.findings.append(f"矩阵样局狼聊夜数不足：human={human_player_id + 1}号")
    if report.day_speech_count < 8:
        report.findings.append(f"矩阵样局白天发言不足：human={human_player_id + 1}号")
    if report.day_angle_variety < 4:
        report.findings.append(f"矩阵样局白天发言角度不足：human={human_player_id + 1}号")
    if game.players[human_player_id].camp != Camp.WEREWOLF:
        snapshot = game.to_snapshot()
        if snapshot.wolf_chat_records or any(item.phase == "wolf_chat" for item in snapshot.visible_timeline):
            report.findings.append(f"矩阵样局非狼人泄漏狼聊：human={human_player_id + 1}号")
    if game.rule_profile.sheriff_enabled or game.rule_profile.guard_enabled:
        report.findings.append("矩阵样局错误启用了扩展规则")


async def _advance_sample_game_once(game: WerwolfGame) -> bool:
    """推进样局一步；返回是否完成一个白天放逐投票。"""
    if game.phase == Phase.WOLF_CHAT:
        for _ in range(12):
            if game.current_speaker_id == game.human_player_id and game.human_player.camp == Camp.WEREWOLF:
                candidates = [player.id for player in game.alive_players() if player.camp != Camp.WEREWOLF]
                target_id = _human_sample_wolf_target(game, candidates)
                action_type = "wolf_confirm" if game.wolf_chat_round >= game.rule_profile.wolf_chat_rounds else "wolf_chat"
                await game.resolve_wolf_chat(
                    HumanNightAction(
                        action_type=action_type,
                        target_id=target_id,
                        chat_content=_human_sample_wolf_chat(game, target_id, action_type),
                    )
                )
            else:
                await game.resolve_wolf_chat(None)
            if game.phase != Phase.WOLF_CHAT:
                break
        return False
    if game.phase == Phase.NIGHT:
        await game.resolve_night(_human_sample_night_action(game))
        return False
    if game.phase == Phase.LAST_WORDS:
        await game.resolve_last_words(_human_sample_last_words(game))
        return False
    if game.phase == Phase.HUNTER_SHOT:
        await game.resolve_hunter_shot(_sample_hunter_shot_target(game))
        return False
    if game.phase == Phase.BADGE_TRANSFER:
        await game.resolve_badge_transfer(None)
        return False
    if game.phase == Phase.DAY_SPEECH:
        if game.current_speaker_id == game.human_player_id:
            await game.resolve_day_speeches(_human_sample_speech(game))
        else:
            scripted = _scripted_sample_speech(game)
            if scripted:
                _force_scripted_sample_speech(game, scripted)
            else:
                await game.resolve_day_speeches("")
        return False
    if game.phase == Phase.DAY_VOTE:
        await game.resolve_votes(_sample_vote_target(game))
        return True
    if game.phase == Phase.EXILE_PK_SPEECH:
        await game._advance_exile_pk_speech("")
        return False
    if game.phase == Phase.EXILE_PK_VOTE and game.exile_pk_candidate_ids:
        await game.resolve_votes(game.exile_pk_candidate_ids[0])
        return True
    return False


def _human_sample_wolf_target(game: WerwolfGame, candidates: list[int]) -> int | None:
    """真人狼人样局盲刀不要稳定落第一个非狼，否则固定板子会过度针对预言家。"""
    if not candidates:
        return None
    if game.wolf_night_plan and game.wolf_night_plan.current_target_id in candidates:
        return game.wolf_night_plan.current_target_id
    offset = (game.human_player_id + game.night_id + len(candidates) // 2) % len(candidates)
    return candidates[offset]


def _human_sample_wolf_chat(game: WerwolfGame, target_id: int | None, action_type: str) -> str:
    """真人狼人样局夜聊。"""
    if target_id is None:
        return "今晚没有合适刀口，我先保留，但不要暴露狼队友关系。"
    if action_type == "wolf_confirm":
        return (
            f"我确认今晚落{target_id + 1}号。这刀能拆带队归票点，"
            "明天我不主动解释死因，只把票型压力往别的位置推。"
        )
    return (
        f"今晚我给{target_id + 1}号做主刀收益，这张牌白天最容易带队归票，"
        "死了以后我们明天口径分开，不要四狼一起踩同一边。"
    )


def _sample_vote_target(game: WerwolfGame) -> int:
    """样局真人投票：基于公开查杀、脚本桌况和票型压力做可重复分歧。"""
    variant = _sample_variant(game)
    if game.phase == Phase.EXILE_PK_VOTE and game.exile_pk_candidate_ids:
        return _sample_pk_vote_target(game, game.exile_pk_candidate_ids, variant)
    candidates = [
        player.id
        for player in game.alive_players()
        if player.id != game.human_player_id
    ]
    if not candidates:
        return game.human_player_id
    human = game.human_player
    split_target = _sample_overconcentration_split_target(game, candidates)
    if split_target is not None:
        return split_target
    witch_claim_target = _latest_human_witch_poison_target(game)
    if human.role == RoleName.WITCH and witch_claim_target is not None:
        pressure = _sample_witch_pressure_target(game, candidates)
        if pressure is not None:
            return pressure
        remaining_wolves = [player.id for player in game.alive_players() if player.camp == Camp.WEREWOLF and player.id in candidates]
        if remaining_wolves and variant in {0, 2}:
            return remaining_wolves[(game.day + game.human_player_id) % len(remaining_wolves)]
        if witch_claim_target in candidates and variant == 0:
            return witch_claim_target
    if human.role == RoleName.HUNTER:
        hunter_pressure = _hunter_sample_suspicion_target(game, candidates)
        if hunter_pressure is not None:
            return hunter_pressure
    claimed = _latest_public_wolf_claim_target(game, candidates)
    if claimed is not None:
        if human.role == RoleName.WITCH and variant == 1:
            counter = _sample_counterclaim_pressure_target(game, claimed, candidates)
            if counter is not None:
                return counter
            return _sample_non_claim_candidate(candidates, claimed)
        if human.role == RoleName.IDIOT and variant == 1:
            counter = _sample_counterclaim_pressure_target(game, claimed, candidates)
            if counter is not None:
                return counter
            return _sample_non_claim_candidate(candidates, claimed)
        if human.camp == Camp.WEREWOLF:
            counter = _sample_counterclaim_pressure_target(game, claimed, candidates)
            if counter is not None:
                return counter
            return claimed if variant == 0 and human.id % 2 == 0 else _sample_non_claim_candidate(candidates, claimed)
        if variant == 1 and human.role in {RoleName.VILLAGER, RoleName.IDIOT}:
            counter = _sample_counterclaim_pressure_target(game, claimed, candidates)
            if counter is not None:
                return counter
        return claimed
    pressure = _sample_vote_pressure(game, candidates)
    if pressure is not None and not (variant == 2 and human.role == RoleName.VILLAGER):
        return pressure
    offset = (game.day + game.night_id + game.human_player_id + variant) % len(candidates)
    return candidates[offset]


def _sample_pk_vote_target(game: WerwolfGame, candidates: list[int], variant: int) -> int:
    """PK 票不再固定投第一个，优先兑现查杀和身份对跳压力。"""
    claimed = _latest_public_wolf_claim_target(game, candidates)
    if claimed is not None:
        return claimed
    pressure = _sample_vote_pressure(game, candidates)
    if pressure is not None:
        return pressure
    return candidates[(game.human_player_id + game.day + variant) % len(candidates)]


def _sample_hunter_shot_target(game: WerwolfGame) -> int | None:
    """样局猎人开枪：优先带走预言家遗言/验人链里的存活查杀，其次看票型。"""
    hunter_id = game.pending_hunter_id
    candidates = [player.id for player in game.alive_players() if player.id != hunter_id]
    if not candidates:
        return None
    anchored = _latest_public_wolf_claim_target(game, candidates)
    if anchored is not None:
        return anchored
    vote_pressure: dict[int, int] = {}
    for vote in game.votes[-24:]:
        if vote.target_id in candidates:
            vote_pressure[vote.target_id] = vote_pressure.get(vote.target_id, 0) + 1
    if vote_pressure:
        return max(vote_pressure, key=lambda target_id: (vote_pressure[target_id], -target_id))
    return candidates[0]


def _hunter_sample_suspicion_target(game: WerwolfGame, candidates: list[int]) -> int | None:
    """猎人样局保留错站边分支，避免所有样本都被验人链单向带飞。"""
    variant = _sample_variant(game)
    if variant != 1:
        return None
    for speech in reversed(game.speeches):
        if speech.player_id in candidates and "预言家" in speech.content:
            return speech.player_id
    return None


def _latest_public_wolf_claim_target(game: WerwolfGame, candidates: list[int]) -> int | None:
    """从公开发言/遗言中提取最近的预言家查杀目标。"""
    for speech in reversed(game.speeches):
        if "预言家" not in speech.content and "验" not in speech.content:
            continue
        for raw in re.findall(r"(\d{1,2})号[^。！？!?，,；;]{0,16}(?:狼人|查杀)", speech.content):
            target_id = int(raw) - 1
            if target_id in candidates:
                return target_id
    return None


def _sample_variant(game: WerwolfGame) -> int:
    """当前样局脚本分支。"""
    return _SAMPLE_SCRIPT_VARIANTS_BY_GAME_ID.get(game.game_id, 0)


def _sample_non_claim_candidate(candidates: list[int], claimed: int) -> int:
    """避开查杀位，制造狼人冲票/好人犹豫的自然分歧。"""
    for candidate in candidates:
        if candidate != claimed:
            return candidate
    return claimed


def _sample_counterclaim_pressure_target(game: WerwolfGame, claimed: int, candidates: list[int]) -> int | None:
    """多预/疑似悍跳时，找出可被反打的预言家宣称者。"""
    claimers: list[int] = []
    for speech in game.speeches:
        if speech.player_id not in candidates:
            continue
        if "预言家" not in speech.content and "跳预" not in speech.content and "真预" not in speech.content:
            continue
        claimers.append(speech.player_id)
    if not claimers:
        return None
    for claimer in reversed(claimers):
        if claimer != claimed:
            return claimer
    return claimers[-1]


def _sample_vote_pressure(game: WerwolfGame, candidates: list[int]) -> int | None:
    """从最近公开发言提到频率里找票型压力位。"""
    pressure: dict[int, int] = {}
    for speech in game.speeches[-18:]:
        for raw in re.findall(r"(\d{1,2})号", speech.content):
            target_id = int(raw) - 1
            if target_id in candidates:
                pressure[target_id] = pressure.get(target_id, 0) + 1
    if not pressure:
        return None
    return max(pressure, key=lambda target_id: (pressure[target_id], -target_id))


def _sample_overconcentration_split_target(game: WerwolfGame, candidates: list[int]) -> int | None:
    """样局中若当前轮已明显单点集中，后置位主动分票保持自然博弈。"""
    current_round = f"day_{game.day}_exile"
    day_votes = [vote for vote in game.votes if vote.vote_type == "exile" and vote.vote_round == current_round]
    if len(day_votes) < 4:
        return None
    counts: dict[int, int] = {}
    for vote in day_votes:
        counts[vote.target_id] = counts.get(vote.target_id, 0) + 1
    top_target = max(counts, key=lambda target_id: counts[target_id])
    if counts[top_target] / len(day_votes) < 0.55:
        return None
    anchored = _latest_public_wolf_claim_target(game, candidates)
    if anchored == top_target:
        return None
    pressure: dict[int, int] = {}
    for speech in game.speeches[-18:]:
        for raw in re.findall(r"(\d{1,2})号", speech.content):
            target_id = int(raw) - 1
            if target_id in candidates and target_id != top_target:
                pressure[target_id] = pressure.get(target_id, 0) + 1
    if pressure:
        return max(pressure, key=lambda target_id: (pressure[target_id], -target_id))
    for candidate in candidates:
        if candidate != top_target:
            return candidate
    return None


def _sample_witch_pressure_target(game: WerwolfGame, candidates: list[int]) -> int | None:
    """女巫公开后优先压持续推女巫/投女巫的位置。"""
    pressure: dict[int, int] = {}
    for vote in game.votes[-24:]:
        if vote.target_id == game.human_player_id and vote.voter_id in candidates:
            pressure[vote.voter_id] = pressure.get(vote.voter_id, 0) + 3
    for speech in game.speeches[-18:]:
        if speech.player_id not in candidates:
            continue
        if "女巫" in speech.content and any(token in speech.content for token in ["不认", "压", "出", "票", "抗推"]):
            pressure[speech.player_id] = pressure.get(speech.player_id, 0) + 2
    if not pressure:
        return None
    return max(pressure, key=lambda player_id: (pressure[player_id], -player_id))


def _human_sample_last_words(game: WerwolfGame) -> str:
    """真人样局遗言：神职死亡时必须把关键私有信息留到桌上。"""
    player_id = game.current_exile_target_id
    if player_id is None or player_id != game.human_player_id:
        return ""
    human = game.human_player
    if human.role == RoleName.SEER:
        inspection = _latest_human_seer_inspection(game)
        if inspection:
            target_seat, result = inspection
            return f"我遗言报清楚：我是预言家，昨晚验到{target_seat}号是{result}。后面按这个验人去看站边和票型。"
        return "我遗言报身份：我是预言家，后面重点看谁借我出局改票。"
    if human.role == RoleName.WITCH:
        return "我遗言不拍太多，重点看谁一直借神职压力收票，药瓶轮次你们自己盘。"
    if human.role == RoleName.HUNTER:
        return "我遗言留一句，别只看谁声音大，看谁在关键票型里补刀。"
    if human.role == RoleName.IDIOT:
        return "我遗言重点点票型，谁把我当抗推位硬塞，谁身份最差。"
    return "我遗言结束，重点看票型别看情绪。"


def _human_sample_night_action(game: WerwolfGame) -> HumanNightAction:
    """样局真人夜间动作，覆盖真实玩家身份技能路径。"""
    human = game.human_player
    variant = _sample_variant(game)
    if not human.alive:
        return HumanNightAction(action_type="skip", target_id=None)
    if human.role == RoleName.SEER:
        candidates = [player.id for player in game.alive_players() if player.id != human.id]
        inspected = {fact.target_id for fact in game.seer_inspection_facts if fact.seer_id == human.id}
        candidates = [player_id for player_id in candidates if player_id not in inspected] or candidates
        wolf_candidates = [player_id for player_id in candidates if game.players[player_id].camp == Camp.WEREWOLF]
        good_candidates = [player_id for player_id in candidates if game.players[player_id].camp != Camp.WEREWOLF]
        if variant == 1 and game.day <= 1 and good_candidates:
            target_id = good_candidates[(human.id + game.night_id) % len(good_candidates)]
        elif wolf_candidates:
            target_id = wolf_candidates[(variant + game.night_id + human.id) % len(wolf_candidates)]
        else:
            target_id = candidates[(variant + game.night_id) % len(candidates)] if candidates else None
        return HumanNightAction(action_type="inspect", target_id=target_id)
    if human.role == RoleName.WITCH:
        alive_non_self = [player.id for player in game.alive_players() if player.id != human.id]
        public_claim = _latest_public_wolf_claim_target(game, alive_non_self)
        if game.witch_state.poison_available and public_claim is not None and game.day >= 2:
            if variant != 1:
                return HumanNightAction(action_type="poison", target_id=public_claim)
        if game.witch_state.poison_available and variant == 2 and game.day >= 3:
            wolves = [player.id for player in game.alive_players() if player.camp == Camp.WEREWOLF]
            if wolves:
                return HumanNightAction(action_type="poison", target_id=wolves[(game.day + human.id) % len(wolves)])
        if (
            game.witch_state.save_available
            and game.wolf_consensus_target_id is not None
            and game._witch_can_save_target(human.id, game.wolf_consensus_target_id)
            and (
                (variant == 0 and (game.day >= 2 or game.players[game.wolf_consensus_target_id].role != RoleName.VILLAGER))
                or (variant == 2 and game.day <= 1)
            )
        ):
            return HumanNightAction(action_type="save", target_id=game.wolf_consensus_target_id)
        if game.witch_state.poison_available and game.day >= 3 and variant == 0:
            candidates = [player.id for player in game.alive_players() if player.id != human.id and player.camp == Camp.WEREWOLF]
            if candidates:
                return HumanNightAction(action_type="poison", target_id=candidates[(game.day + human.id) % len(candidates)])
        return HumanNightAction(action_type="skip", target_id=None)
    return HumanNightAction(action_type="skip", target_id=None)


def _scripted_sample_speech(game: WerwolfGame) -> str:
    """第一天样局脚本只在真实白天发言阶段注入，不能泄漏给首夜狼队。"""
    variant = _sample_variant(game)
    if game.day != 1:
        return ""
    return SCRIPT_VARIANTS[variant].get(game.current_speaker_id, "")


def _force_scripted_sample_speech(game: WerwolfGame, content: str) -> None:
    """样局专用：把指定 AI 玩家脚本真实写入桌面，而不是被 engine fallback 忽略。"""
    speaker_id = game.current_speaker_id
    if speaker_id is None:
        game.phase = Phase.DAY_VOTE
        return
    speaker = game.players[speaker_id]
    speaker.last_speech = content
    game._update_agent_state_after_public_action(speaker, content, "day_speech")
    game.speeches.append(
        SpeechRecord(
            day=game.day,
            player_id=speaker.id,
            player_name=speaker.name,
            content=content,
            speech_type="day",
        )
    )
    game._record_message(
        "talk",
        "day_speech",
        content,
        speaker=speaker,
        action="day_speech",
        turn_index=game.speech_cursor,
    )
    game._broadcast_public_observation(
        f"{speaker.name}（{speaker.id + 1}号）白天发言：{content}",
        phase="day_speech",
        data={"speaker_id": speaker.id, "speech_type": "day"},
    )
    game.speech_cursor += 1
    if game.current_speaker_id is None:
        game._add_event("speech", f"第 {game.day} 天白天发言结束。")
        game.phase = Phase.DAY_VOTE
        game.pending_human_action = "day_vote" if game.human_player.alive and game.human_player.can_vote else None
    else:
        game.pending_human_action = "day_speech" if game.current_speaker_id == game.human_player_id else None


def _human_sample_speech(game: WerwolfGame) -> str:
    """真人样局占位发言，按身份给出不同风格，避免样局质量靠单一句子。"""
    role = game.human_player.role
    if role == RoleName.WEREWOLF:
        if _sample_variant(game) == 1 and game.day == 1:
            return "我跳预言家，昨晚验5号是狼人。5号如果对跳就是被我验穿，今天别给他喘息空间。"
        if game.day <= 2 and game.human_player_id % 3 == 2:
            return "我也拍预言家，昨晚验5号是狼人。真预不会只把票口压成单线，今天先把两个预言家的验人和站边全部摊开。"
        lanes = [
            "我按票型说，今天谁借预言家牌强收票我会先记；4号和7号里，谁跟票更急谁问题更大。",
            "反应速度这块我更看4号。预言家线可以盘，但不能变成谁嗓门大谁拿票权。",
            "补票动作上我会盯7号。现在最脏的不是站错边，是等别人冲完再补安全票。",
            "我按轮次收益盘，今天别把所有票都交给单一身份口径；4号要讲票意，7号要讲站边原因。",
            "我只抓站边兑现。谁嘴上说保留，落票却跟得最快，谁就是我今天最想验票型的位置。",
        ]
        return lanes[(game.day + game.human_player_id + game.night_id) % len(lanes)]
    if role == RoleName.SEER:
        inspection = _latest_human_seer_inspection(game)
        if inspection:
            target_seat, result = inspection
            if result == "狼人":
                return f"我是预言家，昨晚验到{target_seat}号是狼人。今天票别散，谁还在替{target_seat}号缓冲，谁就进我的第二狼坑。"
            return f"我是预言家，昨晚验到{target_seat}号是好人。今天先别动{target_seat}号，重点看谁借这张金水做抗推。"
        return "我是预言家，今天重点不是喊身份，而是把我的验人结果变成能执行的票。"
    if role == RoleName.WITCH:
        poison_target = _latest_human_witch_poison_target(game)
        if poison_target is not None and _sample_variant(game) in {0, 2}:
            return f"我这里拍女巫，昨晚毒了{poison_target + 1}号。今天别再被悍跳预言家牵着走，先按药瓶信息和票型把保狼的人清出来。"
        if game.day >= 2:
            saved = _latest_human_witch_saved_target(game)
            variants = [
                f"我女巫牌必须拍了，{'我救过' + str(saved + 1) + '号，' if saved is not None else ''}现在有人借预言家线强推我，这个收益不对。今天先压推女巫最急的位置，别把神职当抗推。",
                f"我这里是女巫视角，药瓶轮次不能被悍跳预言家随便带走。今天我会反压收票最急的预言家线，不把票无脑交给单边查杀。",
                f"女巫牌摊开说：我不接受好人把神职当安全抗推。今天先复盘谁连续推我，再看那条预言家线是不是被狼接管了。",
            ]
            return variants[(game.day + game.night_id + _sample_variant(game)) % len(variants)]
        if game.day >= 2 and _sample_variant(game) == 2:
            return "我女巫牌给压力：前面有人一直借预言家线收票，这不是单纯站错边。今天先看谁在死因以后还继续保狼坑。"
        return "我会按轮次收益看票，不跟纯情绪。谁把弱理由讲成必出，我就先看谁。"
    if role == RoleName.HUNTER:
        if _sample_variant(game) == 1:
            return "猎人牌不等于无脑站边。预言家线我会听，但今天谁借身份强行收票，我会先把枪口和票意压过去。"
        return "我不怕被压，但今天谁只带节奏不承担票型后果，我会把枪口先记过去。"
    if role == RoleName.IDIOT:
        if _sample_variant(game) == 1:
            variants = [
                "我这轮不想被单预言家绑票。查杀位可以听，但我更怀疑那个急着收口的人，今天我会反压预言家线。",
                "我不怕被点，但不接受全场只按一个验人口径走。谁现在催着所有人同票，谁更像在借身份做局。",
                "今天我看站边速度，不按嗓门站队。查杀位要解释，预言家线也要经得起票型复盘。",
            ]
            return variants[(game.day + game.night_id + game.human_player_id) % len(variants)]
        return "我先不装信息，今天看谁制造假共识，尤其谁在后面补刀最急。"
    return "我今天先看4号和7号，谁投票跟得最急我就记谁。"


def _latest_human_seer_inspection(game: WerwolfGame) -> tuple[int, str] | None:
    """真人预言家最近验人。真实流程读 typed fact，旧测试种子读 data 兼容。"""
    fact = next((item for item in reversed(game.seer_inspection_facts) if item.seer_id == game.human_player_id), None)
    if fact:
        return fact.target_seat_no, fact.result
    memory = game.agent_memories.get(game.human_player_id)
    for item in reversed(memory.private_observations if memory else []):
        target_seat = item.data.get("target_seat_no")
        result = item.data.get("result")
        if isinstance(target_seat, int) and result in {"狼人", "好人"}:
            return target_seat, result
    return None


def _latest_human_witch_poison_target(game: WerwolfGame) -> int | None:
    """真人女巫最近一次毒人目标。"""
    fact = next((item for item in reversed(game.witch_action_facts) if item.witch_id == game.human_player_id), None)
    if fact and fact.poison_target_id is not None:
        return fact.poison_target_id
    return None


def _latest_human_witch_saved_target(game: WerwolfGame) -> int | None:
    """真人女巫最近一次救人目标。"""
    fact = next((item for item in reversed(game.witch_action_facts) if item.witch_id == game.human_player_id), None)
    if fact and fact.saved_target_id is not None:
        return fact.saved_target_id
    return None


def _report_dict(report: PlayabilityReport) -> dict:
    """转成 CLI 友好的 dict。"""
    return {
        "passed": report.passed,
        "completed": report.completed,
        "winner": report.winner,
        "phases_covered": report.phases_covered,
        "day_speech_count": report.day_speech_count,
        "unique_speech_ratio": round(report.unique_speech_ratio, 2),
        "concrete_speech_count": report.concrete_speech_count,
        "claim_response_count": report.claim_response_count,
        "seer_counterclaim_count": report.seer_counterclaim_count,
        "vote_intent_speech_count": report.vote_intent_speech_count,
        "day_angle_variety": report.day_angle_variety,
        "role_strategy_signal_count": report.role_strategy_signal_count,
        "role_strategy_roles": report.role_strategy_roles,
        "wolf_chat_count": report.wolf_chat_count,
        "night_count": report.night_count,
        "wolf_chat_night_count": report.wolf_chat_night_count,
        "wolf_chat_roleplay_variety": report.wolf_chat_roleplay_variety,
        "wolf_chat_stance_variety": report.wolf_chat_stance_variety,
        "vote_audit_count": report.vote_audit_count,
        "max_vote_share": round(report.max_vote_share, 2),
        "findings": report.findings,
    }


def _matrix_payload(results: list[tuple[WerwolfGame, PlayabilityReport]]) -> dict:
    """转成 CLI 输出。"""
    cases = []
    findings: list[str] = []
    for game, report in results:
        case = {
            "human": game.human_player_id + 1,
            "role": game.human_player.role.value,
            "passed": report.passed,
            "completed": report.completed,
            "winner": report.winner,
            "day_speech_count": report.day_speech_count,
            "wolf_chat_night_count": report.wolf_chat_night_count,
            "day_angle_variety": report.day_angle_variety,
            "seer_counterclaim_count": report.seer_counterclaim_count,
            "findings": report.findings,
        }
        cases.append(case)
        findings.extend(f"{case['human']}号{case['role']}: {finding}" for finding in report.findings)
    completed_winners = {case["winner"] for case in cases if case["completed"] and case["winner"]}
    return {
        "passed": not findings,
        "case_count": len(cases),
        "cases": cases,
        "winner_variety": len(completed_winners),
        "findings": findings,
    }


def _balance_payload(results: list[tuple[WerwolfGame, PlayabilityReport]]) -> dict:
    """输出更宽的可玩性/平衡观测报告。"""
    winner_counts: dict[str, int] = {}
    role_counts: dict[str, dict[str, int]] = {}
    findings: list[str] = []
    completed = 0
    for game, report in results:
        winner = report.winner or "未结束"
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        if report.completed:
            completed += 1
        role_name = game.human_player.role.value
        role_counts.setdefault(role_name, {})
        role_counts[role_name][winner] = role_counts[role_name].get(winner, 0) + 1
        findings.extend(f"{game.human_player_id + 1}号{role_name}: {finding}" for finding in report.findings)
    completed_winners = {report.winner for _, report in results if report.completed and report.winner}
    if completed != len(results):
        findings.append(f"平衡样本存在未闭环局：{completed}/{len(results)}")
    if len(completed_winners) < 2:
        findings.append("平衡样本胜负仍为单边，需要继续调策略。")
    for role_name, counts in role_counts.items():
        total = sum(counts.values())
        if total >= len(SCRIPT_VARIANTS) and len(counts) < 2:
            findings.append(f"真人{role_name}样本胜负单边：{counts}")
    return {
        "passed": not findings,
        "case_count": len(results),
        "completed_count": completed,
        "winner_counts": winner_counts,
        "human_role_winner_counts": role_counts,
        "winner_variety": len(completed_winners),
        "findings": findings,
    }


def _sample_excerpt(game: WerwolfGame) -> dict:
    """输出少量样局摘录，便于人工判断趣味性。"""
    return {
        "wolf_chat": [
            {
                "speaker": record.speaker_seat_no,
                "target": record.proposed_target_seat_no,
                "content": record.content,
            }
            for record in game.wolf_chat_records[:6]
        ],
        "day_speeches": [
            {
                "speaker": speech.player_id + 1,
                "content": speech.content,
            }
            for speech in game.speeches
            if speech.speech_type == "day"
        ][:8],
        "votes": [
            {
                "voter": vote.voter_id + 1,
                "target": vote.target_id + 1,
                "round": vote.vote_round,
            }
            for vote in game.votes
            if vote.vote_type == "exile"
        ][:12],
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Run a Werwolf agent playability sample.")
    parser.add_argument("--live", action="store_true", help="使用当前 OpenAI runtime；默认使用本地 fallback，避免 CI 依赖网络。")
    parser.add_argument("--days", type=int, default=1, choices=(1, 2, 3), help="运行到第几天白天放逐后。")
    parser.add_argument("--full", action="store_true", help="跑完整局直到胜负，用于主线流程闭环验证。")
    parser.add_argument("--matrix", action="store_true", help="跑多身份完整样局矩阵，验证可玩性稳定性。")
    parser.add_argument("--balance", action="store_true", help="跑更宽的固定组合平衡观测矩阵。")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告。")
    parser.add_argument("--no-fail", action="store_true", help="质量失败时仍返回 0。")
    args = parser.parse_args(argv)

    if args.matrix:
        results = asyncio.run(run_playability_matrix(use_live_runtime=args.live))
        payload = {"matrix": _matrix_payload(results)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"matrix_passed={payload['matrix']['passed']}")
            print(f"case_count={payload['matrix']['case_count']}")
            for finding in payload["matrix"]["findings"]:
                print(f"finding={finding}")
        return 0 if payload["matrix"]["passed"] or args.no_fail else 1

    if args.balance:
        results = asyncio.run(run_balance_matrix(use_live_runtime=args.live))
        payload = {"balance": _balance_payload(results)}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"balance_passed={payload['balance']['passed']}")
            print(f"case_count={payload['balance']['case_count']}")
            print(f"winner_counts={payload['balance']['winner_counts']}")
            for finding in payload["balance"]["findings"]:
                print(f"finding={finding}")
        return 0 if payload["balance"]["passed"] or args.no_fail else 1

    if args.full:
        game, report = asyncio.run(run_full_sample_game(use_live_runtime=args.live))
    else:
        game, report = asyncio.run(run_sample_game(use_live_runtime=args.live, days=args.days))
    payload = {"report": _report_dict(report), "excerpt": _sample_excerpt(game)}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload["report"].items():
            if key == "findings":
                continue
            print(f"{key}={value}")
        for finding in report.findings:
            print(f"finding={finding}")
        print("excerpt_wolf_chat=" + " | ".join(item["content"] for item in payload["excerpt"]["wolf_chat"][:3]))
        print("excerpt_day_speeches=" + " | ".join(item["content"] for item in payload["excerpt"]["day_speeches"][:3]))
    return 0 if report.passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
