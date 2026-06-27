from __future__ import annotations

import contextlib
import io
import random
import re
import subprocess
import time
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.agents.quality import AI_LEAK_TOKENS, PRIVATE_LEAK_TOKENS, evaluate_playability
from app.agents.runtime import AIContext
from app.agents.advisor import advise
from app.agents.prompts import PERSONA_POOL
from app.agents.runtime import OpenAIAgentRuntime
from app.agents.sample_runner import _balance_payload, _human_sample_last_words, _human_sample_speech, _human_sample_wolf_target, _report_dict, _sample_excerpt, _sample_hunter_shot_target, build_sample_game, main as sample_runner_main, run_balance_matrix, run_full_sample_game, run_playability_matrix, run_sample_game
from app.api.routes import (
    CreateGameRequest,
    NightRequest,
    SheriffRequest,
    SpeechRequest,
    VoteRequest,
    create_game as api_create_game,
    game_manager,
    get_game as api_get_game,
    resolve_badge as api_resolve_badge,
    resolve_hunter_shot as api_resolve_hunter_shot,
    resolve_last_words as api_resolve_last_words,
    resolve_night as api_resolve_night,
    resolve_sheriff as api_resolve_sheriff,
    resolve_speech as api_resolve_speech,
    resolve_vote as api_resolve_vote,
    resolve_wolf_chat as api_resolve_wolf_chat,
)
from app.engine.game import PERSONA_STYLES, WerwolfGame
from app.engine.models import AgentDecision, AgentVisibleContext, Camp, HumanNightAction, Phase, PlayerState, PrivateObservation, PublicClaimEvidence, ROLE_CONFIGS, RoleName, SeatRef, SheriffAction, SpeechRecord, VoteRecord, WolfChatRecord, WolfNightPlan
from app.main import app


class FakeRuntime:
    def __init__(
        self,
        targets: dict[int, int | None] | None = None,
        strict: bool = True,
        contents: dict[str, str] | None = None,
    ) -> None:
        self.targets = targets or {}
        self.strict = strict
        self.contents = contents or {}
        self.contexts = []

    async def decide(self, context):
        self.contexts.append(context)
        target_id = self.targets.get(context.player_id)
        if self.strict and target_id is not None and target_id not in context.allowed_target_ids:
            target_id = context.allowed_target_ids[0] if context.allowed_target_ids else None
        return AgentDecision(
            action="night_action",
            target_id=target_id,
            content=self.contents.get(context.phase, f"{context.player_id}号建议。"),
            reason="test",
        )


class ExplodingRuntime(OpenAIAgentRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True

    async def decide(self, context):
        if not self.enabled:
            return self._fallback_decision(context)
        raise AssertionError("auto fallback should not call live runtime")




def make_player(player_id: int, role: RoleName) -> PlayerState:
    return PlayerState(
        id=player_id,
        name=f"玩家{player_id + 1}",
        role=role,
        camp=Camp.WEREWOLF if role == RoleName.WEREWOLF else Camp.VILLAGER,
        is_human=False,
    )


def make_game(roles: list[RoleName], human_player_id: int = 0) -> WerwolfGame:
    players = [make_player(idx, role) for idx, role in enumerate(roles)]
    players[human_player_id].is_human = True
    game = WerwolfGame(
        player_count=len(players),
        human_player_id=human_player_id,
        day=1,
        phase=Phase.WOLF_CHAT,
        players=players,
    )
    game.runtime = FakeRuntime()
    game.initialize_agent_state()
    return game


def assert_non_ai_table_speech(contents: list[str]) -> None:
    assert contents
    for content in contents:
        assert not any(token in content for token in AI_LEAK_TOKENS), content
        assert re.search(r"\d+号", content), content
        assert len(content.strip()) >= 18, content


def assert_wolf_chat_quality(records: list[WolfChatRecord]) -> None:
    assert records
    for record in records:
        assert record.night_id >= 1
        assert record.proposed_target_seat_no is not None
        assert f"{record.proposed_target_seat_no}号" in record.content
        assert not any(token in record.content for token in AI_LEAK_TOKENS), record.content
        assert record.is_valid_target


def test_default_12_player_role_pool_uses_idiot_not_guard() -> None:
    roles = ROLE_CONFIGS[12]

    assert roles.count(RoleName.WEREWOLF) == 4
    assert roles.count(RoleName.SEER) == 1
    assert roles.count(RoleName.WITCH) == 1
    assert roles.count(RoleName.HUNTER) == 1
    assert roles.count(RoleName.IDIOT) == 1
    assert roles.count(RoleName.VILLAGER) == 4
    assert RoleName.GUARD not in roles


def test_frontend_pending_action_display_uses_player_facing_labels() -> None:
    app_js = Path("app/static/app.js").read_text(encoding="utf-8")

    assert "formatPendingAction(snapshot.pending_human_action)" in app_js
    assert "当前需要执行：${snapshot.pending_human_action}" not in app_js
    for action in [
        "wolf_chat",
        "night",
        "day_speech",
        "day_vote",
        "exile_pk_speech",
        "exile_pk_vote",
        "last_words",
        "hunter_shot",
        "badge_transfer",
        "choose_speech_order",
        "sheriff_election",
        "sheriff_speech",
        "sheriff_vote",
        "sheriff_pk_speech",
        "sheriff_pk_vote",
    ]:
        assert re.search(rf"{action}:\s*\"[^\"]+\"", app_js), action


def test_frontend_template_provides_all_js_dom_ids() -> None:
    html = Path("app/templates/index.html").read_text(encoding="utf-8")
    app_js = Path("app/static/app.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_refs = set(re.findall(r'getElementById\("([^"]+)"\)', app_js))

    assert js_refs - html_ids == set()


def test_frontend_uses_rule_flags_to_hide_extensions_by_default() -> None:
    app_js = Path("app/static/app.js").read_text(encoding="utf-8")

    assert "snapshot.sheriff_enabled" in app_js
    assert "snapshot?.guard_enabled" in app_js


def test_root_page_and_static_assets_load_with_expected_contract() -> None:
    client = TestClient(app)

    html = client.get("/")
    app_js = client.get("/static/app.js")
    style = client.get("/static/style.css")

    assert html.status_code == 200
    assert app_js.status_code == 200
    assert style.status_code == 200
    assert 'id="createGameBtn"' in html.text
    assert 'id="wolfConfirmBtn"' in html.text
    assert "function applySnapshot" in app_js.text
    assert "visible_timeline" in app_js.text


def test_create_game_snapshot_includes_rule_flags_and_default_extensions_off() -> None:
    client = TestClient(app)

    response = client.post("/api/games", json={"player_count": 12})
    response.raise_for_status()
    snapshot = response.json()

    assert snapshot["sheriff_enabled"] is False
    assert snapshot["guard_enabled"] is False
    assert snapshot["sheriff_candidates"] == []
    assert "guard" not in snapshot["human_allowed_night_actions"]


def test_http_snapshot_flow_exposes_actionable_frontend_contract() -> None:
    client = TestClient(app)
    snapshot = client.post("/api/games", json={"player_count": 12}).json()
    game_manager.get_game(snapshot["game_id"]).runtime.enabled = False

    seen_phases = {snapshot["phase"]}
    seen_pending: set[str] = set()
    for _ in range(120):
        _assert_http_snapshot_frontend_contract(snapshot)
        pending = snapshot.get("pending_human_action")
        if pending:
            seen_pending.add(pending)
        game_id = snapshot["game_id"]
        if pending == "wolf_chat":
            target_id = snapshot["human_target_candidates"][0]
            snapshot = client.post(
                f"/api/games/{game_id}/wolf-chat",
                json={
                    "action_type": "wolf_confirm",
                    "target_id": target_id,
                    "chat_content": f"我确认刀{target_id + 1}号，先拆带队位置。",
                },
            ).json()
        elif pending == "night":
            action = snapshot["human_allowed_night_actions"][0]
            target_id = snapshot["human_target_candidates"][0] if snapshot["human_target_candidates"] and action != "skip" else None
            snapshot = client.post(
                f"/api/games/{game_id}/night",
                json={"action_type": action, "target_id": target_id},
            ).json()
        elif pending in {"day_speech", "exile_pk_speech"}:
            snapshot = client.post(
                f"/api/games/{game_id}/speech",
                json={"content": "我这轮先给明确态度，谁借公开信息硬收票，我就先看谁。"},
            ).json()
        elif pending == "day_vote":
            snapshot = client.post(
                f"/api/games/{game_id}/vote",
                json={"target_id": snapshot["human_target_candidates"][0]},
            ).json()
        elif pending == "last_words":
            snapshot = client.post(
                f"/api/games/{game_id}/last-words",
                json={"content": "我的遗言只留一个重点，回头看谁在我身上补票最急。"},
            ).json()
        elif pending == "hunter_shot":
            snapshot = client.post(
                f"/api/games/{game_id}/hunter-shot",
                json={"target_id": snapshot["human_target_candidates"][0]},
            ).json()
        else:
            snapshot = client.get(f"/api/games/{game_id}").json()
        seen_phases.add(snapshot["phase"])
        if {"wolf_chat", "night", "day_speech"}.issubset(seen_phases) and (
            "day_vote" in seen_phases or snapshot["phase"] == "game_over"
        ):
            break

    assert "wolf_chat" in seen_phases
    assert "night" in seen_phases
    assert "day_speech" in seen_phases
    assert len(seen_phases) >= 4
    assert seen_pending or snapshot["phase"] == "game_over"


def test_http_wolf_player_frontend_view_keeps_current_night_only() -> None:
    client = TestClient(app)
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)
    game.runtime.enabled = False
    game.game_id = "http_wolf_frontend_view"
    game_manager._games[game.game_id] = game
    try:
        first = client.get(f"/api/games/{game.game_id}").json()
        assert first["human_is_wolf"] is True
        assert first["wolf_teammate_ids"] == [1]
        assert 0 not in first["human_target_candidates"]
        assert 1 not in first["human_target_candidates"]
        assert "狼队友：2号" in first["human_private_context"]

        confirm = client.post(
            f"/api/games/{game.game_id}/wolf-chat",
            json={
                "action_type": "wolf_confirm",
                "target_id": 2,
                "chat_content": "我确认刀3号，先拆可能带队的验人位置。",
            },
        ).json()
        assert confirm["phase"] == "night"
        assert confirm["wolf_night_plan"]["current_target_id"] == 2
        assert [event["occurrence_key"] for event in confirm["events"] if event["occurrence_key"] == "wolf_chat_final:1"] == ["wolf_chat_final:1"]
        assert [item["occurrence_key"] for item in confirm["visible_timeline"] if item["occurrence_key"] == "wolf_chat_final:1"] == ["wolf_chat_final:1"]

        game._advance_to_next_day()
        second = client.get(f"/api/games/{game.game_id}").json()
        assert second["phase"] == "wolf_chat"
        assert second["night_id"] == 2
        assert second["wolf_chat_records"] == []
        assert not any(event["occurrence_key"] == "wolf_chat_final:1" for event in second["events"])
        assert not any(item["occurrence_key"] == "wolf_chat_final:1" for item in second["visible_timeline"])
    finally:
        game_manager._games.pop(game.game_id, None)


def test_frontend_dom_harness_exercises_action_panel_and_visibility() -> None:
    result = subprocess.run(
        ["node", "tests/frontend_dom_harness.js"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS frontend DOM harness" in result.stdout


def _assert_http_snapshot_frontend_contract(snapshot: dict) -> None:
    assert snapshot["human_private_context"]
    assert f"你是 {snapshot['human_player_id'] + 1} 号位" in snapshot["human_private_context"]
    assert f"身份：{snapshot['human_role']}" in snapshot["human_private_context"]
    assert "阵营：" in snapshot["human_private_context"]
    assert "状态：" in snapshot["human_private_context"]
    assert isinstance(snapshot["visible_timeline"], list)
    assert snapshot["sheriff_enabled"] is False
    assert snapshot["guard_enabled"] is False
    assert "guard" not in snapshot["human_allowed_night_actions"]
    assert snapshot["pending_human_action"] not in {"sheriff_election", "sheriff_speech", "sheriff_vote", "badge_transfer", "choose_speech_order"}
    if not snapshot["human_is_wolf"]:
        assert snapshot["wolf_chat_records"] == []
        assert not any(item["phase"] == "wolf_chat" and item["visibility"] == "wolf" for item in snapshot["visible_timeline"])
        assert not any(event["phase"] == "wolf_chat" for event in snapshot["events"])
        assert "狼队友" not in snapshot["human_private_context"]
    else:
        assert all(target_id not in snapshot["wolf_teammate_ids"] for target_id in snapshot["human_target_candidates"])
        current_night = snapshot["night_id"]
        assert all(record["night_id"] == current_night for record in snapshot["wolf_chat_records"])
        assert all(
            item["night_id"] == current_night
            for item in snapshot["visible_timeline"]
            if item["phase"] == "wolf_chat"
        )
    wolf_final_keys = [
        item["occurrence_key"]
        for item in snapshot["visible_timeline"]
        if item["phase"] == "wolf_chat" and "wolf_chat_final" in item["occurrence_key"]
    ]
    assert len(wolf_final_keys) == len(set(wolf_final_keys))
    occurrence_keys = [
        item["occurrence_key"]
        for item in snapshot["visible_timeline"]
        if item.get("occurrence_key")
    ]
    assert len(occurrence_keys) == len(set(occurrence_keys))


def test_persona_pool_has_20_styles_and_engine_uses_it() -> None:
    assert len(PERSONA_POOL) >= 20
    assert PERSONA_STYLES == list(PERSONA_POOL)


def test_wolf_private_context_has_identity_teammates_and_legal_targets() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    context = game._player_private_context(game.players[0])
    snapshot_context = game.to_snapshot().human_private_context
    wolf_visible = game._build_wolf_visible_state()

    assert "你的座位：player_id=0，1号" in context
    assert "你的身份：狼人" in context
    assert "狼人队友：player_id=1（2号 玩家2，狼人）" in context
    assert "狼人夜间合法刀口：player_id=2（3号 玩家3）、player_id=3（4号 玩家4）" in context
    assert "player_id=1（2号 玩家2" not in context.split("狼人夜间合法刀口：", 1)[1].splitlines()[0]
    assert "本夜合法刀口候选（只能从这里选）" in wolf_visible
    assert "player_id=0（1号 玩家1）" in wolf_visible
    assert "player_id=1（2号 玩家2）" in wolf_visible
    legal_section = wolf_visible.split("本夜合法刀口候选（只能从这里选）：", 1)[1].split("本轮狼队夜聊：", 1)[0]
    assert "player_id=0（1号 玩家1）" not in legal_section
    assert "player_id=1（2号 玩家2）" not in legal_section
    assert "player_id=2（3号 玩家3）" in legal_section
    assert "player_id=3（4号 玩家4）" in legal_section
    assert "你是 1 号位" in snapshot_context
    assert "狼队友：2号 玩家2" in snapshot_context
    assert "本夜可刀目标：3号 玩家3、4号 玩家4" in snapshot_context
    assert "player_id=" not in snapshot_context


async def test_frontend_snapshot_contract_hides_private_wolf_data_from_non_wolves() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=2)
    game.runtime = FakeRuntime({0: 2, 1: 2}, strict=True, contents={"wolf_chat": "今晚刀3号，先拆查验压力。"})

    await game.resolve_wolf_chat(None)

    snapshot = game.to_snapshot()
    assert snapshot.human_role == RoleName.SEER
    assert "身份：预言家" in snapshot.human_private_context
    assert "player_id=" not in snapshot.human_private_context
    assert snapshot.wolf_chat_records == []
    assert snapshot.wolf_history_summaries == []
    assert snapshot.wolf_night_plan is None
    assert snapshot.wolf_teammate_ids == []
    assert not any(event.visibility == "audit" for event in snapshot.events)
    assert not any(event.phase == "wolf_chat" for event in snapshot.events)
    assert all(event.occurrence_key for event in snapshot.events)


async def test_frontend_snapshot_contract_exposes_current_wolf_panel_only_to_wolves() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 2}, strict=True, contents={"wolf_chat": "今晚刀3号，拆掉可能带队的查验压力。"})

    await game.resolve_wolf_chat(game_action("wolf_chat", 2))
    snapshot = game.to_snapshot()

    assert snapshot.human_is_wolf
    assert snapshot.wolf_teammate_ids == [1]
    assert snapshot.wolf_chat_records
    assert all(record.night_id == game.night_id for record in snapshot.wolf_chat_records)
    assert snapshot.wolf_night_plan is not None
    assert snapshot.wolf_night_plan.night_id == game.night_id
    assert any(event.phase == "wolf_chat" for event in snapshot.events)
    assert all(event.visibility != "audit" for event in snapshot.events)


def test_snapshot_redacts_roles_for_non_wolf_human_before_game_over() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)

    snapshot = game.to_snapshot()

    assert snapshot.players[5].role == RoleName.VILLAGER
    assert snapshot.players[5].camp == Camp.VILLAGER
    assert all(player.role == RoleName.VILLAGER for player in snapshot.players if player.id != 5)
    assert all(player.camp == Camp.VILLAGER for player in snapshot.players if player.id != 5)


def test_snapshot_only_reveals_wolf_teammates_to_wolf_human_before_game_over() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    snapshot = game.to_snapshot()

    assert snapshot.players[0].role == RoleName.WEREWOLF
    assert snapshot.players[1].role == RoleName.WEREWOLF
    assert snapshot.players[1].camp == Camp.WEREWOLF
    assert snapshot.players[2].role == RoleName.VILLAGER
    assert snapshot.players[2].camp == Camp.VILLAGER
    assert snapshot.players[3].role == RoleName.VILLAGER
    assert snapshot.players[3].camp == Camp.VILLAGER


def test_human_wolf_snapshot_has_private_team_and_legal_kill_actions() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)
    game._prepare_wolf_chat_order()

    snapshot = game.to_snapshot()

    assert snapshot.human_is_wolf is True
    assert snapshot.pending_human_action == "wolf_chat"
    assert snapshot.human_allowed_night_actions == ["wolf_chat", "wolf_confirm"]
    assert snapshot.human_target_candidates == [2, 3, 4]
    assert snapshot.wolf_teammate_ids == [1]
    assert "狼队友：2号 玩家2" in snapshot.human_private_context
    assert "本夜可刀目标：3号 玩家3、4号 玩家4、5号 玩家5" in snapshot.human_private_context
    assert all(target not in snapshot.human_target_candidates for target in [0, 1])


def test_human_seer_snapshot_and_result_are_private() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.NIGHT

    snapshot = game.to_snapshot()

    assert snapshot.human_allowed_night_actions == ["inspect", "skip"]
    assert snapshot.human_target_candidates == [0, 2, 3]
    assert "身份：预言家" in snapshot.human_private_context


async def test_get_poll_does_not_auto_skip_human_seer_night_action() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=2)
    game.runtime = FakeRuntime({0: 3, 1: 3}, strict=True)
    game.phase = Phase.WOLF_CHAT
    game._prepare_wolf_chat_order()
    game_manager._games[game.game_id] = game

    await game.resolve_wolf_chat(game_action("wolf_confirm", 3))
    snapshot = await api_get_game(game.game_id)

    assert snapshot.phase == Phase.NIGHT
    assert snapshot.pending_human_action == "night"
    assert snapshot.human_allowed_night_actions == ["inspect", "skip"]
    assert not snapshot.night_summaries


def test_human_witch_snapshot_shows_wolf_target_and_legal_potions() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WITCH, RoleName.SEER, RoleName.HUNTER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.NIGHT
    game.day = 1
    game.wolf_consensus_target_id = 1

    snapshot = game.to_snapshot()

    assert snapshot.human_allowed_night_actions == ["skip", "save", "poison"]
    assert 1 in snapshot.human_target_candidates
    assert set(snapshot.human_target_candidates) == {0, 1, 2, 3}
    assert "身份：女巫" in snapshot.human_private_context
    assert "今晚狼人刀口是 2号 玩家2" in snapshot.human_private_context
    assert "解药可用，本夜可以救该目标" in snapshot.human_private_context
    assert "毒药可用" in snapshot.human_private_context

    game.day = 2
    snapshot = game.to_snapshot()

    assert "今晚狼人刀口是 2号 玩家2" in snapshot.human_private_context
    assert "解药可用，本夜不能救该目标" in snapshot.human_private_context
    assert snapshot.human_allowed_night_actions == ["skip", "poison"]


def test_hunter_and_idiot_private_context_includes_skill_state() -> None:
    hunter_game = make_game([RoleName.WEREWOLF, RoleName.HUNTER, RoleName.SEER], human_player_id=1)
    hunter_context = hunter_game.to_snapshot().human_private_context
    assert "猎人技能" in hunter_context
    assert "可以开枪" in hunter_context
    assert "女巫毒杀不能开枪" in hunter_context

    idiot_game = make_game([RoleName.WEREWOLF, RoleName.IDIOT, RoleName.SEER], human_player_id=1)
    idiot_context = idiot_game.to_snapshot().human_private_context
    assert "白痴技能" in idiot_context
    assert "尚未翻牌" in idiot_context
    idiot_game.players[1].idiot_revealed = True
    idiot_game.players[1].can_vote = False
    revealed_context = idiot_game.to_snapshot().human_private_context
    assert "已翻牌" in revealed_context
    assert "失去投票权" in revealed_context


async def test_get_poll_advances_only_one_ai_step_with_delay() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime(strict=True)
    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0, 1, 2, 3, 4]
    game.speech_cursor = 0
    game.pending_human_action = None
    game.auto_step_ready_ts = 0.0
    await game.advance_ready_ai_steps(max_steps=1)
    first_speech_count = len(game.speeches)
    await game.advance_ready_ai_steps(max_steps=1)

    assert first_speech_count == 1
    assert len(game.speeches) == 1
    assert game.current_speaker_id == 1


async def test_get_poll_fast_forwards_ai_only_wolf_chat_to_next_human_point() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime(strict=True)
    game.phase = Phase.WOLF_CHAT
    game._prepare_wolf_chat_order()
    game.pending_human_action = None
    game.auto_step_ready_ts = time.time() + 30
    game_manager._games[game.game_id] = game

    snapshot = await api_get_game(game.game_id)

    assert snapshot.phase == Phase.WOLF_CHAT
    assert len(game.wolf_chat_records) == 1
    assert game.current_speaker_id == 1


async def test_poll_stops_after_single_ai_wolf_chat_step_to_preserve_feedback() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime({0: 2, 1: 2}, strict=True, contents={"wolf_chat": "今晚先刀3号，拆掉可能起跳带队的位置。"})
    game.phase = Phase.WOLF_CHAT
    game._prepare_wolf_chat_order()
    game.pending_human_action = None
    game.auto_step_ready_ts = 0.0
    game_manager._games[game.game_id] = game
    try:
        snapshot = await api_get_game(game.game_id)

        assert snapshot.phase == Phase.WOLF_CHAT
        assert len(snapshot.wolf_chat_records) == 0
        assert len(game.wolf_chat_records) == 1
        assert game.wolf_chat_records[0].player_id == 0
        assert game.current_speaker_id == 1
    finally:
        game_manager._games.pop(game.game_id, None)


def test_snapshot_has_monotonic_version_for_frontend_stale_response_guard() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)

    first = game.to_snapshot()
    game._add_event("setup", "测试版本推进。")
    second = game.to_snapshot()

    assert second.snapshot_seq > first.snapshot_seq


async def test_dead_human_day_vote_auto_resolves_with_rule_candidates() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=3)
    game.phase = Phase.DAY_VOTE
    game.players[3].alive = False
    game.pending_human_action = None
    game.auto_step_ready_ts = 0.0
    game.runtime = FakeRuntime({0: 1, 1: 0, 2: 0}, strict=True)
    game_manager._games[game.game_id] = game

    snapshot = await api_get_game(game.game_id)

    assert len(snapshot.votes) == 3
    assert snapshot.phase != Phase.DAY_VOTE or snapshot.winner is not None


async def test_human_idiot_exile_reveals_and_keeps_day_flow_playable() -> None:
    roles = [RoleName.WEREWOLF, RoleName.IDIOT, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.DAY_VOTE
    game.runtime = FakeRuntime({0: 1, 2: 1, 3: 1})

    await game.resolve_votes(1)

    assert game.players[1].alive is True
    assert game.players[1].idiot_revealed is True
    assert game.players[1].can_vote is False
    assert game.phase == Phase.WOLF_CHAT
    assert any("翻牌白痴" in event.message for event in game.events)


def test_structured_wolf_context_separates_teammates_from_legal_targets() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    context = game._build_agent_context(
        game.players[0],
        "wolf_chat",
        [player.id for player in game.alive_players() if player.camp != Camp.WEREWOLF],
        prompt="test",
        action_type="wolf_chat",
    )

    assert context.structured is not None
    assert context.structured.self_player.player_id == 0
    assert context.structured.self_player.role == RoleName.WEREWOLF
    assert [teammate.player_id for teammate in context.structured.wolf_teammates] == [1]
    assert context.structured.wolf_teammates[0].role == RoleName.WEREWOLF
    assert context.structured.legal_actions[0].action_type == "wolf_chat"
    assert context.structured.legal_actions[0].target_ids == [2, 3]
    assert context.structured.legal_actions[0].target_seats == [3, 4]


async def test_human_wolf_cannot_propose_teammate_as_kill_target() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_kill", 1))

    assert game.wolf_chat_records[-1].player_id == 0
    assert game.wolf_chat_records[-1].proposed_target_id == 2
    assert game.wolf_chat_records[-1].is_valid_target is True
    assert game.wolf_chat_records[-1].proposed_target_id not in {0, 1}
    assert game.decision_audits[-1].requested_target_id == 1
    assert game.decision_audits[-1].final_target_id == 2
    assert game.decision_audits[-1].corrected is True
    assert all(event.phase != "audit" for event in game.events)


async def test_empty_required_wolf_target_falls_back_without_illegal_correction_audit() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(HumanNightAction(action_type="wolf_chat", target_id=None, chat_content="我先不给死口，听后置位补充。"))

    assert game.wolf_chat_records[-1].proposed_target_id == 2
    assert game.decision_audits[-1].requested_target_id is None


async def test_human_wolf_confirm_requires_explicit_legal_target() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    try:
        await game.resolve_wolf_chat(HumanNightAction(action_type="wolf_confirm", target_id=None, chat_content="我确认。"))
    except ValueError as exc:
        assert "必须显式选择一个合法目标" in str(exc)
        return

    raise AssertionError("wolf_confirm should reject missing target instead of silently falling back")
    assert game.decision_audits[-1].final_target_id == 2
    assert game.decision_audits[-1].corrected is False


async def test_ai_wolf_context_excludes_teammates_from_allowed_targets() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=2)
    runtime = FakeRuntime({0: 1}, strict=False)
    game.runtime = runtime

    await game.resolve_wolf_chat(None)

    context = runtime.contexts[-1]
    assert context.phase == "wolf_chat"
    assert context.player_id == 0
    assert context.allowed_target_ids == [2, 3]
    assert 0 not in context.allowed_target_ids
    assert 1 not in context.allowed_target_ids
    assert context.structured is not None
    assert [teammate.player_id for teammate in context.structured.wolf_teammates] == [1]
    assert context.structured.legal_actions[0].target_ids == [2, 3]
    assert "狼人队友（不能作为刀口）" in context.visible_state
    assert "本夜合法刀口候选（只能从这里选）" in context.visible_state
    assert "2号 玩家2" in context.visible_state
    assert "3号 玩家3" in context.visible_state
    assert "4号 玩家4" in context.visible_state
    assert game.wolf_chat_records[-1].proposed_target_id == 2
    assert game.decision_audits[-1].requested_target_id == 1
    assert game.decision_audits[-1].final_target_id == 2
    assert game.decision_audits[-1].corrected is True


async def test_continuous_wolf_chat_final_consensus_uses_last_legal_targets_only() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 1, 2: 1}, strict=False)

    await game.resolve_wolf_chat(game_action("wolf_kill", 1))
    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(None)

    assert game.phase == Phase.WOLF_CHAT
    assert game.wolf_chat_round == 2
    assert [record.player_id for record in game.wolf_chat_records] == [0, 1, 2]
    assert [record.proposed_target_id for record in game.wolf_chat_records] == [3, 3, 3]
    assert all(record.is_valid_target for record in game.wolf_chat_records)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 1))

    assert game.phase == Phase.NIGHT
    assert [record.player_id for record in game.wolf_chat_records] == [0, 1, 2, 0]
    assert [record.proposed_target_id for record in game.wolf_chat_records] == [3, 3, 3, 3]
    assert game.wolf_consensus_target_id == 3
    assert game.wolf_consensus_target_id not in {0, 1, 2}
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.locked is True
    assert game.wolf_night_plan.final_source == "human_confirm"


async def test_regular_wolf_chat_switch_does_not_override_existing_plan() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 4, 2: 3}, strict=True, contents={"wolf_chat": "我给备刀，但不直接推翻已有刀口。"})

    await game.resolve_wolf_chat(game_action("wolf_chat", 3))
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.current_target_id == 3

    await game.resolve_wolf_chat(None)
    assert game.wolf_night_plan.current_target_id == 3
    assert game.wolf_night_plan.opponents == [1]

    await game.resolve_wolf_chat(None)
    assert game.wolf_night_plan.current_target_id == 3
    assert game.wolf_night_plan.supporters == [0, 2]

    game._finish_or_continue_wolf_chat([3, 4])
    game._finalize_wolf_chat([3, 4])

    assert game.wolf_consensus_target_id == 3


async def test_two_wolves_same_round_use_different_fallback_wolf_chat_lanes() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.runtime = runtime
    game.decision_pipeline = None

    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(None)

    records = game._current_wolf_chat_records()
    assert len(records) == 2
    assert records[0].round_id == records[1].round_id == 1
    assert records[0].content != records[1].content
    assert records[1].proposed_target_id != records[0].proposed_target_id
    assert "备刀" in records[1].content
    assert f"{records[1].proposed_target_seat_no}号" in records[1].content


def test_fallback_wolf_chat_blind_night_does_not_always_pick_first_legal_target() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=6)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    context = game._build_agent_context(
        game.players[0],
        "wolf_chat",
        [player.id for player in game.alive_players() if player.camp != Camp.WEREWOLF],
        "test",
        "wolf_chat",
    )

    decision = runtime._fallback_decision(context)

    assert decision.target_id in context.allowed_target_ids
    assert decision.target_id != context.allowed_target_ids[0]


async def test_wolf_chat_current_night_isolated_from_previous_night_records() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 3, 2: 3})

    await game.resolve_wolf_chat(game_action("wolf_confirm", 3))
    assert game.phase == Phase.NIGHT
    first_night_records = game._current_wolf_chat_records()
    assert len(first_night_records) == 1
    assert first_night_records[0].night_id == 1

    game._advance_to_next_day()

    assert game.phase == Phase.WOLF_CHAT
    assert game.night_id == 2
    assert game._current_wolf_chat_records() == []
    assert game.wolf_chat_records[0].night_id == 1
    assert game._wolf_history_summaries()


async def test_wolf_chat_majority_can_change_initial_plan() -> None:
    roles = [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
    ]
    game = make_game(roles, human_player_id=5)
    game.rule_profile.wolf_chat_rounds = 1
    game.runtime = FakeRuntime(
        targets={0: 4, 1: 5, 2: 5},
        strict=True,
        contents={"wolf_chat": "我给一个明确刀口，按票型收益今晚处理这个位置。"},
    )

    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(None)

    assert game.phase == Phase.NIGHT
    assert [record.proposed_target_id for record in game.wolf_chat_records] == [4, 5, 5]
    assert game.wolf_consensus_target_id == 5
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.current_target_id == 5


async def test_wolf_history_summary_is_stored_without_raw_final_source_tokens() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 2))

    raw_summaries = game.camp_memories[Camp.WEREWOLF].summaries
    assert raw_summaries
    raw_text = "\n".join(raw_summaries)
    assert "最终刀口" not in raw_text
    assert "来源" not in raw_text
    assert "当夜落点" not in raw_text
    assert "主要建议" not in raw_text
    assert "3号" not in raw_text
    assert "本夜有明确刀口建议" in raw_text

    game._advance_to_next_day()
    wolf_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3], "test", "wolf_chat")
    assert wolf_context.structured is not None
    context_text = wolf_context.visible_brief()
    assert "最终刀口" not in context_text
    assert "来源" not in context_text
    assert "proposal_vote" not in context_text
    assert "engine_default" not in context_text
    assert "当夜落点" not in context_text
    assert "1号提3号" not in context_text
    assert "2号提3号" not in context_text
    assert "过往夜晚复盘" in context_text


async def test_current_wolf_chat_context_excludes_same_night_history_summary() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 2))
    game._advance_to_next_day()
    await game.resolve_wolf_chat(
        HumanNightAction(
            action_type="wolf_chat",
            target_id=2,
            chat_content="第二夜先刀3号，别把第一夜的话当成本夜计划。",
        )
    )
    game._summarize_current_wolf_night(2, "plan")

    context = game._build_agent_context(game.players[1], "wolf_chat", [2, 3], "test", "wolf_chat")

    assert context.structured is not None
    assert all("夜晚#2" not in summary for summary in context.structured.wolf_history_summaries)
    assert all("第二夜先刀3号" not in summary for summary in context.structured.wolf_history_summaries)
    assert all("主要建议" not in summary for summary in context.structured.wolf_history_summaries)
    assert all("当夜落点" not in summary for summary in context.structured.wolf_history_summaries)
    assert any("过往夜晚复盘" in summary for summary in context.structured.wolf_history_summaries)


async def test_wolf_chat_round_event_uses_round_not_night_wording() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 2}, strict=True, contents={"wolf_chat": "我也认3号，先拆查验压力。"})

    await game.resolve_wolf_chat(game_action("wolf_chat", 2))
    await game.resolve_wolf_chat(None)

    wolf_events = [event.message for event in game.events if event.phase == "wolf_chat"]
    assert any("本夜第 2 轮协商" in message for message in wolf_events)
    assert not any("第 2 轮夜聊" in message for message in wolf_events)


async def test_wolf_chat_system_events_only_show_current_night_to_wolves() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 3, 2: 3})

    await game.resolve_wolf_chat(game_action("wolf_confirm", 3))
    assert any(event.occurrence_key == "wolf_chat_final:1" for event in game.events)

    game._advance_to_next_day()

    visible_messages = [
        event.message
        for event in game._visible_events_for_player(game.players[0])
        if event.phase == "wolf_chat"
    ]
    assert any("狼人开始夜聊" in message for message in visible_messages)
    assert all(event.occurrence_key != "wolf_chat_final:1" for event in game._visible_events_for_player(game.players[0]))


async def test_wolf_snapshot_never_shows_multiple_nights_final_kill_events() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 2))
    assert any(event.occurrence_key == "wolf_chat_final:1" for event in game.events)

    game._advance_to_next_day()
    await game.resolve_wolf_chat(game_action("wolf_confirm", 3))
    snapshot = game.to_snapshot()

    event_keys = [event.occurrence_key for event in snapshot.events if event.phase == "wolf_chat"]
    timeline_keys = [item.occurrence_key for item in snapshot.visible_timeline if item.phase == "wolf_chat"]

    assert "wolf_chat_final:1" not in event_keys
    assert "wolf_chat_final:1" not in timeline_keys
    assert event_keys.count("wolf_chat_final:2") == 1
    assert timeline_keys.count("wolf_chat_final:2") == 1


async def test_good_player_snapshot_never_receives_wolf_chat_broadcasts() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=2)
    game.runtime = FakeRuntime({0: 3, 1: 3}, strict=True, contents={"wolf_chat": "今晚刀4号，别让它白天带队。"})

    await game.resolve_wolf_chat(None)
    snapshot = game.to_snapshot()

    assert snapshot.wolf_chat_records == []
    assert not snapshot.wolf_history_summaries
    assert not any(event.phase == "wolf_chat" for event in snapshot.events)
    assert not any(item.phase == "wolf_chat" for item in snapshot.visible_timeline)
    assert "今晚刀4号" not in snapshot.human_private_context


async def test_prepare_wolf_chat_order_is_idempotent_within_same_night() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_chat", 2))
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.current_target_id == 2
    assert len(game.wolf_chat_records) == 1

    game._prepare_wolf_chat_order()

    assert game.wolf_chat_prepared_night_id == game.night_id
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.current_target_id == 2
    assert len(game.wolf_chat_records) == 1
    start_events = [
        event for event in game._visible_events_for_player(game.players[0])
        if event.phase == "wolf_chat" and event.message.startswith("狼人开始夜聊")
    ]
    assert len(start_events) == 1


def test_wolf_visible_state_keeps_current_night_records_after_long_history() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    for index in range(20):
        game.wolf_chat_records.append(
            WolfChatRecord(
                day=1,
                night_id=1,
                round_id=1,
                turn_index=index,
                player_id=1,
                speaker_seat_no=2,
                player_name="玩家2",
                content=f"历史夜聊{index}",
                proposed_target_id=2,
                proposed_target_seat_no=3,
            )
        )
    game.night_id = 2
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=2,
            night_id=2,
            round_id=1,
            turn_index=0,
            player_id=0,
            speaker_seat_no=1,
            player_name="玩家1",
            content="当前夜第一条",
            proposed_target_id=2,
            proposed_target_seat_no=3,
        )
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=2,
            night_id=2,
            round_id=1,
            turn_index=1,
            player_id=1,
            speaker_seat_no=2,
            player_name="玩家2",
            content="当前夜第二条",
            proposed_target_id=3,
            proposed_target_seat_no=4,
        )
    )

    visible = game._build_wolf_visible_state()

    assert "当前夜第一条" in visible
    assert "当前夜第二条" in visible
    assert "历史夜聊19" not in visible


def test_wolf_chat_agent_view_does_not_mark_previous_night_whispers_as_new() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=2)
    wolf = game.players[0]
    game._record_message(
        "whisper",
        "wolf_chat",
        "上一夜遗留狼聊，不应作为本夜新增。",
        visibility="wolf",
        speaker=wolf,
        target_id=2,
    )
    game.day = 2
    game.night_id = 2
    game.phase = Phase.WOLF_CHAT
    game.wolf_chat_prepared_night_id = None
    game._prepare_wolf_chat_order()
    game._record_message(
        "whisper",
        "wolf_chat",
        "当前夜真实新增狼聊。",
        visibility="wolf",
        speaker=wolf,
        target_id=2,
    )

    context = game._build_agent_context(wolf, "wolf_chat", [2, 3], "test", "wolf_chat").structured

    assert context is not None
    wolf_new_messages = [message for message in context.new_visible_messages if message.phase == "wolf_chat"]
    assert all(message.night_id == 2 for message in wolf_new_messages)
    new_text = "\n".join(message.content for message in wolf_new_messages)
    assert "上一夜遗留狼聊" not in new_text
    assert "当前夜真实新增狼聊" in new_text


async def test_wolf_chat_finalize_is_idempotent_for_public_events() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 2))
    await game.resolve_wolf_chat(game_action("wolf_confirm", 3))

    final_events = [
        event for event in game.events
        if event.phase == "wolf_chat" and "狼人夜谈结束，最终刀口" in event.message
    ]
    assert len(final_events) == 1
    assert game.wolf_consensus_target_id == 2


async def test_wolf_chat_final_event_is_deduped_by_night_business_key() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 2))
    game.phase = Phase.WOLF_CHAT
    game._finalize_wolf_chat([2, 3])

    final_events = [
        event for event in game.events
        if event.occurrence_key == f"wolf_chat_final:{game.night_id}"
    ]
    assert len(final_events) == 1
    assert "3号" in final_events[0].message
    assert "4号" not in final_events[0].message
    assert game.wolf_consensus_target_id == 2


async def test_wolf_chat_normalizes_chinese_teammate_target_leak() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(
        HumanNightAction(
            action_type="wolf_chat",
            target_id=1,
            chat_content="我问刀二号，二号是狼人，先不刀二号太硬了。",
        )
    )

    record = game.wolf_chat_records[-1]
    assert record.proposed_target_id == 2
    assert record.is_valid_target is True
    assert "3号" in record.content
    assert "二号是狼人" not in record.content
    assert "刀二号" not in record.content


async def test_wolf_chat_finalized_plan_blocks_same_night_reentry_change() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)

    await game.resolve_wolf_chat(game_action("wolf_confirm", 2))
    assert game.phase == Phase.NIGHT
    assert game.wolf_consensus_target_id == 2
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.finalized is True

    game.phase = Phase.WOLF_CHAT
    game.speech_order = [0]
    game.speech_cursor = 0
    await game.resolve_wolf_chat(game_action("wolf_confirm", 3))

    assert game.phase == Phase.NIGHT
    assert game.wolf_consensus_target_id == 2
    assert len([event for event in game.events if event.occurrence_key == "wolf_chat_final:1"]) == 1
    assert not any(record.proposed_target_id == 3 for record in game._current_wolf_chat_records())


async def test_agent_visible_brief_dedupes_current_wolf_chat_lines() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime({0: 2}, strict=True, contents={"wolf_chat": "今晚刀3号，拆掉他白天可能带队的归票压力。"})

    await game.resolve_wolf_chat(None)
    context = game._build_agent_context(game.players[1], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    brief = context.visible_brief()

    assert brief.count("今晚刀3号，拆掉他白天可能带队的归票压力") == 1
    assert "狼人夜聊" in brief


async def test_wolf_day_context_uses_summary_not_raw_camp_records() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.rule_profile.wolf_chat_rounds = 1
    game.runtime = FakeRuntime({0: 2, 1: 2}, strict=True, contents={"wolf_chat": "今晚刀3号，拆掉他白天可能带队的归票压力。"})

    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(None)
    assert game.phase == Phase.NIGHT
    game.phase = Phase.DAY_SPEECH
    context = game._build_agent_context(game.players[0], "day_speech", [2, 3, 4], "test", "day_speech")

    assert context.structured is not None
    assert context.structured.wolf_chat_records == []
    assert not hasattr(context.structured, "camp_memory")
    assert not hasattr(context.structured, "memory")
    assert not hasattr(context.structured, "agent_state")
    assert context.structured.wolf_history_summaries


def test_agent_visible_brief_summarizes_public_speech_without_raw_quote_chain() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.DAY_SPEECH
    raw = "我先接2号刚才那句，2号说1号先不把人打死没问题，但我觉得你自己也有点把话说满了。"
    game.speeches.append(SpeechRecord(day=1, player_id=2, player_name="玩家3", speech_type="day", content=raw))

    context = game._build_agent_context(game.players[1], "day_speech", [0, 2, 3], "test", "day_speech")
    brief = context.visible_brief()

    assert "最近公开发言证据：" in brief
    assert "发言侧重" in brief
    assert raw not in brief


def test_model_public_speech_private_leak_falls_back() -> None:
    runtime = OpenAIAgentRuntime()
    context = AIContext(
        player_id=0,
        role=RoleName.WEREWOLF,
        day=1,
        phase="day_speech",
        visible_state="",
        allowed_target_ids=[1, 2, 3],
        prompt="白天发言",
        persona_style="冷刀拆解型",
        strategy_style="深水倒钩流",
    )
    decision = AgentDecision(action="speak", target_id=None, content="我是狼人，昨晚刀口是3号，狼队友别露。", reason="test")

    result = runtime._finalize_model_decision(decision, context)

    assert "我是狼人" not in result.content
    assert "刀口" not in result.content
    assert result.reason == "模型输出含出戏表达或内部字段，已切换为本地真人化兜底。"


def test_model_public_speech_wolf_history_summary_leak_falls_back() -> None:
    runtime = OpenAIAgentRuntime()
    context = AIContext(
        player_id=0,
        role=RoleName.WEREWOLF,
        day=2,
        phase="day_speech",
        visible_state="",
        allowed_target_ids=[1, 2, 3],
        prompt="白天发言",
        persona_style="冷刀拆解型",
        strategy_style="深水倒钩流",
    )
    decision = AgentDecision(
        action="speak",
        target_id=None,
        content="过往夜晚复盘：上夜狼队共识很稳，今天不要复述旧夜具体刀口。",
        reason="test",
    )

    result = runtime._finalize_model_decision(decision, context)

    assert "过往夜晚复盘" not in result.content
    assert "上夜" not in result.content
    assert "狼队共识" not in result.content
    assert result.reason == "模型输出含出戏表达或内部字段，已切换为本地真人化兜底。"


def test_status_map_respects_hidden_first_day_deaths() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.players[2].alive = False
    game.first_day_death_announcement_pending = True
    game.phase = Phase.SHERIFF_SPEECH

    context = game._build_agent_context(game.players[1], "sheriff_speech", [], "test", "sheriff_speech")

    assert context.structured is not None
    assert context.structured.status_map[2] == "ALIVE"


def test_playability_report_flags_chain_commentary_speech() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    for index in range(6):
        content = f"{index + 1}号这里我给明确票意，因为发言和站边都不闭合，今天可以进票池。"
        if index == 2:
            content = "我先接2号刚才那句，2号这句没问题，我先认你偏好人，但后面再看。"
        game.speeches.append(SpeechRecord(day=1, player_id=index, player_name=f"玩家{index + 1}", speech_type="day", content=content))

    report = evaluate_playability(game, require_counterclaim=False)

    assert any("链式接话" in finding or "套话抬轿" in finding for finding in report.findings)


def test_playability_report_flags_public_wolf_history_summary_leak() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.extend(
        [
            SpeechRecord(day=2, player_id=0, player_name="玩家1", content="过往夜晚复盘：上夜狼队共识稳定，今天别复述旧夜具体刀口。", speech_type="day"),
            SpeechRecord(day=2, player_id=1, player_name="玩家2", content="我今天看3号，因为他发言和票型不闭合。", speech_type="day"),
            SpeechRecord(day=2, player_id=2, player_name="玩家3", content="我投2号，他一直只给态度不给链路。", speech_type="day"),
            SpeechRecord(day=2, player_id=3, player_name="玩家4", content="4号这里先说清楚票型，今天我压1号。", speech_type="day"),
            SpeechRecord(day=2, player_id=4, player_name="玩家5", content="我不认1号这轮站边，理由是发言改口太快。", speech_type="day"),
            SpeechRecord(day=2, player_id=5, player_name="玩家6", content="今天票口别散，先看2号和3号谁在补票。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game, require_counterclaim=False)

    assert any("狼队历史摘要" in finding for finding in report.findings)


def test_playability_report_flags_first_round_wolf_chat_without_backup_target() -> None:
    roles = [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.VILLAGER,
    ]
    game = make_game(roles, human_player_id=7)
    for player_id in range(4):
        game.wolf_chat_records.append(
            WolfChatRecord(
                day=1,
                night_id=1,
                round_id=1,
                player_id=player_id,
                speaker_seat_no=player_id + 1,
                player_name=f"玩家{player_id + 1}",
                content="今晚动5号，拆预言家信息和白天归票收益，明天口径分散。",
                proposed_target_id=4,
                proposed_target_seat_no=5,
                is_valid_target=True,
            )
        )

    report = evaluate_playability(game, require_counterclaim=False)

    assert any("首轮无备刀分歧" in finding for finding in report.findings)


def test_playability_allows_unified_wolf_focus_on_exposed_power_role() -> None:
    roles = [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.VILLAGER,
    ]
    game = make_game(roles, human_player_id=7)
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=4,
            player_name="玩家5",
            content="我是预言家，昨晚验到1号是狼人，今天先出1号。",
            speech_type="day",
        )
    )
    for player_id in range(4):
        game.wolf_chat_records.append(
            WolfChatRecord(
                day=2,
                night_id=2,
                round_id=1,
                player_id=player_id,
                speaker_seat_no=player_id + 1,
                player_name=f"玩家{player_id + 1}",
                content="今晚统一动5号，预言家活着会继续带队归票。",
                proposed_target_id=4,
                proposed_target_seat_no=5,
                is_valid_target=True,
            )
        )

    report = evaluate_playability(game, require_counterclaim=False)

    assert not any("第2夜狼聊首轮无备刀分歧" in finding for finding in report.findings)


def test_playability_report_flags_last_words_reused_from_day_template() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    reused = "我不急着站死3号。今天先听链路能不能自洽，再看谁把这个话题当工具去压别人。"
    for index in range(6):
        content = reused if index == 0 else f"{index + 1}号这里我给明确票意，因为发言和站边都不闭合，今天可以进票池。"
        game.speeches.append(SpeechRecord(day=1, player_id=index, player_name=f"玩家{index + 1}", speech_type="day", content=content))
    game.speeches.append(SpeechRecord(day=1, player_id=5, player_name="玩家6", speech_type="last_words", content=reused))

    report = evaluate_playability(game, require_counterclaim=False)

    assert any("遗言复用白天发言" in finding or "遗言像普通白天模板" in finding for finding in report.findings)


def test_playability_report_flags_repeated_claim_response_templates() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.speeches.append(
        SpeechRecord(day=1, player_id=1, player_name="玩家2", speech_type="day", content="我是预言家，昨晚验到1号是狼人，今天先出1号。")
    )
    repeated = [
        "2号先留验证空间。第一天最怕好人被假节奏带散，我会看1号是不是在躲自己的发言责任。",
        "2号先留验证空间。第一天最怕好人被假节奏带散，我会看1号是不是在躲自己的发言责任。",
        "2号先留验证空间。第一天最怕好人被假节奏带散，我会看1号是不是在躲自己的发言责任。",
    ]
    for offset, content in enumerate(repeated, start=2):
        game.speeches.append(SpeechRecord(day=1, player_id=offset, player_name=f"玩家{offset + 1}", speech_type="day", content=content))
    for index in range(4, 6):
        game.speeches.append(
            SpeechRecord(day=1, player_id=index, player_name=f"玩家{index + 1}", speech_type="day", content=f"{index + 1}号给明确票意，因为验人和发言链都要兑现。")
        )

    report = evaluate_playability(game, require_counterclaim=False)

    assert any("身份宣称回应模板重复" in finding for finding in report.findings)


def test_playability_counts_natural_fake_seer_counterclaim() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="我是预言家，昨晚验6号是好人，今天先看1号。", speech_type="day"),
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="我也拍预言家，昨晚验5号是狼人。今天把两个预言家的验人和站边全部摊开。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="我按票型说，今天先看谁站边1号和3号最急。", speech_type="day"),
            SpeechRecord(day=1, player_id=3, player_name="玩家4", content="我不急着站死，今天看验人链和票型能不能闭合。", speech_type="day"),
            SpeechRecord(day=1, player_id=4, player_name="玩家5", content="今天如果要出票，我先压1号，因为他起跳后的链路更散。", speech_type="day"),
            SpeechRecord(day=1, player_id=5, player_name="玩家6", content="我就是普通身份视角，今天投票会看谁借预言家线补票。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game)

    assert report.seer_counterclaim_count >= 1
    assert not any("身份博弈不足" in finding for finding in report.findings)


async def test_wolf_chat_post_snapshot_stops_at_night_with_current_final_feedback() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)
    game_manager._games[game.game_id] = game

    snapshot = await api_resolve_wolf_chat(
        game.game_id,
        NightRequest(action_type="wolf_confirm", target_id=2, chat_content="确认刀3号，明天别一起解释死因。"),
    )

    assert snapshot.phase == Phase.NIGHT
    assert snapshot.night_id == 1
    assert snapshot.wolf_night_plan is not None
    assert snapshot.wolf_night_plan.current_target_id == 2
    assert any(event.occurrence_key == "wolf_chat_final:1" for event in snapshot.events)
    assert any(item.occurrence_key == "wolf_chat_final:1" for item in snapshot.visible_timeline)


async def test_public_system_events_use_unified_sequence_writer() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0, 1, 2, 3]
    game.speech_cursor = 3

    await game.resolve_day_speeches("")

    assert game.events
    seqs = [event.seq for event in game.events]
    assert seqs == list(range(len(seqs)))
    assert all(event.occurrence_key for event in game.events)
    assert all(event.visible_to_player_ids for event in game.events if event.visibility == "public")


async def test_dead_human_does_not_get_day_vote_pending_action() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.players[1].alive = False
    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0, 2, 3]
    game.speech_cursor = 2

    await game.resolve_day_speeches("")

    assert game.phase == Phase.DAY_VOTE
    assert game.pending_human_action is None
    assert game.to_snapshot().human_target_candidates == []


async def test_api_speech_submission_does_not_auto_advance_following_ai_speakers() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.runtime.enabled = False
    game.phase = Phase.DAY_SPEECH
    game.speech_order = [1, 0, 2, 3]
    game.speech_cursor = 0
    game.pending_human_action = "day_speech"
    game_manager._games[game.game_id] = game

    snapshot = await api_resolve_speech(
        game.game_id,
        SpeechRequest(content="我先点1号和3号，后面谁补票最快我就记谁。"),
    )

    assert snapshot.phase == Phase.DAY_SPEECH
    assert len(game.speeches) == 1
    assert game.speeches[0].player_id == 1
    assert snapshot.pending_human_action is None
    assert snapshot.current_speaker_id == 0


async def test_get_poll_advances_only_one_ai_day_speech_step() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime(strict=True)
    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0, 1, 2, 3, 4]
    game.speech_cursor = 0
    game.pending_human_action = None
    game.auto_step_ready_ts = time.time() + 30
    game_manager._games[game.game_id] = game

    try:
        snapshot = await api_get_game(game.game_id)

        assert snapshot.phase == Phase.DAY_SPEECH
        assert len(game.speeches) == 1
        assert game.speeches[0].player_id == 0
        assert snapshot.current_speaker_id == 1
        assert snapshot.pending_human_action is None
    finally:
        game_manager._games.pop(game.game_id, None)


async def test_exile_pk_speech_with_no_current_speaker_advances_to_vote_without_snapshot_crash() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.EXILE_PK_SPEECH
    game.exile_pk_candidate_ids = [0, 2]
    game.speech_order = [0, 2]
    game.speech_cursor = 2
    game.pending_human_action = None

    snapshot = game.to_snapshot()

    assert snapshot.phase == Phase.EXILE_PK_SPEECH
    assert "等待重新投票" in snapshot.current_hint

    await game.advance_ready_ai_step_if_needed()

    assert game.phase == Phase.EXILE_PK_VOTE
    assert game.pending_human_action == "day_vote"


def test_fallback_day_speech_rejects_dead_seat_anchor() -> None:
    runtime = OpenAIAgentRuntime()
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)
    game.players[1].alive = False
    game.phase = Phase.DAY_SPEECH
    context = game._build_agent_context(game.players[0], "day_speech", [], "test", "speak")

    assert runtime._content_needs_fallback("2号这轮我继续追，今天先出2号。", context)
    assert not runtime._content_needs_fallback("3号这轮我会继续听，今天先看3号怎么落票。", context)


def test_snapshot_only_exposes_current_wolf_night_plan() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.wolf_night_plan = WolfNightPlan(day=1, night_id=1, current_target_id=2, locked=True)
    game.night_id = 2

    assert game.to_snapshot().wolf_night_plan is None


async def test_wolf_chat_post_does_not_run_timeout_before_user_action() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.deadline_ts = time.time() - 1
    game.time_limit_seconds = 45
    game_manager._games[game.game_id] = game

    await api_resolve_wolf_chat(
        game.game_id,
        NightRequest(action_type="wolf_confirm", target_id=3, chat_content="确认刀4号，别让他白天带队。"),
    )

    assert len(game.wolf_chat_records) == 1
    assert game.wolf_chat_records[0].player_id == 0
    assert game.wolf_chat_records[0].proposed_target_id == 3
    assert "确认刀4号" in game.wolf_chat_records[0].content
    assert game.wolf_night_plan is not None
    assert game.wolf_night_plan.current_target_id == 3
    if game.night_summaries:
        assert game.night_summaries[-1].wolf_target_id == 3
    else:
        assert game.wolf_consensus_target_id == 3


async def test_auto_ai_steps_use_fast_fallback_runtime() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=2)
    game.runtime = ExplodingRuntime()
    game.decision_pipeline = None
    game.auto_step_ready_ts = 0.0
    game._prepare_wolf_chat_order()

    await game.advance_ready_ai_steps(max_steps=4)

    assert game.phase in {Phase.NIGHT, Phase.WOLF_CHAT}
    assert game.wolf_chat_records


async def test_witch_can_self_save_only_on_first_night() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WITCH, RoleName.SEER, RoleName.HUNTER, RoleName.GUARD, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.NIGHT
    game.day = 1
    game.wolf_consensus_target_id = 1

    await game.resolve_night(action := game_action("save", 1))

    assert not game.night_summaries[-1].deaths
    assert game.night_summaries[-1].witch_saved is True
    assert action.target_id == 1

    game = make_game(roles, human_player_id=1)
    game.phase = Phase.NIGHT
    game.day = 2
    game.wolf_consensus_target_id = 1

    await game.resolve_night(game_action("save", 1))

    assert game.night_summaries[-1].deaths == [1]
    assert game.night_summaries[-1].witch_saved is False


async def test_ai_witch_first_night_self_save_is_in_action_space() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WITCH, RoleName.SEER, RoleName.HUNTER]
    game = make_game(roles, human_player_id=2)
    runtime = FakeRuntime({1: 1}, strict=True)
    game.runtime = runtime
    game.phase = Phase.NIGHT
    game.day = 1
    game.wolf_consensus_target_id = 1

    await game.resolve_night(game_action("skip", None))

    witch_context = next(context for context in runtime.contexts if context.player_id == 1)
    assert 1 in witch_context.allowed_target_ids
    assert witch_context.structured is not None
    assert witch_context.structured.legal_actions[0].target_ids == witch_context.allowed_target_ids
    assert game.night_summaries[-1].witch_saved is True
    assert game.night_summaries[-1].deaths == []


def game_action(action_type: str, target_id: int | None):
    from app.engine.models import HumanNightAction

    return HumanNightAction(action_type=action_type, target_id=target_id)


async def test_day_vote_allows_self_vote() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.DAY_VOTE

    await game.resolve_votes(0)

    human_vote = next(vote for vote in game.votes if vote.voter_id == 0)
    assert human_vote.target_id == 0


async def test_ai_day_vote_action_space_excludes_self_vote() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    runtime = FakeRuntime({1: 1, 2: 2}, strict=True)
    game.runtime = runtime
    game.phase = Phase.DAY_VOTE

    await game.resolve_votes(0)

    ai_votes = [vote for vote in game.votes if vote.voter_id in {1, 2}]
    assert ai_votes
    assert all(vote.target_id != vote.voter_id for vote in ai_votes)
    vote_contexts = [context for context in runtime.contexts if context.phase == "day_vote"]
    assert vote_contexts
    assert all(context.player_id not in context.allowed_target_ids for context in vote_contexts)


def test_human_sample_seer_speech_reports_private_inspection() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=1)
    game._remember_private(
        game.human_player,
        "第1夜查验 玩家1（1号） -> 狼人。",
        {"target_id": 0, "target_seat_no": 1, "result": "狼人"},
    )

    speech = _human_sample_speech(game)

    assert "我是预言家" in speech
    assert "1号是狼人" in speech


def test_human_sample_wolf_target_avoids_fixed_first_non_wolf_blind_kill() -> None:
    game = build_sample_game(use_live_runtime=False, human_player_id=0, script_variant=0)
    candidates = [player.id for player in game.alive_players() if player.camp != Camp.WEREWOLF]

    target_id = _human_sample_wolf_target(game, candidates)

    assert target_id in candidates
    assert target_id != candidates[0]


def test_human_sample_seer_last_words_reports_private_inspection() -> None:
    game = build_sample_game(use_live_runtime=False, human_player_id=4, script_variant=0)
    game._remember_private(
        game.human_player,
        "第1夜查验 玩家1（1号） -> 狼人。",
        {"target_id": 0, "target_seat_no": 1, "result": "狼人"},
    )
    game.current_exile_target_id = game.human_player_id

    last_words = _human_sample_last_words(game)

    assert "我是预言家" in last_words
    assert "1号是狼人" in last_words


def test_sample_game_does_not_seed_future_day_speeches_before_first_night() -> None:
    game = build_sample_game(use_live_runtime=False, human_player_id=8, script_variant=0)

    assert game.phase == Phase.WOLF_CHAT
    assert game.speeches == []
    assert "预言家" not in game.public_state_text()


async def test_exile_tie_uses_exile_pk_phases() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.DAY_VOTE
    game.runtime = FakeRuntime({1: 0, 2: 1, 3: 0})

    await game.resolve_votes(1)

    assert game.phase == Phase.EXILE_PK_SPEECH
    assert game.exile_pk_candidate_ids


async def test_night_death_gets_last_words_before_day_speech() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.GUARD, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=2)
    game.phase = Phase.NIGHT
    game.day = 2
    game.wolf_consensus_target_id = 1

    await game.resolve_night(game_action("skip", None))

    assert game.phase == Phase.LAST_WORDS
    assert game.current_exile_target_id == 1


async def test_multiple_night_deaths_process_each_special_role_after_last_words() -> None:
    roles = [RoleName.WEREWOLF, RoleName.HUNTER, RoleName.WITCH, RoleName.SEER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=2)
    game.phase = Phase.NIGHT
    game.day = 2
    game.wolf_consensus_target_id = 1

    await game.resolve_night(game_action("poison", 3))

    assert game.phase == Phase.LAST_WORDS
    assert game.last_words_queue == [3]
    assert game.death_resolution_player_ids == [1, 3]

    await game.resolve_last_words("我是猎人，先看清票型。")

    assert game.phase == Phase.LAST_WORDS
    assert game.current_exile_target_id == 3

    await game.resolve_last_words("")

    assert game.phase == Phase.HUNTER_SHOT
    assert game.pending_hunter_id == 1
    assert game.death_resolution_player_ids == [3]


async def test_duplicate_last_words_submission_does_not_stall_death_resolution() -> None:
    roles = [RoleName.WEREWOLF, RoleName.HUNTER, RoleName.WITCH, RoleName.SEER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=2)
    game.phase = Phase.LAST_WORDS
    game.day = 2
    game.players[1].alive = False
    game.current_exile_target_id = 1
    game.death_resolution_source = "night"
    game.death_resolution_player_ids = [1]
    game.speeches.append(
        SpeechRecord(
            day=2,
            player_id=1,
            player_name="玩家2",
            content="我已经留过遗言。",
            speech_type="last_words",
        )
    )

    await game.resolve_last_words("重复提交。")

    assert game.phase == Phase.HUNTER_SHOT
    assert game.pending_hunter_id == 1


async def test_badge_transfer_continues_remaining_death_resolution_queue() -> None:
    roles = [RoleName.WEREWOLF, RoleName.HUNTER, RoleName.WITCH, RoleName.SEER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=3)
    game.players[3].is_sheriff = True
    game.sheriff_id = 3
    game.players[1].alive = False
    game.players[3].alive = False
    game.phase = Phase.BADGE_TRANSFER
    game.current_exile_target_id = 3
    game.death_resolution_source = "night"
    game.death_resolution_player_ids = [1]

    await game.resolve_badge_transfer(SheriffAction(badge_target_id=4))

    assert game.phase == Phase.HUNTER_SHOT
    assert game.pending_hunter_id == 1
    assert game.sheriff_id == 4
    assert game.players[4].is_sheriff is True


async def test_playability_smoke_night_to_day_state_and_memory() -> None:
    roles = [
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
    game = make_game(roles, human_player_id=8)
    game.runtime = FakeRuntime({0: 4, 1: 4, 2: 4, 3: 4, 4: 0}, strict=True)

    for _ in range(8):
        await game.resolve_wolf_chat(None)
        if game.phase == Phase.NIGHT:
            break

    assert game.phase == Phase.NIGHT
    assert game.wolf_consensus_target_id == 4
    assert all(event.phase != "audit" for event in game.events)
    assert game._wolf_history_summaries()
    assert game.agent_states[0].last_internal_plan
    assert "5号" not in game.agent_states[0].last_internal_plan
    assert game.agent_states[0].current_focus == ""

    await game.resolve_night(game_action("skip", None))

    assert game.phase in {Phase.LAST_WORDS, Phase.DAY_SPEECH, Phase.GAME_OVER}
    assert not game.agent_states[0].last_public_position
    assert game.agent_states[4].private_summary


async def test_playability_full_day_loop_has_private_boundaries_and_non_ai_speeches() -> None:
    roles = [
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
    game = make_game(roles, human_player_id=8)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.runtime = runtime
    game.decision_pipeline = None
    game.speeches.extend(
        [
            SpeechRecord(
                day=0,
                player_id=4,
                player_name="玩家5",
                content="我是预言家，昨晚验了9号是好人，今天我会先压4号的发言。",
                speech_type="day",
            ),
            SpeechRecord(
                day=0,
                player_id=6,
                player_name="玩家7",
                content="4号刚才打人只给态度不给链路，9号被保以后反应偏自然，我先站9号。",
                speech_type="day",
            ),
        ]
    )

    for _ in range(10):
        await game.resolve_wolf_chat(None)
        if game.phase == Phase.NIGHT:
            break

    assert game.phase == Phase.NIGHT
    assert game.wolf_consensus_target_id == 4
    assert_wolf_chat_quality(game._current_wolf_chat_records())

    await game.resolve_night(game_action("skip", None))
    snapshot = game.to_snapshot()
    assert all(summary.wolf_target_id is None for summary in snapshot.night_summaries)
    assert all(summary.seer_target_id is None for summary in snapshot.night_summaries)
    assert all(summary.seer_result is None for summary in snapshot.night_summaries)
    assert not snapshot.wolf_chat_records
    assert not snapshot.wolf_history_summaries
    assert not any(event.phase == "wolf_chat" for event in snapshot.events)

    if game.phase == Phase.LAST_WORDS:
        while game.phase == Phase.LAST_WORDS:
            await game.resolve_last_words("")

    assert game.phase == Phase.DAY_SPEECH
    while game.phase == Phase.DAY_SPEECH:
        if game.current_speaker_id == game.human_player_id:
            await game.resolve_day_speeches("我今天先看4号和7号，谁投票跟得最急我就记谁。")
        else:
            await game.resolve_day_speeches("")

    day_speeches = [speech for speech in game.speeches if speech.day == 1 and speech.speech_type == "day"]
    assert len(day_speeches) == len(game.alive_players())
    assert_non_ai_table_speech([speech.content for speech in day_speeches])
    assert all(not any(token in speech.content for token in PRIVATE_LEAK_TOKENS) for speech in day_speeches)
    assert len({speech.content for speech in day_speeches}) >= 2

    assert game.phase == Phase.DAY_VOTE
    await game.resolve_votes(0)
    assert game.phase in {Phase.LAST_WORDS, Phase.HUNTER_SHOT, Phase.WOLF_CHAT, Phase.GAME_OVER, Phase.EXILE_PK_SPEECH}
    assert all(event.phase != "audit" for event in game.events)
    for _ in range(12):
        if game.phase == Phase.LAST_WORDS:
            await game.resolve_last_words("")
            continue
        if game.phase == Phase.HUNTER_SHOT:
            await game.resolve_hunter_shot(None)
            continue
        if game.phase == Phase.EXILE_PK_SPEECH:
            await game._advance_exile_pk_speech("")
            continue
        if game.phase == Phase.EXILE_PK_VOTE:
            await game.resolve_votes(game.exile_pk_candidate_ids[0])
            continue
        break

    assert game.phase in {Phase.WOLF_CHAT, Phase.GAME_OVER}
    if game.phase == Phase.WOLF_CHAT:
        assert game.day == 2
        assert game.night_id == 2
        assert game._current_wolf_chat_records() == []
        assert game._wolf_history_summaries()
        assert game.to_snapshot().events and not any(event.phase == "wolf_chat" for event in game.to_snapshot().events)
        visible = game._build_agent_context(
            game.alive_wolves()[0],
            "wolf_chat",
            [player.id for player in game.alive_players() if player.camp != Camp.WEREWOLF],
            prompt="test",
            action_type="wolf_chat",
        )
        assert visible.structured is not None
        assert visible.structured.wolf_chat_records == []
        assert visible.structured.wolf_history_summaries

    report = evaluate_playability(game)
    assert report.findings == []
    assert report.day_speech_count >= 4
    assert report.wolf_chat_count >= 1


def test_playability_report_flags_bad_sample() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=3)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="我是预言家，昨晚验了2号是好人。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="我先接结构化上下文 target_id=2。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="我先接结构化上下文 target_id=2。", speech_type="day"),
        ]
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=1,
            player_id=0,
            speaker_seat_no=1,
            player_name="玩家1",
            content="随便看看。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            is_valid_target=True,
        )
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=1,
            player_id=1,
            speaker_seat_no=2,
            player_name="玩家2",
            content="随便看看。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            is_valid_target=True,
        )
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=1,
            player_id=2,
            speaker_seat_no=3,
            player_name="玩家3",
            content="随便看看。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            is_valid_target=True,
        )
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=1,
            player_id=3,
            speaker_seat_no=4,
            player_name="玩家4",
            content="我不同意刀3号，先别动3号，随便看看。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            is_valid_target=True,
        )
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=2,
            player_id=0,
            speaker_seat_no=1,
            player_name="玩家1",
            content="第二轮我收口，5号不换。10号留到明天会继续产出信息和归票压力。",
            proposed_target_id=9,
            proposed_target_seat_no=10,
            is_valid_target=True,
        )
    )

    report = evaluate_playability(game)

    assert not report.passed
    assert any("重复度" in finding for finding in report.findings)
    assert any("AI字段泄漏" in finding for finding in report.findings)
    assert any("白天身份宣称回应不足" in finding for finding in report.findings)
    assert any("狼聊文本目标与结构化目标不一致" in finding for finding in report.findings)
    assert any("狼聊文本否定了结构化刀口" in finding for finding in report.findings)
    assert any("狼聊文本存在冲突收口目标" in finding for finding in report.findings)
    assert any("狼聊重复度" in finding for finding in report.findings)
    assert any("狼聊分工不足" in finding for finding in report.findings)


def test_playability_flags_missing_role_strategy_chain() -> None:
    roles = [
        RoleName.WEREWOLF,
        RoleName.WEREWOLF,
        RoleName.SEER,
        RoleName.WITCH,
        RoleName.HUNTER,
        RoleName.IDIOT,
        RoleName.VILLAGER,
        RoleName.VILLAGER,
    ]
    game = make_game(roles, human_player_id=7)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="1号觉得2号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="2号觉得3号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="3号觉得4号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=3, player_name="玩家4", content="4号觉得5号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=4, player_name="玩家5", content="5号觉得6号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=5, player_name="玩家6", content="6号觉得7号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=6, player_name="玩家7", content="7号觉得8号发言不顺，今天先听后面。", speech_type="day"),
            SpeechRecord(day=1, player_id=7, player_name="玩家8", content="8号觉得1号发言不顺，今天先听后面。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game, require_counterclaim=False)

    assert any("角色打法链路不足" in finding for finding in report.findings)


def test_playability_report_flags_pending_action_without_candidates() -> None:
    roles = [RoleName.HUNTER, RoleName.WEREWOLF, RoleName.SEER]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.HUNTER_SHOT
    game.players[0].alive = False
    game.players[1].alive = False
    game.players[2].alive = False
    game.pending_human_action = "hunter_shot"
    game.pending_hunter_id = 0

    report = evaluate_playability(game)

    assert any("缺少候选目标" in finding for finding in report.findings)


async def test_sample_runner_produces_passing_playability_report() -> None:
    game, report = await run_sample_game()

    assert report.passed
    assert report.day_speech_count >= 4
    assert report.wolf_chat_count >= 1
    assert report.wolf_chat_roleplay_variety >= 3
    assert report.concrete_speech_count >= 2
    assert report.claim_response_count >= 2
    assert report.vote_intent_speech_count >= 2
    assert report.day_angle_variety >= 4
    report_payload = _report_dict(report)
    assert report.role_strategy_signal_count >= 3
    assert report_payload["passed"] is True
    assert report_payload["role_strategy_signal_count"] == report.role_strategy_signal_count
    assert report_payload["role_strategy_roles"] == report.role_strategy_roles
    assert report_payload["wolf_chat_stance_variety"] == report.wolf_chat_stance_variety
    assert _sample_excerpt(game)["wolf_chat"]
    assert _sample_excerpt(game)["day_speeches"]


async def test_fallback_wolf_chat_content_semantically_matches_target() -> None:
    game, report = await run_sample_game(days=1, human_player_id=8)

    assert game.wolf_chat_records
    assert not any("狼聊文本否定了结构化刀口" in finding for finding in report.findings)
    for record in game.wolf_chat_records:
        assert record.proposed_target_seat_no is not None
        assert f"{record.proposed_target_seat_no}号" in record.content


async def test_sample_runner_two_day_game_keeps_playability_and_night_isolation() -> None:
    game, report = await run_sample_game(days=2)

    assert report.passed
    assert report.night_count >= 1
    assert report.wolf_chat_night_count >= 1
    assert game.day == 2
    assert game.night_id == 2
    assert game.phase in {
        Phase.WOLF_CHAT,
        Phase.NIGHT,
        Phase.DAY_SPEECH,
        Phase.DAY_VOTE,
        Phase.LAST_WORDS,
        Phase.HUNTER_SHOT,
        Phase.BADGE_TRANSFER,
        Phase.GAME_OVER,
    }
    assert report.wolf_chat_roleplay_variety >= 3
    assert report.claim_response_count >= 2
    assert report.vote_intent_speech_count >= 2
    assert report.day_angle_variety >= 4
    assert report.max_vote_share <= 0.85
    assert all(record.night_id == game.night_id for record in game._current_wolf_chat_records())
    if game.human_player.camp != Camp.WEREWOLF:
        assert not game.to_snapshot().wolf_chat_records
    seer_speeches = [
        speech.content
        for speech in game.speeches
        if game.players[speech.player_id].role == RoleName.SEER and speech.speech_type == "day"
    ]
    assert any("夜里" in content or "验" in content or "好人信息" in content for content in seer_speeches)


async def test_sample_wolf_chat_same_round_has_no_duplicate_full_lines() -> None:
    game, report = await run_sample_game(days=1, human_player_id=8)

    assert report.passed
    grouped: dict[tuple[int, int], list[str]] = {}
    for record in game.wolf_chat_records:
        grouped.setdefault((record.night_id, record.round_id), []).append(record.content)
    for lines in grouped.values():
        assert len(lines) == len(set(lines))


async def test_sample_day_speeches_avoid_mechanical_external_slot_wording() -> None:
    game, report = await run_sample_game(days=2, human_player_id=8, script_variant=2)

    assert report.passed
    joined = "\n".join(record.content for record in game.speeches)
    assert "这种外置位" not in joined
    assert "这类外置位" not in joined


async def test_sample_visible_player_text_avoids_external_slot_jargon() -> None:
    game, report = await run_sample_game(days=2, human_player_id=0, script_variant=0)

    assert report.passed
    joined = "\n".join([*(record.content for record in game.wolf_chat_records), *(speech.content for speech in game.speeches)])
    assert "外置位" not in joined


def test_playability_report_flags_external_slot_jargon() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=1,
            round_id=1,
            player_id=0,
            speaker_seat_no=1,
            player_name="玩家1",
            content="今晚刀3号，明天把票型压力往外置位推。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            is_valid_target=True,
        )
    )

    report = evaluate_playability(game)

    assert any("机械桌游术语" in finding for finding in report.findings)


async def test_sample_runner_matrix_covers_multiple_human_roles() -> None:
    cases = [
        (0, RoleName.WEREWOLF, 0),
        (4, RoleName.SEER, 2),
        (5, RoleName.WITCH, 1),
        (6, RoleName.HUNTER, 2),
        (7, RoleName.IDIOT, 1),
        (8, RoleName.VILLAGER, 0),
    ]

    for human_player_id, expected_role, script_variant in cases:
        game, report = await run_sample_game(days=1, human_player_id=human_player_id, script_variant=script_variant)
        assert game.human_player.role == expected_role
        assert report.passed, (expected_role, report.findings)
        assert report.day_speech_count >= 4
        assert report.wolf_chat_count >= 1
        snapshot = game.to_snapshot()
        if expected_role != RoleName.WEREWOLF:
            assert not snapshot.wolf_chat_records
            assert not any(event.phase == "wolf_chat" for event in snapshot.events)


async def test_full_sample_runner_reaches_game_over_with_core_flow_covered() -> None:
    game, report = await run_full_sample_game(human_player_id=8, script_variant=0)

    assert report.passed, report.findings
    assert report.completed
    assert game.phase == Phase.GAME_OVER
    assert report.winner in {"好人阵营", "狼人阵营"}
    assert report.day_speech_count >= 8
    assert report.wolf_chat_night_count >= 1
    assert {"wolf_chat", "night", "day_speech", "vote", "result"}.issubset(set(report.phases_covered))


async def test_full_sample_game_meets_playability_acceptance_bar() -> None:
    game, report = await run_full_sample_game(human_player_id=8, script_variant=0)
    payload = _report_dict(report)

    assert report.passed, report.findings
    assert report.completed
    assert report.winner in {"好人阵营", "狼人阵营"}
    assert report.wolf_chat_night_count >= 3
    assert report.night_count >= 3
    assert len({speech.day for speech in game.speeches if speech.speech_type == "day"}) >= 3
    assert len({vote.day for vote in game.votes if vote.vote_type == "exile"}) >= 3
    assert len([speech for speech in game.speeches if speech.speech_type == "last_words"]) >= 2
    assert report.seer_counterclaim_count >= 3
    assert report.claim_response_count >= 8
    assert report.role_strategy_signal_count >= 5
    assert {"狼人", "预言家", "女巫", "猎人", "平民"}.issubset(set(report.role_strategy_roles))
    wolf_stances = {record.stance_to_previous for record in game.wolf_chat_records}
    assert {"proposal", "support", "switch"}.issubset(wolf_stances)
    assert report.wolf_chat_stance_variety >= 2
    assert report.day_angle_variety >= 5
    assert report.vote_intent_speech_count >= 8
    assert report.max_vote_share <= 0.72
    assert payload["role_strategy_signal_count"] == report.role_strategy_signal_count
    assert payload["role_strategy_roles"] == report.role_strategy_roles


def test_sample_runner_cli_returns_success_for_fallback_sample() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        assert sample_runner_main(["--json"]) == 0


def test_sample_runner_cli_returns_success_for_two_day_sample() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        assert sample_runner_main(["--json", "--days", "2"]) == 0


def test_sample_runner_cli_returns_success_for_full_game() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        assert sample_runner_main(["--json", "--full"]) == 0


def test_fallback_vote_respects_good_role_claims_by_camp_view() -> None:
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    players = [
        SeatRef(player_id=0, seat_no=1, name="玩家1", alive=True, role=RoleName.WEREWOLF, camp=Camp.WEREWOLF),
        SeatRef(player_id=1, seat_no=2, name="玩家2", alive=True, role=RoleName.WITCH, camp=Camp.VILLAGER),
        SeatRef(player_id=2, seat_no=3, name="玩家3", alive=True, role=RoleName.VILLAGER, camp=Camp.VILLAGER),
    ]
    claim = PublicClaimEvidence(
        day=1,
        speaker_id=1,
        speaker_seat_no=2,
        claimed_role=RoleName.WITCH,
        source_text="我是女巫，昨晚毒口要按票型复盘。",
    )

    villager_context = AIContext(
        player_id=2,
        role=RoleName.VILLAGER,
        day=1,
        phase="day_vote",
        visible_state="",
        allowed_target_ids=[0, 1],
        prompt="test",
        structured=AgentVisibleContext(
            self_player=players[2],
            day=1,
            night_id=1,
            phase="day_vote",
            public_players=players,
            public_claims=[claim],
        ),
    )
    wolf_context = AIContext(
        player_id=0,
        role=RoleName.WEREWOLF,
        day=1,
        phase="day_vote",
        visible_state="",
        allowed_target_ids=[1, 2],
        prompt="test",
        structured=AgentVisibleContext(
            self_player=players[0],
            day=1,
            night_id=1,
            phase="day_vote",
            public_players=players,
            public_claims=[claim],
        ),
    )

    assert runtime._select_fallback_target(villager_context, runtime._score_targets(villager_context)) == 0
    assert runtime._select_fallback_target(wolf_context, runtime._score_targets(wolf_context)) == 1


def test_sample_runner_cli_returns_success_for_playability_matrix() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        assert sample_runner_main(["--json", "--matrix"]) == 0


def test_sample_runner_cli_returns_success_for_balance_matrix() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        assert sample_runner_main(["--json", "--balance"]) == 0


async def test_playability_matrix_covers_core_human_roles_and_passes() -> None:
    results = await run_playability_matrix(use_live_runtime=False)

    roles = {game.human_player.role for game, _ in results}
    assert {RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER}.issubset(roles)
    assert all(report.passed for _, report in results)
    assert all(report.completed for _, report in results)


async def test_balance_matrix_observes_both_camps_across_fixed_cases() -> None:
    results = await run_balance_matrix(use_live_runtime=False)
    payload = _balance_payload(results)

    assert payload["case_count"] == 36
    assert payload["completed_count"] == 36
    assert payload["winner_variety"] >= 2
    assert payload["passed"] is True


async def test_wolf_counterclaim_sample_keeps_vote_split_without_hard_anchor() -> None:
    game, report = await run_full_sample_game(human_player_id=2, script_variant=1)

    day_two_votes = [vote for vote in game.votes if vote.vote_type == "exile" and vote.day == 2]
    assert report.passed
    assert day_two_votes
    assert report.max_vote_share <= 0.85
    assert len({vote.target_id for vote in day_two_votes}) >= 3


async def test_default_rule_rejects_sheriff_extension_api() -> None:
    snapshot = await api_create_game(CreateGameRequest(player_count=12))

    try:
        await api_resolve_sheriff(
            snapshot.game_id,
            SheriffRequest(run_for_sheriff=True),
        )
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
        assert "未启用警长" in getattr(exc, "detail", "")
    else:
        raise AssertionError("默认主规则不应允许警长扩展 API")


async def test_public_observations_are_broadcast_without_wolf_chat_leakage() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime(
        {0: 2, 1: 2, 2: 0, 3: None},
        strict=True,
        contents={
            "wolf_chat": "今晚先刀3号，这张牌像预言家，明天别聊漏队友。",
            "day_speech": "1号这轮发言像在做身份，前后站边不闭环，今天要进票池。",
        },
    )

    await game.resolve_wolf_chat(None)
    villager_memory_text = "\n".join(item.content for item in game.agent_memories[4].observations)
    wolf_memory_text = "\n".join(item.content for item in game.agent_memories[0].observations)
    assert "今晚先刀3号" not in villager_memory_text
    assert "今晚先刀3号" not in wolf_memory_text
    assert game.camp_memories[Camp.WEREWOLF].records

    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0, 1, 2, 3, 4]
    game.speech_cursor = 0
    await game.resolve_day_speeches("")

    public_line = "玩家1（1号）白天发言"
    assert all(
        any(public_line in item.content for item in memory.public_observations)
        for memory in game.agent_memories.values()
    )
    context = game._build_agent_context(game.players[2], "day_speech", [], "test", "speak")
    assert context.structured is not None
    assert not any(public_line in item.content for item in context.structured.private_observations)
    assert any(public_line in item.content for item in game.agent_memories[2].public_observations)

    game.phase = Phase.DAY_VOTE
    await game.resolve_votes(0)
    vote_line = "放逐投票给"
    assert all(
        any(vote_line in item.content for item in memory.public_observations)
        for memory in game.agent_memories.values()
    )


def test_agent_context_contains_structured_public_evidence() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=1,
            player_name="玩家2",
            content="我跳预言家，昨晚验了4号是好人，今天先听3号怎么解释。",
            speech_type="day",
        )
    )
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="2号这个预言家我不认，4号被保得太快，像提前做身份。",
            speech_type="day",
        )
    )
    game.votes.append(
        VoteRecord(
            day=1,
            voter_id=2,
            voter_name="玩家3",
            target_id=1,
            target_name="玩家2",
            vote_type="exile",
            vote_round="day_1",
        )
    )

    context = game._build_agent_context(game.players[0], "day_speech", [], "test", "speak")

    assert context.structured is not None
    assert context.structured.recent_public_speeches[-2].speaker_seat_no == 2
    assert context.structured.recent_public_speeches[-2].mentioned_seat_nos == [4, 3]
    assert "预言家" in context.structured.recent_public_speeches[-2].stance_keywords
    assert context.structured.public_claims[-1].speaker_seat_no == 2
    assert context.structured.public_claims[-1].claimed_role == RoleName.SEER
    assert context.structured.recent_votes[-1].voter_seat_no == 3
    assert context.structured.recent_votes[-1].target_seat_no == 2


def test_public_claim_detection_does_not_treat_role_discussion_as_claim() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=1,
            player_name="玩家2",
            content="我跳预言家，昨晚验了4号是好人，今天先听3号怎么解释。",
            speech_type="day",
        )
    )
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="别把预言家牌当普通焦点打，先看谁借这个身份冲票。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[0], "day_speech", [], "test", "speak")

    assert context.structured is not None
    assert [claim.speaker_seat_no for claim in context.structured.public_claims] == [2]


def test_public_claim_detection_parses_give_wolf_inspection_wording() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=1,
            player_name="玩家2",
            content="我这里是真预视角，6号给狼人。5号像悍跳抢节奏。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[3], "day_vote", [0, 1, 2, 3, 4, 5], "test", "vote")

    assert context.structured is not None
    claim = context.structured.public_claims[-1]
    assert claim.claimed_role == RoleName.SEER
    assert claim.inspected_target_seat_no == 6
    assert claim.inspected_result == "狼人"


def test_visible_brief_avoids_raw_json_field_names() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="我是预言家，昨晚验了5号是好人，今天我会压1号。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    brief = context.visible_brief()

    assert "你是1号" in brief
    assert "底牌狼人" in brief
    assert "狼队友：2号" in brief
    assert "公开身份宣称" in brief
    assert "3号声称预言家" in brief
    assert "可选择的目标号位：3号、4号、5号" in brief
    assert "target_id" not in brief
    assert "player_id" not in brief
    assert "{" not in brief
    assert "}" not in brief


def test_agent_visible_context_does_not_expose_raw_memory_objects() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.agent_memories[0].private_observations.append(
        PrivateObservation(day=1, night_id=1, phase="night", content="不应通过 raw memory 透传。")
    )
    game.agent_states[0].private_summary = "我是1号狼人，保留摘要。"

    context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    fields = set(context.structured.model_fields)

    assert "memory" not in fields
    assert "camp_memory" not in fields
    assert "agent_state" not in fields
    assert context.structured.private_summary == "我是1号狼人，保留摘要。"


def test_advisor_uses_visible_context_without_targeting_wolf_teammates() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="我是预言家，昨晚验了5号是好人，今天我会压1号。",
            speech_type="day",
        )
    )
    context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")

    assert context.structured is not None
    advice = advise(context.structured)

    assert advice.recommended_target_id == 2
    assert all(item.player_id != 1 for item in advice.suspicions)
    assert any("预言家" in "；".join(item.reasons) for item in advice.suspicions if item.player_id == 2)
    prompt = game.decision_pipeline._build_evidence_text(context)
    assert "基于你可见信息的怀疑排序" in prompt
    assert "3号" in prompt


async def test_visible_brief_renders_messages_as_table_language() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime({0: 2}, strict=True, contents={"wolf_chat": "今晚建议刀3号，先拆预言家口径。"})

    await game.resolve_wolf_chat(None)

    context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    brief = context.visible_brief()

    assert "本次新增可见信息" in brief
    assert "狼人夜聊" in brief
    assert "建议刀3号" in brief
    assert "/wolf_chat/" not in brief
    assert "/whisper" not in brief
    assert "night_action" not in brief


async def test_agent_visible_context_tracks_incremental_messages() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime({0: 2, 1: 2}, strict=True, contents={"wolf_chat": "今晚建议刀3号，先拆预言家口径。"})

    await game.resolve_wolf_chat(None)
    first_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    second_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    await game.resolve_wolf_chat(None)
    third_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")

    assert first_context.structured is not None
    assert second_context.structured is not None
    assert third_context.structured is not None
    assert len(first_context.structured.new_visible_messages) == 1
    assert second_context.structured.new_visible_messages == []
    assert len(third_context.structured.new_visible_messages) == 1
    assert third_context.structured.new_visible_messages[0].speaker_id == 1


async def test_table_message_log_separates_public_wolf_and_private_views() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime({0: 2, 1: 2, 2: 0}, strict=True)

    await game.resolve_wolf_chat(None)

    assert game.message_log
    wolf_message = game.message_log[-1]
    assert wolf_message.message_type == "whisper"
    assert wolf_message.visibility == "wolf"
    assert wolf_message.night_id == 1
    assert wolf_message.target_id == 2
    assert {0, 1}.issubset(set(wolf_message.visible_to_player_ids))
    assert 2 not in wolf_message.visible_to_player_ids

    wolf_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    seer_context = game._build_agent_context(game.players[2], "night_action", [0, 1, 3, 4], "test", "inspect")
    assert wolf_context.structured is not None
    assert seer_context.structured is not None
    assert any(message.message_type == "whisper" for message in wolf_context.structured.visible_messages)
    assert not any(message.message_type == "whisper" for message in seer_context.structured.visible_messages)

    game.phase = Phase.NIGHT
    game.wolf_consensus_target_id = 3
    await game.resolve_night(game_action("skip", None))

    seer_private = [
        message
        for message in game.message_log
        if message.message_type == "night_action" and message.action == "inspect"
    ]
    assert seer_private
    assert seer_private[-1].visibility == "private"
    assert seer_private[-1].visible_to_player_ids == [2]
    private_events = [event for event in game.events if event.visibility == "private"]
    assert private_events
    assert private_events[-1].visible_to_player_ids == [2]
    assert any("查验" in event.message for event in game._visible_events_for_player(game.players[2]))
    assert not any("查验" in event.message for event in game._visible_events_for_player(game.players[0]))
    assert not any("查验" in event.message for event in game._visible_events_for_player(game.human_player))
    wolf_day_context = game._build_agent_context(game.players[0], "day_speech", [], "test", "speak")
    seer_day_context = game._build_agent_context(game.players[2], "day_speech", [], "test", "speak")
    assert wolf_day_context.structured is not None
    assert seer_day_context.structured is not None
    assert not any(message.action == "inspect" for message in wolf_day_context.structured.visible_messages)
    assert any(message.action == "inspect" for message in seer_day_context.structured.visible_messages)


async def test_wolf_day_agent_context_receives_summary_not_raw_night_chat() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime(
        {0: 2, 1: 2},
        strict=True,
        contents={"wolf_chat": "今晚建议刀3号，明天别聊漏队友，白天把票口推给4号。"},
    )
    game.rule_profile.wolf_chat_rounds = 1

    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(None)
    assert game.phase == Phase.NIGHT

    wolf_day_context = game._build_agent_context(game.players[0], "day_speech", [], "test", "speak")

    assert wolf_day_context.structured is not None
    assert not any(message.message_type == "whisper" for message in wolf_day_context.structured.visible_messages)
    assert not any(message.message_type == "whisper" for message in wolf_day_context.structured.new_visible_messages)
    assert "今晚建议刀3号" not in wolf_day_context.visible_state
    assert "明天别聊漏队友" not in wolf_day_context.visible_state
    assert wolf_day_context.structured.wolf_history_summaries
    assert "过往夜晚" in wolf_day_context.structured.wolf_history_summaries[-1]


async def test_agent_view_has_identity_status_and_turn_quota_contract() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game._prepare_wolf_chat_order()

    wolf_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")
    assert wolf_context.structured is not None
    assert wolf_context.structured.status_map == {0: "ALIVE", 1: "ALIVE", 2: "ALIVE", 3: "ALIVE", 4: "ALIVE"}
    assert wolf_context.structured.known_role_map == {0: RoleName.WEREWOLF, 1: RoleName.WEREWOLF}
    assert wolf_context.structured.whisper_quota[0] == 1
    assert wolf_context.structured.whisper_quota[1] == 1
    assert wolf_context.structured.talk_quota[0] == 0

    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0, 1, 2, 3, 4]
    game.speech_cursor = 2
    seer_context = game._build_agent_context(game.players[2], "day_speech", [], "test", "speak")
    assert seer_context.structured is not None
    assert seer_context.structured.known_role_map == {2: RoleName.SEER}
    assert seer_context.structured.talk_quota[0] == 0
    assert seer_context.structured.talk_quota[2] == 1
    assert seer_context.structured.talk_quota[4] == 1
    assert "当前公开发言剩余额度：你还能发言1次" in seer_context.visible_brief()


def test_agent_public_players_do_not_reuse_human_wolf_snapshot_view() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.DAY_SPEECH

    seer_context = game._build_agent_context(game.players[2], "day_speech", [], "test", "speak")

    assert seer_context.structured is not None
    human_ref = next(player for player in seer_context.structured.public_players if player.player_id == 0)
    wolf_teammate_ref = next(player for player in seer_context.structured.public_players if player.player_id == 1)
    assert human_ref.role is None
    assert human_ref.camp is None
    assert wolf_teammate_ref.role is None
    assert wolf_teammate_ref.camp is None
    assert seer_context.structured.known_role_map == {2: RoleName.SEER}


def test_wolf_teammate_roles_stay_in_private_wolf_fields_not_public_players() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)

    wolf_context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4], "test", "wolf_chat")

    assert wolf_context.structured is not None
    public_teammate = next(player for player in wolf_context.structured.public_players if player.player_id == 1)
    assert public_teammate.role is None
    assert public_teammate.camp is None
    assert wolf_context.structured.known_role_map == {0: RoleName.WEREWOLF, 1: RoleName.WEREWOLF}
    assert wolf_context.structured.wolf_teammates[0].role == RoleName.WEREWOLF


async def test_day_speech_goal_is_role_specific_without_private_leakage() -> None:
    roles = [
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
    game = make_game(roles, human_player_id=8)
    runtime = FakeRuntime(
        contents={
            "day_speech": "2号这轮站边太快，理由没有闭环，今天我会优先看这个位置。",
        }
    )
    game.runtime = runtime
    game.phase = Phase.NIGHT
    game.wolf_consensus_target_id = 9

    await game.resolve_night(game_action("skip", None))
    if game.phase == Phase.LAST_WORDS:
        while game.phase == Phase.LAST_WORDS:
            await game.resolve_last_words("")
    assert game.phase == Phase.DAY_SPEECH

    # 推到预言家发言位。
    while game.current_speaker_id != 4:
        await game.resolve_day_speeches("")

    await game.resolve_day_speeches("")
    seer_context = runtime.contexts[-1]
    assert seer_context.player_id == 4
    assert "你是预言家" in seer_context.prompt
    assert "验人结果" in seer_context.prompt
    assert seer_context.structured.seer_inspections

    # 其他玩家的私有观察不能拿到预言家的验人结果。
    villager_memory = "\n".join(item.content for item in game.agent_memories[8].observations)
    assert "夜查验" not in villager_memory

    wolf_context = game._build_agent_context(game.players[0], "day_speech", [], game._day_speech_goal(game.players[0]), "speak")
    assert "你是狼人" in wolf_context.prompt
    assert "存活狼队友号位" in wolf_context.prompt


async def test_seer_agent_uses_typed_inspection_fact_without_chinese_memory_marker() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.runtime = FakeRuntime({2: 0}, strict=True)
    game.phase = Phase.NIGHT
    game.wolf_consensus_target_id = 4

    await game.resolve_night(None)

    seer_memory = game.agent_memories[2].private_observations[-1]
    seer_memory.content = "结构化查验事实已写入。"
    context = game._build_agent_context(game.players[2], "night_action", [0, 1, 3, 4], "test", "inspect")
    runtime = OpenAIAgentRuntime()

    assert context.structured.seer_inspections
    assert runtime._seer_result_history(context) == [(1, "狼人")]
    assert runtime._seer_inspection_history(context) == [0]


def test_witch_fallback_reads_typed_night_info_without_prompt_text() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.phase = Phase.NIGHT
    game.wolf_consensus_target_id = 2
    context = game._build_agent_context(game.players[2], "night_action", [0, 1, 2, 3, 4], "test", "witch_action")
    context.prompt = ""
    context.visible_state = ""
    runtime = OpenAIAgentRuntime()

    target_id, reason = runtime._fallback_witch_target(context, [0, 1, 3, 4])

    assert context.structured.witch_night_info is not None
    assert context.structured.witch_night_info.wolf_target_id == 2
    assert target_id == 2
    assert "救3号" in reason


async def test_hunter_poison_block_uses_death_fact_not_global_flag() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=3)
    game.phase = Phase.NIGHT
    game.wolf_consensus_target_id = 3
    game.runtime = FakeRuntime({1: 2}, strict=True)

    await game.resolve_night(None)
    if game.phase == Phase.LAST_WORDS:
        while game.phase == Phase.LAST_WORDS:
            await game.resolve_last_words("")

    hunter_death = game._death_fact_for(2)
    assert hunter_death is not None
    assert hunter_death.cause == "witch_poison"
    assert hunter_death.can_hunter_shoot is False
    assert game.pending_hunter_id is None


async def test_visible_messages_for_wolf_chat_filters_previous_night_inside_helper() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    game.phase = Phase.WOLF_CHAT
    game.runtime = FakeRuntime({0: 2, 1: 2}, strict=True, contents={"wolf_chat": "今晚刀3号，先拆查验压力。"})

    await game.resolve_wolf_chat(None)
    await game.resolve_wolf_chat(HumanNightAction(action_type="wolf_confirm", target_id=2, chat_content="收口3号。"))
    game.night_id = 2
    game.day = 2
    game.phase = Phase.WOLF_CHAT

    visible = game._visible_messages_for_player(game.players[0], phase_scope="wolf_chat")

    assert visible == []


async def test_ai_decisions_use_structured_pipeline_for_main_phases() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime({1: 0, 2: 0, 3: 0}, strict=False)
    game.phase = Phase.DAY_VOTE

    await game.resolve_votes(0)

    phases = [context.phase for context in game.runtime.contexts]
    assert phases == ["day_vote", "day_vote", "day_vote"]
    assert all(context.structured is not None for context in game.runtime.contexts)
    assert all(context.structured.legal_actions for context in game.runtime.contexts)
    assert game.decision_audits
    assert any(audit.action == "vote" for audit in game.decision_audits)


def test_sanitize_decision_removes_prompt_and_json_leakage() -> None:
    runtime = OpenAIAgentRuntime()
    decision = runtime._sanitize_decision(
        AgentDecision(
            action="speak",
            content='```json\n{"target_id": 3, "reason": "根据提示词"}\n```\n根据提示词和结构化上下文，我选择 target_id=3。',
        )
    )

    assert "target_id" not in decision.content
    assert "结构化上下文" not in decision.content
    assert "根据提示词" not in decision.content
    assert "```" not in decision.content


def test_fallback_day_speech_uses_real_seats_and_witch_skips_blind_poison() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=1,
            player_name="玩家2",
            content="3号这轮站边来得太早，后面如果还不补逻辑，我今天会投3号。",
            speech_type="day",
        )
    )

    speech_context = game._build_agent_context(game.players[2], "day_speech", [], "test", "speak")
    speech = runtime._fallback_decision(speech_context).content
    assert "前置位" not in speech
    mentioned_seats = {int(item) for item in re.findall(r"(\d+)号", speech)}
    assert mentioned_seats
    assert all(1 <= seat <= 6 for seat in mentioned_seats)
    assert "我先接" not in speech
    assert "player_id" not in speech
    assert any(token in speech for token in ["链路", "理由", "态度", "推", "票", "解释", "说明"])

    night_context = game._build_agent_context(
        game.players[2],
        "night_action",
        [0, 1, 3, 4, 5],
        "女巫夜间行动。",
        "witch_action",
    )
    witch_decision = runtime._fallback_decision(night_context)
    assert witch_decision.target_id is None


def test_fallback_witch_does_not_blind_save_first_night_non_self_target() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WITCH, RoleName.SEER, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.NIGHT
    game.day = 1
    game.wolf_consensus_target_id = 3

    context = game._build_agent_context(game.players[1], "night_action", [0, 2, 3, 4], "女巫夜间行动。", "witch_action")
    decision = runtime._fallback_decision(context)

    assert decision.target_id is None
    assert "保留药瓶" in decision.reason


def test_fallback_witch_saves_public_power_claim_after_first_night() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WITCH, RoleName.SEER, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.NIGHT
    game.day = 2
    game.wolf_consensus_target_id = 2
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="我是预言家，昨晚验了1号是狼人，今天票别散。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[1], "night_action", [0, 2, 3, 4], "女巫夜间行动。", "witch_action")
    decision = runtime._fallback_decision(context)

    assert decision.target_id == 2
    assert "救3号" in decision.reason


def test_fallback_opening_day_speeches_are_role_driven_not_echo_chain() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.DAY_SPEECH
    speeches = []

    for player in game.players[:5]:
        context = game._build_agent_context(player, "day_speech", [], "test", "speak")
        content = runtime._fallback_decision(context).content
        speeches.append(content)

    assert_non_ai_table_speech(speeches)
    joined = "\n".join(speeches)
    assert "我先接" not in joined
    assert "接入" not in joined
    assert "上下文" not in joined
    assert joined.count("前置位") == 0
    assert len(set(speeches)) == len(speeches)
    assert any("票口" in speech or "收窄" in speech or "预言家" in speech or "验" in speech for speech in speeches)
    assert any("验" in speech or "预言家" in speech or "夜里" in speech for speech in speeches)
    assert any("做局" in speech or "轮次" in speech or "进票池" in speech for speech in speeches)
    assert any("枪" in speech or "承担后果" in speech for speech in speeches)


def test_fallback_vote_uses_public_suspicion_evidence() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=1,
                player_name="玩家2",
                content="4号这个位置我不放，前面说保3号，后面又想票3号，逻辑是断的。",
                speech_type="day",
            ),
            SpeechRecord(
                day=1,
                player_id=2,
                player_name="玩家3",
                content="我也会看4号，4号打人只给结论不给链路，今天可以进票池。",
                speech_type="day",
            ),
            SpeechRecord(
                day=1,
                player_id=3,
                player_name="玩家4",
                content="我是预言家，2号金水，今天先别动我。",
                speech_type="day",
            ),
        ]
    )

    context = game._build_agent_context(game.players[4], "day_vote", [0, 1, 2, 3, 4, 5], "test", "vote")
    decision = runtime._fallback_decision(context)

    assert decision.target_id in {2, 3}
    assert decision.target_id != 4
    assert "公开声称" in decision.reason or "公开发言点到" in decision.reason


def test_fallback_vote_pushes_non_claimed_public_suspicion_target() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=1,
                player_name="玩家2",
                content="4号这个位置我不放，前面说保3号，后面又想票3号，逻辑是断的。",
                speech_type="day",
            ),
            SpeechRecord(
                day=1,
                player_id=2,
                player_name="玩家3",
                content="我也会看4号，4号打人只给结论不给链路，今天可以进票池。",
                speech_type="day",
            ),
        ]
    )

    context = game._build_agent_context(game.players[4], "day_vote", [0, 1, 2, 3, 4, 5], "test", "vote")
    decision = runtime._fallback_decision(context)

    assert decision.target_id in {2, 3}
    assert decision.target_id != 4
    assert "公开发言点到" in decision.reason


def test_good_fallback_vote_does_not_blindly_follow_late_fake_seer_counterclaim() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.DAY_VOTE
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=3,
                player_name="玩家4",
                content="我是预言家，昨晚验了1号是狼人，今天票别散。",
                speech_type="day",
            ),
            SpeechRecord(
                day=1,
                player_id=0,
                player_name="玩家1",
                content="我也把身份拍了：预言家，昨晚验4号是狼人。4号如果是真预，不该把票口收得那么急。",
                speech_type="day",
            ),
        ]
    )

    context = game._build_agent_context(game.players[4], "day_vote", [0, 1, 2, 3, 4, 5], "test", "vote")
    decision = runtime._fallback_decision(context)

    assert decision.target_id == 0


def test_wolf_teammates_countervote_when_fake_seer_teammate_is_checked() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.DAY_VOTE
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=3,
                player_name="玩家4",
                content="我是预言家，昨晚验了1号是狼人，今天票别散。",
                speech_type="day",
            ),
            SpeechRecord(
                day=1,
                player_id=0,
                player_name="玩家1",
                content="我也把身份拍了：预言家，昨晚验4号是狼人。4号如果是真预，不该把票口收得那么急。",
                speech_type="day",
            ),
        ]
    )

    targets = []
    for player_id in [1, 2]:
        context = game._build_agent_context(game.players[player_id], "day_vote", [0, 1, 2, 3, 4, 5], "test", "vote")
        targets.append(runtime._fallback_decision(context).target_id)

    assert targets == [3, 3]


def test_counterclaim_vote_split_avoids_all_good_players_piling_latest_seer_claim() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 2
    game.phase = Phase.DAY_VOTE
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=2,
                player_name="玩家3",
                content="我是预言家，昨晚验到1号是狼人，今天票别散。",
                speech_type="day",
            ),
            SpeechRecord(
                day=2,
                player_id=1,
                player_name="玩家2",
                content="我不认3号，我跳预言家，夜里验到4号狼人。",
                speech_type="day",
            ),
        ]
    )

    targets = []
    for player_id in [3, 4, 5]:
        context = game._build_agent_context(game.players[player_id], "day_vote", [0, 1, 2, 3, 4, 5], "test", "vote")
        targets.append(runtime._fallback_decision(context).target_id)

    assert len(set(targets)) >= 2
    assert any(target != 1 for target in targets)


def test_multi_night_true_seer_chain_keeps_good_votes_on_claimed_wolf() -> None:
    roles = [
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
    game = make_game(roles, human_player_id=11)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 2
    game.phase = Phase.DAY_VOTE
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=4,
                player_name="玩家5",
                content="我是预言家，昨晚验到1号是狼人。今天票别散。",
                speech_type="day",
            ),
            SpeechRecord(
                day=2,
                player_id=1,
                player_name="玩家2",
                content="我不认5号，我跳预言家，夜里验到5号狼人。",
                speech_type="day",
            ),
            SpeechRecord(
                day=2,
                player_id=4,
                player_name="玩家5",
                content="我是预言家，昨晚验到2号是狼人。两天验人链是1号狼人、2号狼人，今天先出2号。",
                speech_type="day",
            ),
        ]
    )

    targets = []
    for player_id in [5, 6, 7, 8, 9, 10, 11]:
        context = game._build_agent_context(game.players[player_id], "day_vote", list(range(12)), "test", "vote")
        targets.append(runtime._fallback_decision(context).target_id)

    assert targets.count(1) >= 2
    assert targets.count(4) <= 2
    assert len(set(targets)) >= 2


def test_live_seer_chain_keeps_some_vote_uncertainty_before_endgame() -> None:
    roles = [
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
    game = make_game(roles, human_player_id=11)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 2
    game.phase = Phase.DAY_VOTE
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=4,
                player_name="玩家5",
                content="我是预言家，昨晚验到1号是狼人。今天票别散。",
                speech_type="day",
            ),
            SpeechRecord(
                day=2,
                player_id=4,
                player_name="玩家5",
                content="我把验人链报完整：1号狼人，2号狼人。今天先出2号。",
                speech_type="day",
            ),
            SpeechRecord(
                day=2,
                player_id=1,
                player_name="玩家2",
                content="5号这条链太顺了，我不认这个查杀，今天先验票型。",
                speech_type="day",
            ),
        ]
    )

    targets = []
    for player_id in [5, 6, 7, 8, 9, 10, 11]:
        context = game._build_agent_context(game.players[player_id], "day_vote", list(range(12)), "test", "vote")
        targets.append(runtime._fallback_decision(context).target_id)

    assert targets.count(1) >= 3
    assert len(set(targets)) >= 2


def test_dead_seer_multi_wolf_chain_guides_endgame_votes_off_hunter() -> None:
    roles = [
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
    game = make_game(roles, human_player_id=6)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 4
    game.phase = Phase.DAY_VOTE
    for player_id in [0, 1, 2, 4, 5, 8, 9]:
        game.players[player_id].alive = False
    game.speeches.extend(
        [
            SpeechRecord(
                day=1,
                player_id=4,
                player_name="玩家5",
                content="我是预言家，昨晚验到1号是狼人。",
                speech_type="day",
            ),
            SpeechRecord(
                day=2,
                player_id=4,
                player_name="玩家5",
                content="我把验人链报完整：1号狼人，3号狼人，4号狼人。今天继续按验人链收。",
                speech_type="last_words",
            ),
            SpeechRecord(
                day=4,
                player_id=3,
                player_name="玩家4",
                content="我觉得7号带枪压力太大，今天先把7号推出去。",
                speech_type="day",
            ),
        ]
    )
    alive_ids = [player.id for player in game.alive_players()]

    targets = []
    for player_id in [6, 7, 10, 11]:
        context = game._build_agent_context(game.players[player_id], "day_vote", alive_ids, "test", "vote")
        targets.append(runtime._fallback_decision(context).target_id)

    assert targets.count(3) >= 3
    assert 6 not in targets


def test_fallback_day_speech_does_not_anchor_on_dead_player_evidence() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.DAY_SPEECH
    game.players[1].alive = False
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="4号逻辑不闭合，我今天先压4号。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="1号站边太快，我今天会继续看1号。", speech_type="day"),
        ]
    )

    context = game._build_agent_context(game.players[3], "day_speech", [], "test", "speak")
    decision = runtime._fallback_decision(context)

    assert "2号" not in decision.content
    assert "3号" in decision.content or "1号" in decision.content


def test_claimed_wolf_vote_split_keeps_majority_without_unanimous_pileon() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=6)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 2
    game.phase = Phase.DAY_VOTE
    game.speeches.append(
        SpeechRecord(
            day=2,
            player_id=1,
            player_name="玩家2",
            content="我是预言家，昨晚验了1号是狼人，今天先出1号，票别散。",
            speech_type="day",
        )
    )

    targets = []
    for player_id in [2, 3, 4, 5, 6]:
        context = game._build_agent_context(game.players[player_id], "day_vote", [0, 1, 2, 3, 4, 5, 6], "test", "vote")
        targets.append(runtime._fallback_decision(context).target_id)

    assert targets.count(0) >= 2
    assert len(set(targets)) >= 2


def test_fallback_wolf_chat_prioritizes_claimed_power_role_and_explains_value() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="我是预言家，昨晚验了5号是好人，今天我会压1号的发言。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[0], "wolf_chat", [2, 3, 4, 5], "test", "wolf_chat")
    decision = runtime._fallback_decision(context)

    assert decision.target_id == 2
    assert "3号" in decision.content
    assert "预言家" in decision.content
    assert "带队" in decision.content or "归票" in decision.content


def test_wolf_chat_target_keeps_public_power_claim_pressure_instead_of_forced_backup() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.WOLF_CHAT
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=2,
            player_name="玩家3",
            content="我是预言家，昨晚验了5号是好人，今天我会压1号的发言。",
            speech_type="day",
        )
    )
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=1,
            night_id=1,
            round_id=1,
            player_id=0,
            speaker_seat_no=1,
            player_name="玩家1",
            content="我先提主刀：今晚动3号。3号已经公开往预言家方向聊，明天容易带队归票。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            stance_to_previous="proposal",
            is_valid_target=True,
        )
    )

    context = game._build_agent_context(game.players[1], "wolf_chat", [2, 3, 4, 5], "test", "wolf_chat")

    assert runtime._select_wolf_chat_target(context, [2, 3, 4, 5]) == 2


def test_playability_flags_awkward_repeated_connectives() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="1号但逻辑没闭合但落点不干净。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="我投1号，因为他发言和票型都不顺。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="2号身份压力很重，所以我要压他。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game)

    assert any("重复转折" in finding for finding in report.findings)


def test_playability_flags_cross_night_wolf_summary_repeated_as_current_chat() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=4)
    game.wolf_chat_records.append(
        WolfChatRecord(
            day=2,
            night_id=2,
            round_id=1,
            turn_index=0,
            player_id=0,
            speaker_seat_no=1,
            player_name="玩家1",
            content="第1天夜晚#1: 最终刀口5号，来源proposal_vote；今晚继续照这个摘要走。",
            proposed_target_id=2,
            proposed_target_seat_no=3,
            is_valid_target=True,
        )
    )

    report = evaluate_playability(game)

    assert any("复述历史摘要" in finding for finding in report.findings)


def test_playability_flags_seer_claim_without_counterplay() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="我是预言家，昨晚验了5号是好人，今天先看1号。", speech_type="day"),
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="3号说完以后我先不打死，今天听站边。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="我也先听3号后续，票型再看。", speech_type="day"),
            SpeechRecord(day=1, player_id=3, player_name="玩家4", content="5号先放一轮，1号要解释发言。", speech_type="day"),
            SpeechRecord(day=1, player_id=4, player_name="玩家5", content="我今天投1号，因为他发言和票型都不顺。", speech_type="day"),
            SpeechRecord(day=1, player_id=5, player_name="玩家6", content="我压1号，理由是他只给态度不给链路。", speech_type="day"),
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="我继续看4号，他保人太快。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="今天票别散，先归1号。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game)

    assert any("身份博弈不足" in finding for finding in report.findings)


def test_playability_does_not_require_counterplay_for_last_position_seer_claim() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="我今天先压2号，因为他发言和票型都不顺。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="1号只给态度不给链路，我票会看1号。", speech_type="day"),
            SpeechRecord(day=1, player_id=3, player_name="玩家4", content="今天看谁补票最急，2号和5号先进视野。", speech_type="day"),
            SpeechRecord(day=1, player_id=4, player_name="玩家5", content="我就是普通视角，先看1号有没有改口。", speech_type="day"),
            SpeechRecord(day=1, player_id=5, player_name="玩家6", content="我压1号，理由是他一直绕开投票倾向。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="我是预言家，昨晚验了5号是好人，遗留一下验人。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game)

    assert not any("身份博弈不足" in finding for finding in report.findings)
    assert not any("身份宣称回应不足" in finding for finding in report.findings)


def test_fallback_seer_day_speech_uses_private_inspection() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=3)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game._remember_private(
        game.players[1],
        "第1夜查验 玩家1（1号） -> 狼人。",
        {"target_id": 0, "target_seat_no": 1, "result": "狼人"},
    )

    context = game._build_agent_context(game.players[1], "day_speech", [], "test", "speak")
    decision = runtime._fallback_decision(context)

    assert "1号" in decision.content
    assert "狼人" in decision.content
    assert "验" in decision.content or "夜里" in decision.content


def test_fallback_seer_night_inspection_avoids_rechecking_known_target() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.phase = Phase.NIGHT
    game._remember_private(
        game.players[1],
        "第1夜查验 玩家1（1号） -> 狼人。",
        {"target_id": 0, "target_seat_no": 1, "result": "狼人"},
    )
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=0,
            player_name="玩家1",
            content="我跳预言家，昨晚验了3号是好人，今天别动我。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[1], "night_action", [0, 2, 3, 4], "test", "inspect")
    decision = runtime._fallback_decision(context)

    assert decision.target_id in {2, 3, 4}
    assert decision.target_id != 0


def test_fallback_seer_day_speech_reports_multi_night_inspection_chain() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=4)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game._remember_private(
        game.players[1],
        "第1夜查验 玩家1（1号） -> 狼人。",
        {"target_id": 0, "target_seat_no": 1, "result": "狼人"},
    )
    game._remember_private(
        game.players[1],
        "第2夜查验 玩家3（3号） -> 好人。",
        {"target_id": 2, "target_seat_no": 3, "result": "好人"},
    )

    context = game._build_agent_context(game.players[1], "day_speech", [], "test", "speak")
    decision = runtime._fallback_decision(context)

    assert "验人链" in decision.content or "信息不是单点" in decision.content
    assert "1号狼人" in decision.content
    assert "3号好人" in decision.content
    assert "出票" in decision.content or "票口" in decision.content


def test_fallback_wolf_can_fake_seer_claim_without_leaking_wolf_context() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.players[0].strategy_style = "控场悍跳流"
    game.phase = Phase.DAY_SPEECH

    context = game._build_agent_context(game.players[0], "day_speech", [], "test", "speak")
    decision = runtime._fallback_decision(context)

    assert "预言家" in decision.content
    assert "验" in decision.content or "摸" in decision.content
    assert "狼人" in decision.content or "好人" in decision.content
    assert "狼队友" not in decision.content
    assert "刀口" not in decision.content
    assert "狼人夜聊" not in decision.content


def test_fake_seer_claim_enters_public_claim_evidence_for_later_agents() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.phase = Phase.DAY_SPEECH
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=0,
            player_name="玩家1",
            content="我也把身份拍了：预言家，昨晚验3号是狼人。今天先别散票。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[3], "day_speech", [], "test", "speak")

    assert context.structured is not None
    assert context.structured.public_claims
    claim = context.structured.public_claims[-1]
    assert claim.speaker_seat_no == 1
    assert claim.claimed_role == RoleName.SEER
    assert claim.inspected_target_seat_no == 3
    assert claim.inspected_result == "狼人"


def test_public_seer_claim_evidence_extracts_multi_target_chain() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.append(
        SpeechRecord(
            day=3,
            player_id=2,
            player_name="玩家3",
            content="我把验人链报完整：1号狼人，3号狼人，6号好人。今天先围绕3号出票。",
            speech_type="last_words",
        )
    )

    claims = game._public_claim_evidence()
    parsed = [(claim.inspected_target_seat_no, claim.inspected_result) for claim in claims if claim.claimed_role == RoleName.SEER]

    assert (1, "狼人") in parsed
    assert (3, "狼人") in parsed
    assert (6, "好人") in parsed


def test_midgame_fallback_speech_does_not_say_first_day() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 4
    game.phase = Phase.DAY_SPEECH
    game.speeches.append(
        SpeechRecord(
            day=4,
            player_id=1,
            player_name="玩家2",
            content="我跳预言家，昨晚验到1号是狼人，今天别散票。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[5], "day_speech", [], "test", "speak")
    decision = runtime._fallback_decision(context)

    assert "第一天" not in decision.content
    assert "第一轮" not in decision.content
    assert "今天" in decision.content or "这轮" in decision.content


def test_exile_pk_fallback_speeches_do_not_duplicate_between_candidates() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 3
    game.phase = Phase.EXILE_PK_SPEECH
    game.exile_pk_candidate_ids = [1, 2]
    game.speeches.append(
        SpeechRecord(
            day=3,
            player_id=0,
            player_name="玩家1",
            content="2号和3号都在票口里，但两个人的问题不一样。",
            speech_type="day",
        )
    )

    first_context = game._build_agent_context(game.players[1], "exile_pk_speech", [], "test", "speak")
    second_context = game._build_agent_context(game.players[2], "exile_pk_speech", [], "test", "speak")
    first = runtime._fallback_decision(first_context).content
    second = runtime._fallback_decision(second_context).content

    assert first != second
    assert "我这轮被顶到 PK 不是因为我像狼，而是有人在借势做公共坑" not in first
    assert "我这轮被顶到 PK 不是因为我像狼，而是有人在借势做公共坑" not in second
    assert "PK" in first
    assert "PK" in second


def test_claim_response_pressure_target_excludes_claim_speaker() -> None:
    roles = [
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
    ]
    game = make_game(roles, human_player_id=9)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.day = 3
    game.phase = Phase.DAY_SPEECH
    game.speeches.append(
        SpeechRecord(
            day=3,
            player_id=3,
            player_name="玩家4",
            content="我是预言家，昨晚验到1号是狼人，今天先别散票。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[9], "day_speech", [], "test", "speak")
    content = runtime._fallback_claim_response(context, [3, 0, 1]) or ""

    assert content
    assert "4号这种外置位" not in content
    assert "看4号怎么解释" not in content
    assert "压4号" not in content


def test_natural_fake_seer_claim_phrase_is_counted_as_claim_response_anchor() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="我拍身份带节奏：预言家，5号狼人。第一天不怕对跳。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="1号预言家身份我先不全认，但验人链至少落地。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="我会看谁站边1号最急，5号先解释。", speech_type="day"),
        ]
    )

    context = game._build_agent_context(game.players[3], "day_speech", [], "test", "speak")
    report = evaluate_playability(game)

    assert context.structured is not None
    assert context.structured.public_claims[-1].claimed_role == RoleName.SEER
    assert report.claim_response_count >= 2


def test_playability_counts_natural_seer_counterplay_phrases() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    game.speeches.extend(
        [
            SpeechRecord(day=1, player_id=0, player_name="玩家1", content="我拍身份带节奏：预言家，5号狼人。", speech_type="day"),
            SpeechRecord(day=1, player_id=1, player_name="玩家2", content="1号起跳后别把场面聊散，我先顺着他的验人线压一轮。", speech_type="day"),
            SpeechRecord(day=1, player_id=2, player_name="玩家3", content="1号这张我先给真预面，但5号也要解释。", speech_type="day"),
        ]
    )

    report = evaluate_playability(game)

    assert report.seer_counterclaim_count >= 1


def test_fallback_wolves_do_not_chain_fake_seer_claims_after_teammate_jump() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=5)
    runtime = OpenAIAgentRuntime()
    runtime.enabled = False
    game.players[0].strategy_style = "控场悍跳流"
    game.players[1].strategy_style = "控场悍跳流"
    game.phase = Phase.DAY_SPEECH
    game.speeches.append(
        SpeechRecord(
            day=1,
            player_id=0,
            player_name="玩家1",
            content="我直接跳预言家，昨晚验3号是狼人。今天先别散票。",
            speech_type="day",
        )
    )

    context = game._build_agent_context(game.players[1], "day_speech", [], "test", "speak")
    decision = runtime._fallback_decision(context)

    assert "我直接跳预言家" not in decision.content
    assert "我对跳预言家" not in decision.content
    assert "我也把身份拍了：预言家" not in decision.content
    assert "我这里是真预视角" not in decision.content
    assert "我跳预言家" not in decision.content


async def test_hunter_shot_uses_selected_target() -> None:
    roles = [RoleName.HUNTER, RoleName.WEREWOLF, RoleName.SEER]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.HUNTER_SHOT
    game.players[0].alive = False
    game.pending_hunter_id = 0

    await game.resolve_hunter_shot(2)

    assert game.players[2].alive is False
    assert game.players[1].alive is True


def test_sample_hunter_shot_prefers_latest_public_seer_wolf_anchor() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.HUNTER, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=3)
    game.phase = Phase.HUNTER_SHOT
    game.players[3].alive = False
    game.pending_hunter_id = 3
    game.speeches.append(
        SpeechRecord(
            day=2,
            player_id=2,
            player_name="玩家3",
            content="我遗言报清楚：我是预言家，昨晚验到2号是狼人。后面按这个验人去看站边和票型。",
            speech_type="last_words",
        )
    )

    assert _sample_hunter_shot_target(game) == 1


def test_dead_human_hunter_snapshot_keeps_shot_candidates() -> None:
    roles = [RoleName.HUNTER, RoleName.WEREWOLF, RoleName.SEER]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.HUNTER_SHOT
    game.players[0].alive = False
    game.pending_hunter_id = 0
    game.pending_human_action = "hunter_shot"

    snapshot = game.to_snapshot()

    assert snapshot.human_alive is False
    assert snapshot.pending_human_action == "hunter_shot"
    assert snapshot.human_target_candidates == [1, 2]


def test_dead_human_sheriff_snapshot_keeps_badge_candidates() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=1)
    game.phase = Phase.BADGE_TRANSFER
    game.players[1].alive = False
    game.players[1].is_sheriff = True
    game.sheriff_id = 1
    game.current_exile_target_id = 1
    game.pending_human_action = "badge_transfer"

    snapshot = game.to_snapshot()

    assert snapshot.human_alive is False
    assert snapshot.pending_human_action == "badge_transfer"
    assert snapshot.human_target_candidates == [0, 2, 3]


def test_snapshot_rebuilds_pending_action_from_current_phase() -> None:
    roles = [RoleName.HUNTER, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)

    game.pending_human_action = "wolf_chat"
    game.phase = Phase.DAY_VOTE
    snapshot = game.to_snapshot()
    assert snapshot.pending_human_action == "day_vote"

    game.pending_human_action = "night"
    game.phase = Phase.EXILE_PK_VOTE
    game.exile_pk_candidate_ids = [1, 2]
    snapshot = game.to_snapshot()
    assert snapshot.pending_human_action == "day_vote"
    assert snapshot.human_target_candidates == [1, 2]

    game.pending_human_action = "day_vote"
    game.phase = Phase.LAST_WORDS
    game.players[0].alive = False
    game.current_exile_target_id = 0
    snapshot = game.to_snapshot()
    assert snapshot.pending_human_action == "last_words"

    game.pending_human_action = "last_words"
    game.phase = Phase.HUNTER_SHOT
    game.current_exile_target_id = None
    game.pending_hunter_id = 0
    snapshot = game.to_snapshot()
    assert snapshot.pending_human_action == "hunter_shot"
    assert snapshot.human_target_candidates == [1, 2, 3]

    game.pending_human_action = "hunter_shot"
    game.phase = Phase.BADGE_TRANSFER
    game.pending_hunter_id = None
    game.current_exile_target_id = 0
    snapshot = game.to_snapshot()
    assert snapshot.pending_human_action == "badge_transfer"
    assert snapshot.human_target_candidates == [1, 2, 3]


def test_snapshot_private_context_contract_for_each_core_role() -> None:
    roles = [
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

    for human_player_id, role in enumerate(roles):
        game = make_game(roles, human_player_id=human_player_id)
        snapshot = game.to_snapshot()
        private_context = snapshot.human_private_context

        assert f"你是 {human_player_id + 1} 号位" in private_context
        assert f"身份：{role.value}" in private_context
        assert "阵营：" in private_context
        if role == RoleName.WEREWOLF:
            assert snapshot.human_is_wolf
            assert snapshot.wolf_teammate_ids
            assert "狼队友：" in private_context
            assert "本夜可刀目标：" in private_context
            assert all(target_id not in snapshot.wolf_teammate_ids for target_id in snapshot.human_target_candidates)
        else:
            assert not snapshot.human_is_wolf
            assert snapshot.wolf_teammate_ids == []
            assert snapshot.wolf_chat_records == []
            assert snapshot.wolf_history_summaries == []
            assert "狼队友：" not in private_context


def test_create_game_rejects_non_12_player_default_rule() -> None:
    try:
        WerwolfGame.create(11)
    except ValueError:
        return
    raise AssertionError("WerwolfGame.create should reject non-12-player default games")


async def test_api_smoke_pending_human_actions_are_playable_through_routes() -> None:
    snapshot = await api_create_game(CreateGameRequest(player_count=12))
    game_manager.get_game(snapshot.game_id).runtime.enabled = False
    assert snapshot.game_id
    assert snapshot.phase in {Phase.WOLF_CHAT, Phase.NIGHT, Phase.DAY_SPEECH, Phase.LAST_WORDS, Phase.GAME_OVER}
    seen_pending_actions: set[str] = set()

    for step in range(80):
        snapshot = await api_get_game(snapshot.game_id)
        if snapshot.phase == Phase.GAME_OVER:
            break

        pending = snapshot.pending_human_action
        if pending:
            seen_pending_actions.add(pending)
            _assert_snapshot_pending_action_is_actionable(snapshot)

        if pending == "wolf_chat":
            target_id = snapshot.human_target_candidates[0] if snapshot.human_target_candidates else None
            snapshot = await api_resolve_wolf_chat(
                snapshot.game_id,
                NightRequest(
                    action_type="wolf_chat",
                    target_id=target_id,
                    chat_content=f"API烟测第{step}步，先统一这个刀口，别刀狼队友。",
                ),
            )
            continue
        if pending == "night":
            action_type = snapshot.human_allowed_night_actions[0] if snapshot.human_allowed_night_actions else "skip"
            target_id = snapshot.human_target_candidates[0] if action_type != "skip" and snapshot.human_target_candidates else None
            snapshot = await api_resolve_night(
                snapshot.game_id,
                NightRequest(action_type=action_type, target_id=target_id),
            )
            continue
        if pending in {"day_speech", "exile_pk_speech"}:
            snapshot = await api_resolve_speech(
                snapshot.game_id,
                SpeechRequest(content="我这轮先给明确态度，谁前后站边不闭环，投票我会优先看谁。"),
            )
            continue
        if pending == "last_words":
            snapshot = await api_resolve_last_words(
                snapshot.game_id,
                SpeechRequest(content="遗言留一句：回头看谁借我出局补票。"),
            )
            continue
        if pending in {"day_vote", "hunter_shot"}:
            target_id = snapshot.human_target_candidates[0]
            if pending == "hunter_shot":
                snapshot = await api_resolve_hunter_shot(snapshot.game_id, VoteRequest(target_id=target_id))
            else:
                snapshot = await api_resolve_vote(snapshot.game_id, VoteRequest(target_id=target_id))
            continue
        if pending == "badge_transfer":
            target_id = snapshot.human_target_candidates[0] if snapshot.human_target_candidates else None
            snapshot = await api_resolve_badge(
                snapshot.game_id,
                SheriffRequest(badge_target_id=target_id, tear_badge=target_id is None),
            )
            continue
        if pending in {"sheriff_election", "sheriff_speech", "sheriff_vote"}:
            vote_target_id = snapshot.human_target_candidates[0] if snapshot.human_target_candidates else None
            snapshot = await api_resolve_sheriff(
                snapshot.game_id,
                SheriffRequest(
                    run_for_sheriff=False,
                    vote_target_id=vote_target_id,
                    speech="我警上先给一条主线，别让低信息位混过去。",
                ),
            )
            continue

    assert snapshot.phase in {
        Phase.WOLF_CHAT,
        Phase.NIGHT,
        Phase.DAY_SPEECH,
        Phase.DAY_VOTE,
        Phase.LAST_WORDS,
        Phase.HUNTER_SHOT,
        Phase.BADGE_TRANSFER,
        Phase.EXILE_PK_SPEECH,
        Phase.EXILE_PK_VOTE,
        Phase.GAME_OVER,
    }
    assert seen_pending_actions or snapshot.phase != Phase.GAME_OVER
    assert snapshot.human_private_context
    assert "身份：" in snapshot.human_private_context
    assert all(event.visibility != "audit" for event in snapshot.events)


async def test_api_wolf_chat_confirm_requests_are_serialized_per_game() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH]
    game = make_game(roles, human_player_id=0)
    game.runtime.enabled = False
    game.game_id = "test_concurrent_wolf_confirm"
    game_manager._games[game.game_id] = game

    try:
        await asyncio.gather(
            api_resolve_wolf_chat(
                game.game_id,
                NightRequest(action_type="wolf_confirm", target_id=2, chat_content="我确认刀3号，直接拆预言家压力位。"),
            ),
            api_resolve_wolf_chat(
                game.game_id,
                NightRequest(action_type="wolf_confirm", target_id=3, chat_content="我确认刀4号，别再改刀。"),
            ),
        )
    finally:
        game_manager._games.pop(game.game_id, None)

    final_events = [
        event
        for event in game.events
        if event.occurrence_key == f"wolf_chat_final:{game.night_id}"
    ]
    assert len(final_events) == 1
    assert game.phase == Phase.NIGHT
    assert game.wolf_consensus_target_id == 2
    assert "3号" in final_events[0].message
    assert "4号" not in final_events[0].message


async def test_api_smoke_multiple_random_human_roles_do_not_get_stuck() -> None:
    seen_roles: set[RoleName] = set()
    for seed in range(8):
        random.seed(seed)
        snapshot = await api_create_game(CreateGameRequest(player_count=12))
        game_manager.get_game(snapshot.game_id).runtime.enabled = False
        seen_roles.add(snapshot.human_role)

        for step in range(60):
            snapshot = await api_get_game(snapshot.game_id)
            if snapshot.phase == Phase.GAME_OVER:
                break
            pending = snapshot.pending_human_action
            if pending:
                _assert_snapshot_pending_action_is_actionable(snapshot)
            if pending == "wolf_chat":
                snapshot = await api_resolve_wolf_chat(
                    snapshot.game_id,
                    NightRequest(
                        action_type="wolf_chat",
                        target_id=snapshot.human_target_candidates[0],
                        chat_content=f"seed{seed}-step{step} 狼聊给刀口，先看能带队的位置。",
                    ),
                )
                continue
            if pending == "night":
                action_type = snapshot.human_allowed_night_actions[0] if snapshot.human_allowed_night_actions else "skip"
                target_id = snapshot.human_target_candidates[0] if action_type != "skip" and snapshot.human_target_candidates else None
                snapshot = await api_resolve_night(
                    snapshot.game_id,
                    NightRequest(action_type=action_type, target_id=target_id),
                )
                continue
            if pending in {"day_speech", "exile_pk_speech"}:
                snapshot = await api_resolve_speech(
                    snapshot.game_id,
                    SpeechRequest(content="我这轮给明确票意，谁借公共结论补刀我就看谁。"),
                )
                continue
            if pending == "last_words":
                snapshot = await api_resolve_last_words(
                    snapshot.game_id,
                    SpeechRequest(content="遗言留票型，回看补刀最急的位置。"),
                )
                continue
            if pending in {"day_vote", "hunter_shot"}:
                target_id = snapshot.human_target_candidates[0]
                if pending == "hunter_shot":
                    snapshot = await api_resolve_hunter_shot(snapshot.game_id, VoteRequest(target_id=target_id))
                else:
                    snapshot = await api_resolve_vote(snapshot.game_id, VoteRequest(target_id=target_id))
                continue
            if pending == "badge_transfer":
                target_id = snapshot.human_target_candidates[0] if snapshot.human_target_candidates else None
                snapshot = await api_resolve_badge(
                    snapshot.game_id,
                    SheriffRequest(badge_target_id=target_id, tear_badge=target_id is None),
                )
                continue
            if pending in {"sheriff_election", "sheriff_speech", "sheriff_vote"}:
                vote_target_id = snapshot.human_target_candidates[0] if snapshot.human_target_candidates else None
                snapshot = await api_resolve_sheriff(
                    snapshot.game_id,
                    SheriffRequest(
                        run_for_sheriff=False,
                        vote_target_id=vote_target_id,
                        speech="我警上不抢空话，只看谁发言能闭环。",
                    ),
                )
                continue

        assert snapshot.phase in {
            Phase.WOLF_CHAT,
            Phase.NIGHT,
            Phase.DAY_SPEECH,
            Phase.DAY_VOTE,
            Phase.LAST_WORDS,
            Phase.HUNTER_SHOT,
            Phase.BADGE_TRANSFER,
            Phase.EXILE_PK_SPEECH,
            Phase.EXILE_PK_VOTE,
            Phase.GAME_OVER,
        }
        assert snapshot.human_private_context
        assert all(event.visibility != "audit" for event in snapshot.events)

    assert len(seen_roles) >= 3


def _assert_snapshot_pending_action_is_actionable(snapshot) -> None:
    pending = snapshot.pending_human_action
    if pending == "wolf_chat":
        assert snapshot.phase == Phase.WOLF_CHAT
        assert snapshot.human_is_wolf
        assert snapshot.current_speaker_id == snapshot.human_player_id
        assert snapshot.human_target_candidates
        assert all(target not in snapshot.wolf_teammate_ids for target in snapshot.human_target_candidates)
        return
    if pending == "night":
        assert snapshot.phase == Phase.NIGHT
        assert snapshot.human_allowed_night_actions
        if snapshot.human_allowed_night_actions != ["skip"]:
            assert snapshot.human_target_candidates
        return
    if pending in {"day_speech", "exile_pk_speech"}:
        assert snapshot.current_speaker_id == snapshot.human_player_id
        return
    if pending == "last_words":
        assert snapshot.phase == Phase.LAST_WORDS
        return
    if pending == "day_vote":
        assert snapshot.phase in {Phase.DAY_VOTE, Phase.EXILE_PK_VOTE}
        assert snapshot.human_target_candidates
        return
    if pending == "hunter_shot":
        assert snapshot.phase == Phase.HUNTER_SHOT
        assert snapshot.human_target_candidates
        return
    if pending == "badge_transfer":
        assert snapshot.phase == Phase.BADGE_TRANSFER
        return
    if pending in {"sheriff_election", "sheriff_speech", "sheriff_vote", "choose_speech_order"}:
        return
    raise AssertionError(f"未覆盖的 pending_human_action: {pending}")


async def test_snapshot_visible_timeline_keeps_only_current_wolf_chat_night() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.IDIOT, RoleName.VILLAGER]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime(
        {0: 2, 1: 2},
        strict=True,
        contents={"wolf_chat": "今晚先刀3号，拆信息位，明天别聊漏队友。"},
    )
    game.rule_profile.wolf_chat_rounds = 1

    await game.resolve_wolf_chat(HumanNightAction(action_type="wolf_confirm", target_id=2, chat_content="我确认刀3号，先拆预言家空间。"))
    assert game.phase == Phase.NIGHT
    first_key = "wolf_chat_final:1"
    assert any(event.occurrence_key == first_key for event in game.events)

    game.day = 2
    game.night_id = 2
    game.phase = Phase.WOLF_CHAT
    game.wolf_consensus_target_id = None
    game.wolf_night_plan = None
    game.speech_order = []
    game.speech_cursor = 0
    game.wolf_chat_prepared_night_id = None
    game._prepare_wolf_chat_order()

    snapshot = game.to_snapshot()
    event_keys = [event.occurrence_key for event in snapshot.events if event.phase == "wolf_chat"]
    timeline_keys = [item.occurrence_key for item in snapshot.visible_timeline if item.phase == "wolf_chat"]
    assert first_key not in event_keys
    assert first_key not in timeline_keys
    assert all(item.night_id == game.night_id for item in snapshot.visible_timeline if item.phase == "wolf_chat")


async def test_wolf_chat_timeline_orders_current_night_events_and_messages_once() -> None:
    roles = [RoleName.WEREWOLF, RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)
    game.runtime = FakeRuntime(
        {1: 2},
        strict=True,
        contents={"wolf_chat": "我建议刀3号，先拆信息位，白天别一起补同一条线。"},
    )
    game.rule_profile.wolf_chat_rounds = 1

    await game.resolve_wolf_chat(HumanNightAction(action_type="wolf_chat", target_id=2, chat_content="我先给3号，验人位风险最大。"))
    in_chat_snapshot = game.to_snapshot()
    in_chat_items = [item for item in in_chat_snapshot.visible_timeline if item.phase == "wolf_chat"]
    assert [item.kind for item in in_chat_items][:2] == ["event", "wolf_chat"]

    await game.resolve_wolf_chat(None)
    assert game.phase == Phase.NIGHT

    game.phase = Phase.WOLF_CHAT
    final_snapshot = game.to_snapshot()
    final_items = [item for item in final_snapshot.visible_timeline if item.phase == "wolf_chat"]
    final_keys = [item.occurrence_key for item in final_items if item.occurrence_key == "wolf_chat_final:1"]
    assert len(final_keys) == 1
    assert final_items[-1].occurrence_key == "wolf_chat_final:1"


async def test_visible_timeline_does_not_duplicate_public_speeches_or_votes() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=0)
    game.phase = Phase.DAY_SPEECH
    game.speech_order = [0]
    game.speech_cursor = 0

    await game.resolve_day_speeches("我今天先点2号，刚才站边太快，后面如果不给具体理由我会投他。")
    speech_items = [
        item
        for item in game.to_snapshot().visible_timeline
        if item.speaker_id == 0 and item.action == "day_speech"
    ]
    assert len(speech_items) == 1
    assert speech_items[0].kind == "message"

    game.runtime = FakeRuntime({1: 0, 2: 0, 3: 0}, strict=True)
    await game.resolve_votes(1)

    vote_items = [
        item
        for item in game.to_snapshot().visible_timeline
        if item.message_type == "vote" and item.action == "day_vote"
    ]
    assert len(vote_items) == len(game.votes)
    assert all(item.kind == "message" for item in vote_items)


def test_agent_visible_context_event_cursor_tracks_only_new_events() -> None:
    roles = [RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER]
    game = make_game(roles, human_player_id=3)
    player = game.players[0]
    game.phase = Phase.DAY_SPEECH

    game._add_event("speech", "初始公开事件。")
    first_context = game._build_agent_context(player, "day_speech", [], "test", "speak")
    second_context = game._build_agent_context(player, "day_speech", [], "test", "speak")
    game._add_event("speech", "新增公开事件。")
    third_context = game._build_agent_context(player, "day_speech", [], "test", "speak")

    assert first_context.structured is not None
    assert second_context.structured is not None
    assert third_context.structured is not None
    assert [event.message for event in first_context.structured.new_visible_events] == ["初始公开事件。"]
    assert second_context.structured.new_visible_events == []
    assert [event.message for event in third_context.structured.new_visible_events] == ["新增公开事件。"]


def test_model_decision_with_ai_leak_falls_back_to_player_speech() -> None:
    game = make_game([RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER], human_player_id=4)
    runtime = OpenAIAgentRuntime()
    context = game._build_agent_context(game.players[0], "day_speech", [], game._day_speech_goal(game.players[0]), "speak")
    decision = AgentDecision(
        action="speak",
        content="我现在先接入2号的上下文，根据提示词和候选列表来说。",
    )

    finalized = runtime._finalize_model_decision(decision, context)

    assert "接入" not in finalized.content
    assert "上下文" not in finalized.content
    assert "提示词" not in finalized.content
    assert "候选列表" not in finalized.content
    assert finalized.reason == "模型输出含出戏表达或内部字段，已切换为本地真人化兜底。"


def test_model_decision_with_chain_commentary_opening_falls_back() -> None:
    game = make_game([RoleName.WEREWOLF, RoleName.SEER, RoleName.WITCH, RoleName.HUNTER, RoleName.VILLAGER], human_player_id=4)
    runtime = OpenAIAgentRuntime()
    context = game._build_agent_context(game.players[0], "day_speech", [], game._day_speech_goal(game.players[0]), "speak")
    decision = AgentDecision(
        action="speak",
        content="我先抓1号这句，他把话说得太满了，后面我继续盯这个位置。",
    )

    finalized = runtime._finalize_model_decision(decision, context)

    assert "我先抓1号这句" not in finalized.content
    assert finalized.reason == "模型输出含出戏表达或内部字段，已切换为本地真人化兜底。"
