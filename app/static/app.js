const state = {
  gameId: null,
  snapshot: null,
  countdownTimer: null,
  pollTimer: null,
  renderedSpeechKeys: new Set(),
  deferredSnapshot: null,
  lastSnapshotSeq: -1,
  boardScrollLocks: new WeakMap(),
  optionSignatures: {
    target: "",
    wolfTarget: "",
    vote: "",
    sheriffVote: "",
    badgeTarget: "",
    nightAction: "",
  },
  lockedActions: {
    wolf_chat: false,
    sheriff: false,
    speech: false,
    last_words: false,
    night: false,
    vote: false,
    badge: false,
    direction: false,
  },
};

const playerCount = document.getElementById("playerCount");
const createGameBtn = document.getElementById("createGameBtn");
const refreshBtn = document.getElementById("refreshBtn");
const speechBtn = document.getElementById("speechBtn");
const nightBtn = document.getElementById("nightBtn");
const voteBtn = document.getElementById("voteBtn");
const wolfChatBtn = document.getElementById("wolfChatBtn");
const wolfConfirmBtn = document.getElementById("wolfConfirmBtn");
const sheriffBtn = document.getElementById("sheriffBtn");
const lastWordsBtn = document.getElementById("lastWordsBtn");
const badgeTransferBtn = document.getElementById("badgeTransferBtn");
const badgeTearBtn = document.getElementById("badgeTearBtn");
const selfDestructBtn = document.getElementById("selfDestructBtn");
const directionLeftBtn = document.getElementById("directionLeftBtn");
const directionRightBtn = document.getElementById("directionRightBtn");

const speechInput = document.getElementById("speechInput");
const sheriffSpeechInput = document.getElementById("sheriffSpeechInput");
const lastWordsInput = document.getElementById("lastWordsInput");
const wolfChatInput = document.getElementById("wolfChatInput");

const targetSelect = document.getElementById("targetSelect");
const wolfTargetSelect = document.getElementById("wolfTargetSelect");
const voteSelect = document.getElementById("voteSelect");
const sheriffVoteSelect = document.getElementById("sheriffVoteSelect");
const badgeTargetSelect = document.getElementById("badgeTargetSelect");
const nightActionType = document.getElementById("nightActionType");
const currentHint = document.getElementById("currentHint");
const pendingActionText = document.getElementById("pendingActionText");
const privateContextCard = document.getElementById("privateContextCard");
const timerBadge = document.getElementById("timerBadge");
const humanRole = document.getElementById("humanRole");
const wolfChatCard = document.getElementById("wolfChatCard");
const rightPanel = document.querySelector(".right-panel");
const eventsBoard = document.getElementById("eventsBoard");
const historyBoard = document.getElementById("historyBoard");

const speechSection = document.getElementById("speechSection");
const wolfChatSection = document.getElementById("wolfChatSection");
const sheriffSection = document.getElementById("sheriffSection");
const directionSection = document.getElementById("directionSection");
const lastWordsSection = document.getElementById("lastWordsSection");
const nightSection = document.getElementById("nightSection");
const voteSection = document.getElementById("voteSection");
const badgeSection = document.getElementById("badgeSection");
const selfDestructSection = document.getElementById("selfDestructSection");
const spectatorSection = document.getElementById("spectatorSection");
const wolfChatBoard = document.getElementById("wolfChatBoard");
const speechFeedBoard = document.getElementById("speechFeedBoard");
const runForSheriffCheckbox = document.getElementById("runForSheriffCheckbox");

function resolveThemeClass(phase) {
  const nightPhases = new Set(["wolf_chat", "night"]);
  return nightPhases.has(phase) ? "theme-night" : "theme-day";
}

function scrollBoardToBottom(board) {
  board.scrollTop = board.scrollHeight;
}

function isNearBottom(board, threshold = 28) {
  return board.scrollHeight - board.clientHeight - board.scrollTop <= threshold;
}

function shouldStickToBottom(board) {
  if (!state.boardScrollLocks.has(board)) {
    state.boardScrollLocks.set(board, true);
  }
  return state.boardScrollLocks.get(board);
}

function updateBoardScrollLock(board) {
  state.boardScrollLocks.set(board, isNearBottom(board));
}

function maybeStickBoardToBottom(board, force = false) {
  if (force || shouldStickToBottom(board)) {
    scrollBoardToBottom(board);
    state.boardScrollLocks.set(board, true);
  }
}

function activeActionField() {
  const active = document.activeElement;
  if (!active) {
    return null;
  }
  if (!(active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement || active instanceof HTMLSelectElement)) {
    return null;
  }
  if (!active.closest(".action-panel")) {
    return null;
  }
  return active;
}

function isUserEditingActionPanel() {
  return Boolean(activeActionField());
}

function flushDeferredSnapshot() {
  if (!state.deferredSnapshot) {
    return;
  }
  const nextSnapshot = state.deferredSnapshot;
  state.deferredSnapshot = null;
  applySnapshot(nextSnapshot);
}

function buildSelectSignature(items, formatter) {
  return items.map(formatter).join("|");
}

function setSelectOptionsIfChanged(select, signatureKey, items, formatter, includeEmptyLabel = null) {
  const optionSignature = `${includeEmptyLabel ?? ""}::${buildSelectSignature(items, formatter)}`;
  if (state.optionSignatures[signatureKey] === optionSignature) {
    return;
  }
  state.optionSignatures[signatureKey] = optionSignature;
  const previousValue = select.value;
  select.innerHTML = "";
  if (includeEmptyLabel !== null) {
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = includeEmptyLabel;
    select.appendChild(emptyOption);
  }
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = formatter(item).value;
    option.textContent = formatter(item).label;
    select.appendChild(option);
  });
  if ([...select.options].some((option) => option.value === previousValue)) {
    select.value = previousValue;
  }
}

function captureInputState() {
  const active = document.activeElement;
  if (!active || !(active instanceof HTMLTextAreaElement || active instanceof HTMLInputElement)) {
    return null;
  }
  return {
    id: active.id,
    selectionStart: typeof active.selectionStart === "number" ? active.selectionStart : null,
    selectionEnd: typeof active.selectionEnd === "number" ? active.selectionEnd : null,
    scrollTop: active.scrollTop,
  };
}

function restoreInputState(inputState) {
  if (!inputState?.id) {
    return;
  }
  const element = document.getElementById(inputState.id);
  if (!element || element.disabled || element.classList.contains("hidden")) {
    return;
  }
  if (!(element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement)) {
    return;
  }
  element.focus();
  if (typeof inputState.selectionStart === "number" && typeof inputState.selectionEnd === "number") {
    element.setSelectionRange(inputState.selectionStart, inputState.selectionEnd);
  }
  element.scrollTop = inputState.scrollTop || 0;
}

async function requestJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : undefined,
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = payload?.detail ? `：${payload.detail}` : "";
    } catch (_) {
      detail = "";
    }
    throw new Error(`请求失败 ${response.status}${detail}`);
  }
  return response.json();
}

function showActionError(error) {
  console.error("操作失败", error);
  const privateMessage = document.getElementById("privateMessage");
  if (!privateMessage) {
    return;
  }
  privateContextCard?.classList.remove("hidden");
  privateMessage.classList.remove("hidden");
  privateMessage.classList.add("action-error");
  privateMessage.textContent = error instanceof Error ? error.message : "操作失败，请刷新后重试。";
}

function clearActionError() {
  const privateMessage = document.getElementById("privateMessage");
  privateMessage?.classList.remove("action-error");
}

async function runLockedAction(lockKey, button, action) {
  if (button?.disabled) return;
  if (lockKey) {
    state.lockedActions[lockKey] = true;
  }
  if (button) {
    button.disabled = true;
  }
  clearActionError();
  try {
    const snapshot = await action();
    applySnapshot(snapshot);
  } catch (error) {
    if (lockKey) {
      state.lockedActions[lockKey] = false;
    }
    if (button) {
      button.disabled = false;
    }
    showActionError(error);
    syncActionPanel(state.snapshot);
  }
}

createGameBtn.addEventListener("click", async () => {
  await runLockedAction(null, createGameBtn, async () => requestJson("/api/games", {
    player_count: Number(playerCount.value),
  }));
  createGameBtn.disabled = false;
});

refreshBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  const response = await fetch(`/api/games/${state.gameId}`);
  const snapshot = await response.json();
  applySnapshot(snapshot);
});

speechBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("speech", speechBtn, async () => {
    const snapshot = await requestJson(`/api/games/${state.gameId}/speech`, {
      content: speechInput.value,
    });
    speechInput.value = "";
    return snapshot;
  });
});

nightBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("night", nightBtn, async () => requestJson(`/api/games/${state.gameId}/night`, {
      action_type: nightActionType.value,
      target_id: targetSelect.value === "" ? null : Number(targetSelect.value),
    }));
});

voteBtn.addEventListener("click", async () => {
  if (!state.gameId || voteSelect.value === "") return;
  const endpoint = state.snapshot?.phase === "hunter_shot" ? "hunter-shot" : "vote";
  await runLockedAction("vote", voteBtn, async () => requestJson(`/api/games/${state.gameId}/${endpoint}`, {
      target_id: Number(voteSelect.value),
    }));
});

wolfChatBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("wolf_chat", wolfChatBtn, async () => {
    const snapshot = await requestJson(`/api/games/${state.gameId}/wolf-chat`, {
      action_type: "wolf_chat",
      target_id: wolfTargetSelect.value === "" ? null : Number(wolfTargetSelect.value),
      chat_content: wolfChatInput.value,
    });
    wolfChatInput.value = "";
    return snapshot;
  });
});

wolfConfirmBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (wolfTargetSelect.value === "") {
    showActionError(new Error("请先选择一个合法刀口，再确认最终目标。"));
    return;
  }
  await runLockedAction("wolf_chat", wolfConfirmBtn, async () => {
    const snapshot = await requestJson(`/api/games/${state.gameId}/wolf-chat`, {
      action_type: "wolf_confirm",
      target_id: wolfTargetSelect.value === "" ? null : Number(wolfTargetSelect.value),
      chat_content: wolfChatInput.value || "我确认这个最终刀口，今晚统一执行。",
    });
    wolfChatInput.value = "";
    return snapshot;
  });
});

sheriffBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("sheriff", sheriffBtn, async () => {
    const snapshot = await requestJson(`/api/games/${state.gameId}/sheriff`, {
      run_for_sheriff: runForSheriffCheckbox.checked,
      vote_target_id: sheriffVoteSelect.value === "" ? null : Number(sheriffVoteSelect.value),
      speech: sheriffSpeechInput.value,
    });
    sheriffSpeechInput.value = "";
    return snapshot;
  });
});

lastWordsBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("last_words", lastWordsBtn, async () => {
    const snapshot = await requestJson(`/api/games/${state.gameId}/last-words`, {
      content: lastWordsInput.value,
    });
    lastWordsInput.value = "";
    return snapshot;
  });
});

badgeTransferBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("badge", badgeTransferBtn, async () => requestJson(`/api/games/${state.gameId}/badge`, {
      badge_target_id: badgeTargetSelect.value === "" ? null : Number(badgeTargetSelect.value),
      tear_badge: false,
    }));
});

badgeTearBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("badge", badgeTearBtn, async () => requestJson(`/api/games/${state.gameId}/badge`, {
      tear_badge: true,
    }));
});

selfDestructBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction(null, selfDestructBtn, async () => requestJson(`/api/games/${state.gameId}/self-destruct`));
});

directionLeftBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("direction", directionLeftBtn, async () => requestJson(`/api/games/${state.gameId}/speech-order`, {
      speech_order_direction: "left",
    }));
});

directionRightBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  await runLockedAction("direction", directionRightBtn, async () => requestJson(`/api/games/${state.gameId}/speech-order`, {
      speech_order_direction: "right",
    }));
});

function applySnapshot(snapshot) {
  const nextSeq = Number(snapshot.snapshot_seq ?? 0);
  const currentSeq = Number(state.lastSnapshotSeq ?? -1);
  if (state.gameId === snapshot.game_id && nextSeq < currentSeq) {
    return;
  }
  state.lastSnapshotSeq = Math.max(currentSeq, nextSeq);
  const inputState = captureInputState();
  const previousSnapshot = state.snapshot;
  state.gameId = snapshot.game_id;
  state.snapshot = snapshot;
  ensurePolling();

  const gameTitle = document.getElementById("gameTitle");
  const gameMeta = document.getElementById("gameMeta");
  const phaseBadge = document.getElementById("phaseBadge");
  const privateMessage = document.getElementById("privateMessage");
  const winnerBadge = document.getElementById("winnerBadge");

  if (gameTitle) {
    gameTitle.textContent = `对局 ${snapshot.game_id}`;
  }
  if (gameMeta) {
    gameMeta.textContent =
      `第 ${snapshot.day} 天 / 当前阶段：${formatPhase(snapshot.phase)} / 你的座位：${snapshot.human_player_id + 1}`;
  }
  if (phaseBadge) {
    phaseBadge.textContent = formatPhase(snapshot.phase);
  }
  document.body.classList.remove("theme-day", "theme-night");
  document.body.classList.add(resolveThemeClass(snapshot.phase));
  humanRole.classList.remove("hidden");
  humanRole.textContent = `你的身份：${snapshot.human_role}`;
  if (pendingActionText) {
    pendingActionText.textContent =
      snapshot.pending_human_action ? `当前需要执行：${formatPendingAction(snapshot.pending_human_action)}` : "当前无强制操作标记";
  }
  if (currentHint) {
    currentHint.textContent = snapshot.current_hint || "等待下一步。";
  }

  const privateText = snapshot.human_private_context || snapshot.human_private_message;
  if (privateMessage && privateText) {
    privateContextCard?.classList.remove("hidden");
    privateMessage.classList.remove("hidden");
    privateMessage.classList.remove("action-error");
    privateMessage.textContent = privateText;
  } else if (privateMessage) {
    privateContextCard?.classList.add("hidden");
    privateMessage.classList.add("hidden");
    privateMessage.textContent = "";
  }

  if (winnerBadge && snapshot.winner) {
    winnerBadge.classList.remove("hidden");
    winnerBadge.textContent = `胜者：${snapshot.winner}`;
  } else if (winnerBadge) {
    winnerBadge.classList.add("hidden");
  }

  syncTimer(snapshot);

  wolfChatCard.classList.toggle("hidden", !snapshot.human_is_wolf);
  rightPanel.classList.toggle("has-wolf-chat", snapshot.human_is_wolf);
  renderPlayers(snapshot.players, snapshot);
  renderWolfChats(snapshot);
  renderSpeechFeed(snapshot.speeches, snapshot);
  renderEvents(snapshot.visible_timeline, snapshot);
  renderHistory(snapshot.votes, snapshot.night_summaries, snapshot.players);
  rebuildTargetOptions(snapshot);
  syncActionLocks(snapshot, previousSnapshot);
  syncActionPanel(snapshot);
  restoreInputState(inputState);
}

if (!state.gameId) {
  humanRole.classList.add("hidden");
  wolfChatCard.classList.add("hidden");
  rightPanel.classList.remove("has-wolf-chat");
  document.body.classList.add("theme-day");
}

function ensurePolling() {
  if (state.pollTimer || !state.gameId) {
    return;
  }
  state.pollTimer = setInterval(async () => {
    if (!state.gameId) {
      return;
    }
    try {
      const response = await fetch(`/api/games/${state.gameId}`);
      const next = await response.json();
      if (isUserEditingActionPanel()) {
        state.deferredSnapshot = next;
        return;
      }
      applySnapshot(next);
    } catch (error) {
      console.error("轮询对局失败", error);
    }
  }, 2000);
}

function syncTimer(snapshot) {
  if (state.countdownTimer) {
    clearInterval(state.countdownTimer);
    state.countdownTimer = null;
  }

  timerBadge.classList.remove("timer-warn", "timer-danger");

  if (!snapshot.time_limit_seconds || !snapshot.deadline_ts) {
    timerBadge.classList.add("hidden");
    timerBadge.textContent = "";
    return;
  }

  timerBadge.classList.remove("hidden");

  const updateView = () => {
    const rawRemaining = snapshot.deadline_ts - Date.now() / 1000;
    const remaining = Math.max(0, Math.ceil(rawRemaining));
    timerBadge.textContent = remaining > 0
      ? `${snapshot.timer_label || "剩余时间"} ${remaining} 秒`
      : `${snapshot.timer_label || "剩余时间"} 处理中...`;
    timerBadge.classList.remove("timer-warn", "timer-danger");
    if (remaining <= 8 && remaining > 0) {
      timerBadge.classList.add("timer-danger");
    } else if (remaining <= 20 && remaining > 0) {
      timerBadge.classList.add("timer-warn");
    }
  };

  updateView();
  state.countdownTimer = setInterval(updateView, 1000);
}

function renderPlayers(players, snapshot) {
  const board = document.getElementById("playersBoard");
  board.innerHTML = "";
  const revealActiveSpeaker = !["wolf_chat", "night"].includes(snapshot.phase);
  players.forEach((player) => {
    const row = document.createElement("div");
    const isCurrentSpeaker = revealActiveSpeaker && snapshot.current_speaker_id === player.id;
    const isWolfMate = (snapshot.wolf_teammate_ids || []).includes(player.id);
    const shouldRevealRole = snapshot.phase === "game_over" || player.is_human;
    const roleText = shouldRevealRole ? `身份：${player.role}` : (isWolfMate ? "身份：狼队友" : "身份：未知");
    row.className = `player-row ${player.alive ? "alive" : "dead"} ${isCurrentSpeaker ? "active-speaker" : ""}`.trim();
    row.innerHTML = `
      <div class="player-topline">
        <strong>${player.name}${player.is_sheriff ? " · 警长" : ""}</strong>
        ${player.is_human ? `<span class="player-tag self-tag">你</span>` : ""}
        ${isWolfMate ? `<span class="player-tag wolf-tag">队友</span>` : ""}
        ${isCurrentSpeaker ? `<span class="player-tag">当前发言</span>` : ""}
      </div>
      <div class="player-meta">
        <span class="seat-badge">${player.id + 1}号位</span>
        <span class="camp-badge ${player.alive ? "alive" : "dead"}">${player.alive ? "存活" : "死亡"}</span>
      </div>
      <div class="player-roleline">${roleText}</div>
    `;
    board.appendChild(row);
  });
}

function renderWolfChats(snapshot) {
  wolfChatBoard.innerHTML = "";
  if (!snapshot.human_is_wolf) {
    wolfChatBoard.innerHTML = `<div class="empty-state">你不是狼人，无法查看狼队夜聊。</div>`;
    return;
  }
  const historySummaries = Array.isArray(snapshot.wolf_history_summaries)
    ? snapshot.wolf_history_summaries.filter(Boolean)
    : [];
  if (historySummaries.length) {
    const historyRow = document.createElement("div");
    historyRow.className = "wolf-chat-card wolf-history-card";
    historyRow.innerHTML = `
      <div class="speech-meta">
        <strong>狼队历史摘要</strong>
        <span>仅狼人可见</span>
      </div>
      <div class="wolf-history-list">
        ${historySummaries.slice(-3).map((summary) => `<div class="wolf-history-item">${summary}</div>`).join("")}
      </div>
    `;
    wolfChatBoard.appendChild(historyRow);
  }
  const currentNightRecords = Array.isArray(snapshot.wolf_chat_records)
    ? snapshot.wolf_chat_records.filter((record) => record.night_id === snapshot.night_id)
    : [];
  if (!currentNightRecords.length) {
    const emptyRow = document.createElement("div");
    emptyRow.className = "empty-state";
    emptyRow.textContent = "本轮暂无狼人夜聊。";
    wolfChatBoard.appendChild(emptyRow);
    return;
  }
  currentNightRecords.forEach((record) => {
    const row = document.createElement("div");
    const isFinal = snapshot.wolf_night_plan?.locked &&
      snapshot.wolf_night_plan?.final_confirmer_id === record.player_id &&
      snapshot.wolf_night_plan?.current_target_id === record.proposed_target_id;
    row.className = `wolf-chat-card ${isFinal ? "wolf-chat-final" : ""}`.trim();
    row.innerHTML = `
      <div class="speech-meta">
        <strong>${record.player_name}</strong>
        <span class="seat-badge">${record.speaker_seat_no || record.player_id + 1}号位</span>
        <span>第 ${record.round_id || 1} 轮</span>
        ${isFinal ? `<span class="player-tag">最终确认</span>` : ""}
      </div>
      <div class="wolf-chat-content">${record.content}</div>
      <div class="summary-meta">
        ${isFinal ? "最终目标" : "建议目标"}：${formatTarget(record.proposed_target_id, snapshot.players)}
        ${record.stance_to_previous ? ` / ${formatWolfStance(record.stance_to_previous)}` : ""}
      </div>
    `;
    wolfChatBoard.appendChild(row);
  });
  if (
    snapshot.wolf_night_plan?.current_target_id !== null &&
    snapshot.wolf_night_plan?.current_target_id !== undefined &&
    !snapshot.wolf_night_plan.locked
  ) {
    const planRow = document.createElement("div");
    planRow.className = "wolf-chat-card wolf-plan-card";
    planRow.innerHTML = `
      <div class="speech-meta"><strong>当前狼队计划</strong></div>
      <div class="wolf-chat-content">
        ${formatTarget(snapshot.wolf_night_plan.current_target_id, snapshot.players)}
        （未最终确认）
      </div>
    `;
    wolfChatBoard.appendChild(planRow);
  }
  maybeStickBoardToBottom(wolfChatBoard);
}

function renderSpeechFeed(speeches, snapshot) {
  speechFeedBoard.innerHTML = "";
  const uniqueSpeeches = speeches.filter((speech, index, source) => {
    const speechKey = `${speech.day}-${speech.player_id}-${speech.speech_type}-${speech.content}`;
    return index === source.findIndex((item) =>
      `${item.day}-${item.player_id}-${item.speech_type}-${item.content}` === speechKey
    );
  });

  if (!uniqueSpeeches.length) {
    speechFeedBoard.innerHTML = `<div class="empty-state">这里会逐条显示所有玩家的发言与遗言。</div>`;
    state.renderedSpeechKeys.clear();
    return;
  }

  uniqueSpeeches.forEach((speech) => {
    const row = document.createElement("div");
    const isSelf = speech.player_id === snapshot.human_player_id;
    const isActive = speech.player_id === snapshot.current_speaker_id;
    const speechKey = `${speech.day}-${speech.player_id}-${speech.speech_type}-${speech.content}`;
    row.className = `chat-bubble speech-bubble ${isSelf ? "me" : ""} ${isActive ? "active-speaker" : ""}`.trim();
    if (!state.renderedSpeechKeys.has(speechKey)) {
      row.classList.add("incoming");
      state.renderedSpeechKeys.add(speechKey);
    }
    row.innerHTML = `
      <div class="speech-meta">
        <strong>${speech.player_name}</strong>
        <span class="seat-badge">${speech.player_id + 1}号位</span>
        <span>${formatSpeechType(speech.speech_type)}</span>
        ${isSelf ? `<span class="player-tag self-tag">你</span>` : ""}
      </div>
      <div class="speech-content">${speech.content}</div>
    `;
    speechFeedBoard.appendChild(row);
  });

  const activeSpeaker = snapshot.players.find((player) => player.id === snapshot.current_speaker_id);
  if (
    activeSpeaker &&
    !activeSpeaker.is_human &&
    ["day_speech", "sheriff_speech", "sheriff_pk_speech", "exile_pk_speech", "last_words"].includes(snapshot.phase)
  ) {
    const thinkingRow = document.createElement("div");
    thinkingRow.className = "chat-bubble speech-bubble thinking-bubble";
    thinkingRow.innerHTML = `
      <div class="speech-meta">
        <strong>${activeSpeaker.name}</strong>
        <span class="seat-badge">${activeSpeaker.id + 1}号位</span>
        <span>正在组织发言</span>
      </div>
      <div class="speech-content">
        <span class="thinking-text">思考中</span>
        <span class="thinking-dots"><span>.</span><span>.</span><span>.</span></span>
      </div>
    `;
    speechFeedBoard.appendChild(thinkingRow);
  }

  maybeStickBoardToBottom(speechFeedBoard);
}

function renderEvents(timeline, snapshot = null) {
  eventsBoard.innerHTML = "";
  const sourceEvents = Array.isArray(timeline)
    ? timeline
        .filter((item) => item.kind === "event")
        .map((item) => ({
          phase: item.phase,
          visibility: item.visibility,
          day: item.day,
          night_id: item.night_id,
          message: item.content,
          occurrence_key: item.occurrence_key || item.item_id,
        }))
    : [];
  const scopedEvents = sourceEvents.filter((event) => {
    if (event.visibility === "audit") return false;
    if (event.phase === "wolf_chat" && snapshot?.night_id !== undefined && event.night_id !== snapshot.night_id) return false;
    return true;
  });
  const uniqueEvents = scopedEvents.filter((event, index, source) =>
    index === source.findIndex((item) => {
      const leftKey = item.occurrence_key || `${item.phase}:${item.visibility}:${item.day}:${item.night_id}:${item.message}`;
      const rightKey = event.occurrence_key || `${event.phase}:${event.visibility}:${event.day}:${event.night_id}:${event.message}`;
      return leftKey === rightKey;
    })
  );
  if (!uniqueEvents.length) {
    eventsBoard.innerHTML = `<div class="empty-state">这里会显示系统播报。</div>`;
    return;
  }
  uniqueEvents.forEach((event) => {
    const row = document.createElement("div");
    row.className = "chat-bubble system-bubble";
    row.innerHTML = `
      <div class="summary-meta">
        <strong>${formatPhase(event.phase)}</strong>
        <span>系统播报</span>
      </div>
      <div class="summary-content">${event.message}</div>
    `;
    eventsBoard.appendChild(row);
  });
  maybeStickBoardToBottom(eventsBoard);
}

function renderHistory(votes, nightSummaries, players) {
  historyBoard.innerHTML = "";
  const groupedVoteMap = new Map();
  votes.forEach((vote) => {
    const voteRound = vote.vote_round || vote.vote_type;
    const key = `${voteRound}-${vote.vote_type}-${vote.day}-${vote.target_id}`;
    if (!groupedVoteMap.has(key)) {
      groupedVoteMap.set(key, {
        vote_type: vote.vote_type,
        vote_round: voteRound,
        day: vote.day,
        target_id: vote.target_id,
        target_name: vote.target_name,
        voter_names: [],
        voter_ids: new Set(),
      });
    }
    const groupedVote = groupedVoteMap.get(key);
    if (!groupedVote.voter_ids.has(vote.voter_id)) {
      groupedVote.voter_ids.add(vote.voter_id);
      groupedVote.voter_names.push(vote.voter_name);
    }
  });
  const voteItems = [...groupedVoteMap.values()].map((vote, index) => ({
    kind: "vote",
    order: nightSummaries.length + index,
    payload: vote,
  }));
  const summaryItems = nightSummaries
    .filter((summary, index, source) =>
      index === source.findIndex((item) =>
        item.day === summary.day &&
        item.wolf_target_id === summary.wolf_target_id &&
        item.seer_target_id === summary.seer_target_id &&
        item.witch_saved === summary.witch_saved &&
        item.witch_poison_target_id === summary.witch_poison_target_id &&
        JSON.stringify(item.deaths) === JSON.stringify(summary.deaths)
      )
    )
    .map((summary, index) => ({
      kind: "summary",
      order: index,
      payload: summary,
    }));
  const merged = [
    ...summaryItems,
    ...voteItems,
  ];

  if (!merged.length) {
    historyBoard.innerHTML = `<div class="empty-state">这里会显示投票与夜间摘要。</div>`;
    return;
  }

  merged.forEach((item) => {
    const row = document.createElement("div");
    if (item.kind === "summary") {
      const summary = item.payload;
      const isRevealMode = summary.wolf_target_id !== null || summary.seer_target_id !== null || summary.witch_poison_target_id !== null || summary.witch_saved;
      row.className = "chat-bubble summary-bubble summary-sheet";
      if (isRevealMode) {
        row.innerHTML = `
          <div class="summary-meta">
            <strong>第 ${summary.day} 夜复盘</strong>
            <span>终局公开信息</span>
          </div>
          <div class="summary-grid">
            <div class="summary-row">
              <span class="summary-label">狼人目标</span>
              <span class="summary-value">${formatTarget(summary.wolf_target_id, players)}</span>
            </div>
            <div class="summary-row">
              <span class="summary-label">预言家查验</span>
              <span class="summary-value">${formatTarget(summary.seer_target_id, players)}${summary.seer_result ? `（${summary.seer_result}）` : ""}</span>
            </div>
            <div class="summary-row">
              <span class="summary-label">女巫救人</span>
              <span class="summary-value">${summary.witch_saved ? "是" : "否"}</span>
            </div>
            <div class="summary-row">
              <span class="summary-label">女巫毒人</span>
              <span class="summary-value">${formatTarget(summary.witch_poison_target_id, players)}</span>
            </div>
            <div class="summary-row">
              <span class="summary-label">死亡结果</span>
              <span class="summary-value">${summary.deaths.length ? summary.deaths.map((id) => playerName(id, players)).join("、") : "无"}</span>
            </div>
          </div>
        `;
      } else {
        row.innerHTML = `
          <div class="summary-meta">
            <strong>第 ${summary.day} 夜公开摘要</strong>
            <span>进行中可见信息</span>
          </div>
          <div class="summary-grid">
            <div class="summary-row">
              <span class="summary-label">公开死讯</span>
              <span class="summary-value">${summary.deaths.length ? summary.deaths.map((id) => playerName(id, players)).join("、") : "平安夜 / 尚未公布"}</span>
            </div>
          </div>
        `;
      }
    } else {
      const vote = item.payload;
      const voteTitle = vote.vote_type === "sheriff"
        ? (vote.vote_round === "sheriff_pk_vote" ? "警长PK票" : "警长票")
        : "放逐票";
      row.className = "chat-bubble summary-bubble vote-sheet";
      row.innerHTML = `
        <div class="summary-meta">
          <strong>${voteTitle}</strong>
          <span>${vote.day ? `第 ${vote.day} 天` : ""}</span>
        </div>
        <div class="vote-line">
          <span class="vote-actor">${vote.voter_names.join("、")}</span>
          <span class="vote-arrow">共同投给</span>
          <span class="vote-target">${vote.target_name}</span>
        </div>
      `;
    }
    historyBoard.appendChild(row);
  });
  maybeStickBoardToBottom(historyBoard);
}

function rebuildTargetOptions(snapshot) {
  const legalTargets = snapshot.players.filter((player) =>
    snapshot.human_target_candidates.includes(player.id)
  );
  setSelectOptionsIfChanged(
    targetSelect,
    "target",
    legalTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "无目标 / 跳过"
  );
  setSelectOptionsIfChanged(
    wolfTargetSelect,
    "wolfTarget",
    legalTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "请选择狼队目标"
  );
  setSelectOptionsIfChanged(
    voteSelect,
    "vote",
    legalTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` })
  );
  const sheriffCandidates = (snapshot.sheriff_candidates || [])
    .map((candidateId) => snapshot.players.find((item) => item.id === candidateId))
    .filter(Boolean);
  setSelectOptionsIfChanged(
    sheriffVoteSelect,
    "sheriffVote",
    sheriffCandidates,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "请选择候选人"
  );
  setSelectOptionsIfChanged(
    badgeTargetSelect,
    "badgeTarget",
    legalTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "请选择移交对象"
  );

  rebuildNightActionOptions(snapshot.human_allowed_night_actions || [], snapshot);
}

function rebuildNightActionOptions(allowedActions, snapshot = null) {
  const labels = {
    skip: "跳过",
    inspect: "查验",
    guard: "守护",
    save: "救人",
    poison: "毒人",
  };

  const filteredActions = (allowedActions.length ? allowedActions : ["skip"])
    .filter((action) => action !== "guard" || snapshot?.guard_enabled);
  const actionItems = (filteredActions.length ? filteredActions : ["skip"]).map((action) => ({
    value: action,
    label: labels[action] || action,
  }));
  setSelectOptionsIfChanged(
    nightActionType,
    "nightAction",
    actionItems,
    (item) => ({ value: item.value, label: item.label })
  );
}

function syncActionPanel(snapshot) {
  const phase = snapshot.phase;
  const humanAlive = snapshot.human_alive;
  const pending = snapshot.pending_human_action || "";

  [
    wolfChatSection,
    sheriffSection,
    directionSection,
    speechSection,
    lastWordsSection,
    nightSection,
    voteSection,
    badgeSection,
    selfDestructSection,
  ].forEach((section) => {
    section.classList.add("hidden");
  });
  spectatorSection.classList.add("hidden");

  resetDisabledState();

  if (phase === "game_over") {
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (snapshot.can_self_destruct) {
    selfDestructSection.classList.remove("hidden");
    selfDestructBtn.disabled = false;
  }

  if (!humanAlive && phase !== "last_words" && phase !== "badge_transfer" && phase !== "hunter_shot") {
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (phase === "wolf_chat") {
    if (snapshot.human_is_wolf) {
      wolfChatSection.classList.remove("hidden");
      const myTurn = snapshot.current_speaker_id === snapshot.human_player_id;
      const lockedPlan = Boolean(snapshot.wolf_night_plan?.locked);
      const wolfChatLocked = state.lockedActions.wolf_chat;
      wolfChatBtn.disabled = !myTurn || lockedPlan || state.lockedActions.wolf_chat;
      wolfConfirmBtn.disabled = !myTurn || lockedPlan || wolfChatLocked;
      wolfChatInput.disabled = !myTurn || lockedPlan || wolfChatLocked;
      wolfTargetSelect.disabled = !myTurn || lockedPlan || wolfChatLocked;
      return;
    }
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (phase === "night") {
    const onlySkip = (snapshot.human_allowed_night_actions || []).length === 1 &&
      snapshot.human_allowed_night_actions[0] === "skip";
    if (onlySkip) {
      spectatorSection.classList.remove("hidden");
      return;
    }
    nightSection.classList.remove("hidden");
    nightBtn.disabled = state.lockedActions.night;
    targetSelect.disabled = state.lockedActions.night;
    nightActionType.disabled = state.lockedActions.night;
    return;
  }

  if (phase === "sheriff_election" || phase === "sheriff_speech" || phase === "sheriff_vote" || phase === "sheriff_pk_speech" || phase === "sheriff_pk_vote") {
    if (!snapshot.sheriff_enabled) {
      spectatorSection.classList.remove("hidden");
      return;
    }
    const isElection = phase === "sheriff_election";
    const isSpeechPhase = phase === "sheriff_speech" || phase === "sheriff_pk_speech";
    const isVotePhase = phase === "sheriff_vote" || phase === "sheriff_pk_vote";
    const mySpeechTurn = isSpeechPhase && snapshot.current_speaker_id === snapshot.human_player_id;
    const canHumanVoteSheriff = isVotePhase && snapshot.human_target_candidates.length > 0;
    sheriffSection.classList.remove("hidden");
    sheriffBtn.disabled = state.lockedActions.sheriff || (!isElection && !mySpeechTurn && !canHumanVoteSheriff);
    runForSheriffCheckbox.disabled = !isElection || state.lockedActions.sheriff;
    sheriffVoteSelect.disabled = !canHumanVoteSheriff || state.lockedActions.sheriff;
    sheriffSpeechInput.disabled = !mySpeechTurn || state.lockedActions.sheriff;
    if (!isElection && !mySpeechTurn && !canHumanVoteSheriff) {
      spectatorSection.classList.remove("hidden");
    }
    return;
  }

  if (snapshot.sheriff_enabled && phase === "day_speech" && snapshot.available_speech_directions?.length) {
    directionSection.classList.remove("hidden");
    directionLeftBtn.disabled = state.lockedActions.direction;
    directionRightBtn.disabled = state.lockedActions.direction;
    return;
  }

  if (phase === "day_speech") {
    if (snapshot.current_speaker_id === snapshot.human_player_id || pending === "day_speech") {
      speechSection.classList.remove("hidden");
      speechBtn.disabled = state.lockedActions.speech;
      speechInput.disabled = state.lockedActions.speech;
      return;
    }
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (phase === "exile_pk_speech") {
    if (snapshot.current_speaker_id === snapshot.human_player_id || pending === "exile_pk_speech") {
      speechSection.classList.remove("hidden");
      speechBtn.disabled = state.lockedActions.speech;
      speechInput.disabled = state.lockedActions.speech;
      return;
    }
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (phase === "day_vote" || phase === "exile_pk_vote" || phase === "hunter_shot") {
    const humanPlayer = snapshot.players.find((player) => player.id === snapshot.human_player_id);
    if (phase !== "hunter_shot" && !humanPlayer?.can_vote) {
      spectatorSection.classList.remove("hidden");
      return;
    }
    voteSection.classList.remove("hidden");
    const voteLabel = voteSection.querySelector("label[for='voteSelect']");
    if (voteLabel) {
      voteLabel.textContent = phase === "hunter_shot" ? "猎人开枪目标" : "白天投票";
    }
    voteBtn.textContent = phase === "hunter_shot" ? "提交开枪目标" : "提交投票";
    voteBtn.disabled = state.lockedActions.vote;
    voteSelect.disabled = state.lockedActions.vote;
    return;
  }

  if (phase === "last_words") {
    if (pending === "last_words") {
      lastWordsSection.classList.remove("hidden");
      lastWordsBtn.disabled = state.lockedActions.last_words;
      lastWordsInput.disabled = state.lockedActions.last_words;
      return;
    }
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (phase === "badge_transfer") {
    if (!snapshot.sheriff_enabled) {
      spectatorSection.classList.remove("hidden");
      return;
    }
    if (pending === "badge_transfer") {
      badgeSection.classList.remove("hidden");
      badgeTransferBtn.disabled = state.lockedActions.badge;
      badgeTearBtn.disabled = state.lockedActions.badge;
      badgeTargetSelect.disabled = state.lockedActions.badge;
      return;
    }
    spectatorSection.classList.remove("hidden");
    return;
  }

  spectatorSection.classList.remove("hidden");
}

function resetDisabledState() {
  const voteLabel = voteSection.querySelector("label[for='voteSelect']");
  if (voteLabel) {
    voteLabel.textContent = "白天投票";
  }
  voteBtn.textContent = "提交投票";

  [
    wolfChatBtn,
    wolfConfirmBtn,
    sheriffBtn,
    speechBtn,
    lastWordsBtn,
    nightBtn,
    voteBtn,
    badgeTransferBtn,
    badgeTearBtn,
    selfDestructBtn,
    directionLeftBtn,
    directionRightBtn,
  ].forEach((button) => {
    button.disabled = true;
  });

  [
    wolfChatInput,
    sheriffSpeechInput,
    speechInput,
    lastWordsInput,
    wolfTargetSelect,
    sheriffVoteSelect,
    targetSelect,
    voteSelect,
    badgeTargetSelect,
    nightActionType,
    runForSheriffCheckbox,
  ].forEach((el) => {
    el.disabled = true;
  });
}

function syncActionLocks(snapshot, previousSnapshot = null) {
  if (
    previousSnapshot &&
    (
      previousSnapshot.phase !== snapshot.phase ||
      previousSnapshot.day !== snapshot.day ||
      previousSnapshot.night_id !== snapshot.night_id ||
      previousSnapshot.pending_human_action !== snapshot.pending_human_action ||
      previousSnapshot.current_speaker_id !== snapshot.current_speaker_id
    )
  ) {
    Object.keys(state.lockedActions).forEach((key) => {
      state.lockedActions[key] = false;
    });
  }

  if (snapshot.pending_human_action !== "wolf_chat") {
    state.lockedActions.wolf_chat = false;
  }
  if (!["sheriff_election", "sheriff_speech", "sheriff_vote", "sheriff_pk_speech", "sheriff_pk_vote"].includes(snapshot.phase)) {
    state.lockedActions.sheriff = false;
  }
  if (snapshot.pending_human_action !== "day_speech" && snapshot.pending_human_action !== "exile_pk_speech") {
    state.lockedActions.speech = false;
  }
  if (snapshot.pending_human_action !== "last_words") {
    state.lockedActions.last_words = false;
  }
  if (snapshot.phase !== "night") {
    state.lockedActions.night = false;
  }
  if (!["day_vote", "exile_pk_vote", "hunter_shot"].includes(snapshot.phase)) {
    state.lockedActions.vote = false;
  }
  if (snapshot.pending_human_action !== "badge_transfer") {
    state.lockedActions.badge = false;
  }
  if (!(snapshot.phase === "day_speech" && snapshot.available_speech_directions?.length)) {
    state.lockedActions.direction = false;
  }
}

[
  speechFeedBoard,
  eventsBoard,
  historyBoard,
  wolfChatBoard,
].forEach((board) => {
  board.addEventListener("scroll", () => updateBoardScrollLock(board));
});

[
  speechInput,
  sheriffSpeechInput,
  lastWordsInput,
  wolfChatInput,
  targetSelect,
  wolfTargetSelect,
  voteSelect,
  sheriffVoteSelect,
  badgeTargetSelect,
  nightActionType,
  runForSheriffCheckbox,
].forEach((element) => {
  element.addEventListener("blur", () => {
    setTimeout(() => {
      if (!isUserEditingActionPanel()) {
        flushDeferredSnapshot();
      }
    }, 0);
  });
  element.addEventListener("change", () => {
    if (!isUserEditingActionPanel()) {
      flushDeferredSnapshot();
    }
  });
});

function formatNullable(value) {
  return value === null || value === undefined ? "无" : value;
}

function playerName(id, players) {
  const player = players.find((item) => item.id === id);
  return player ? player.name : `玩家${id + 1}`;
}

function formatTarget(id, players) {
  if (id === null || id === undefined) return "无";
  return `${playerName(id, players)}（${id + 1}号）`;
}

function formatSpeechType(type) {
  const map = {
    campaign: "警上发言",
    pk_campaign: "PK 发言",
    exile_pk: "放逐PK发言",
    day: "白天发言",
    last_words: "遗言",
  };
  return map[type] || type;
}

function formatWolfStance(stance) {
  const map = {
    proposal: "提出刀口",
    support: "支持当前刀口",
    switch: "改刀建议",
    skip: "未给目标",
  };
  return map[stance] || stance;
}

function formatPendingAction(action) {
  const map = {
    wolf_chat: "狼队夜聊发言",
    night: "夜间技能选择",
    day_speech: "白天发言",
    day_vote: "白天放逐投票",
    exile_pk_speech: "放逐PK发言",
    exile_pk_vote: "放逐PK投票",
    last_words: "遗言",
    hunter_shot: "猎人开枪",
    badge_transfer: "警徽移交",
    choose_speech_order: "选择发言顺序",
    sheriff_election: "警长竞选选择",
    sheriff_speech: "警上发言",
    sheriff_vote: "警长投票",
    sheriff_pk_speech: "警长PK发言",
    sheriff_pk_vote: "警长PK投票",
  };
  return map[action] || "玩家操作";
}

function formatPhase(phase) {
  const map = {
    setup: "准备",
    wolf_chat: "狼人夜聊",
    night: "夜晚",
    sheriff_election: "上警报名",
    sheriff_speech: "警上发言",
    sheriff_vote: "警长投票",
    sheriff_pk_speech: "警长PK发言",
    sheriff_pk_vote: "警长PK投票",
    exile_pk_speech: "放逐PK发言",
    exile_pk_vote: "放逐PK投票",
    last_words: "遗言",
    hunter_shot: "猎人开枪",
    badge_transfer: "警徽移交",
    day_speech: "白天发言",
    day_vote: "放逐投票",
    sheriff: "警长流程",
    speech: "白天发言",
    vote: "投票",
    badge: "警徽",
    hunter: "猎人技能",
    explode: "狼人自爆",
    result: "结算",
  };
  return map[phase] || phase;
}
