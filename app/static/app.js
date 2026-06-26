const state = {
  gameId: null,
  snapshot: null,
  countdownTimer: null,
  pollTimer: null,
  renderedSpeechKeys: new Set(),
  deferredSnapshot: null,
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
  return response.json();
}

createGameBtn.addEventListener("click", async () => {
  const snapshot = await requestJson("/api/games", {
    player_count: Number(playerCount.value),
  });
  applySnapshot(snapshot);
});

refreshBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  const response = await fetch(`/api/games/${state.gameId}`);
  const snapshot = await response.json();
  applySnapshot(snapshot);
});

speechBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (speechBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/speech`, {
    content: speechInput.value,
  });
  speechInput.value = "";
  state.lockedActions.speech = true;
  applySnapshot(snapshot);
});

nightBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (nightBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/night`, {
    action_type: nightActionType.value,
    target_id: targetSelect.value === "" ? null : Number(targetSelect.value),
  });
  state.lockedActions.night = true;
  applySnapshot(snapshot);
});

voteBtn.addEventListener("click", async () => {
  if (!state.gameId || voteSelect.value === "") return;
  if (voteBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/vote`, {
    target_id: Number(voteSelect.value),
  });
  state.lockedActions.vote = true;
  applySnapshot(snapshot);
});

wolfChatBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (wolfChatBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/wolf-chat`, {
    action_type: "wolf_kill",
    target_id: wolfTargetSelect.value === "" ? null : Number(wolfTargetSelect.value),
    chat_content: wolfChatInput.value,
  });
  wolfChatInput.value = "";
  state.lockedActions.wolf_chat = true;
  applySnapshot(snapshot);
});

sheriffBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (sheriffBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/sheriff`, {
    run_for_sheriff: runForSheriffCheckbox.checked,
    vote_target_id: sheriffVoteSelect.value === "" ? null : Number(sheriffVoteSelect.value),
    speech: sheriffSpeechInput.value,
  });
  sheriffSpeechInput.value = "";
  state.lockedActions.sheriff = true;
  applySnapshot(snapshot);
});

lastWordsBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (lastWordsBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/last-words`, {
    content: lastWordsInput.value,
  });
  lastWordsInput.value = "";
  state.lockedActions.last_words = true;
  applySnapshot(snapshot);
});

badgeTransferBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (badgeTransferBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/badge`, {
    badge_target_id: badgeTargetSelect.value === "" ? null : Number(badgeTargetSelect.value),
    tear_badge: false,
  });
  state.lockedActions.badge = true;
  applySnapshot(snapshot);
});

badgeTearBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (badgeTearBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/badge`, {
    tear_badge: true,
  });
  state.lockedActions.badge = true;
  applySnapshot(snapshot);
});

selfDestructBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/self-destruct`);
  applySnapshot(snapshot);
});

directionLeftBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (directionLeftBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/speech-order`, {
    speech_order_direction: "left",
  });
  state.lockedActions.direction = true;
  applySnapshot(snapshot);
});

directionRightBtn.addEventListener("click", async () => {
  if (!state.gameId) return;
  if (directionRightBtn.disabled) return;
  const snapshot = await requestJson(`/api/games/${state.gameId}/speech-order`, {
    speech_order_direction: "right",
  });
  state.lockedActions.direction = true;
  applySnapshot(snapshot);
});

function applySnapshot(snapshot) {
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
      snapshot.pending_human_action ? `当前需要执行：${snapshot.pending_human_action}` : "当前无强制操作标记";
  }
  if (currentHint) {
    currentHint.textContent = snapshot.current_hint || "等待下一步。";
  }

  if (privateMessage && snapshot.human_private_message) {
    privateMessage.classList.remove("hidden");
    privateMessage.textContent = snapshot.human_private_message;
  } else if (privateMessage) {
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
  renderEvents(snapshot.events);
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
    const shouldRevealRole = snapshot.phase === "game_over" || player.is_human || isWolfMate;
    const roleText = shouldRevealRole ? `身份：${player.role}` : "身份：未知";
    row.className = `player-row ${player.alive ? "alive" : "dead"} ${isCurrentSpeaker ? "active-speaker" : ""}`.trim();
    row.innerHTML = `
      <div class="player-topline">
        <strong>${player.name}${player.is_sheriff ? " · 警长" : ""}</strong>
        ${player.is_human ? `<span class="player-tag self-tag">你</span>` : ""}
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
  if (!snapshot.wolf_chat_records.length) {
    wolfChatBoard.innerHTML = `<div class="empty-state">本轮暂无狼人夜聊。</div>`;
    return;
  }
  snapshot.wolf_chat_records.forEach((record) => {
    const row = document.createElement("div");
    row.className = "wolf-chat-card";
    row.innerHTML = `
      <div class="speech-meta">
        <strong>${record.player_name}</strong>
        <span class="seat-badge">${record.player_id + 1}号位</span>
      </div>
      <div class="wolf-chat-content">${record.content}</div>
      <div class="summary-meta">建议目标：${formatTarget(record.proposed_target_id, snapshot.players)}</div>
    `;
    wolfChatBoard.appendChild(row);
  });
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
    ["day_speech", "sheriff_speech", "sheriff_pk_speech", "last_words"].includes(snapshot.phase)
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

function renderEvents(events) {
  eventsBoard.innerHTML = "";
  const uniqueEvents = events.filter((event, index, source) =>
    index === source.findIndex((item) => item.phase === event.phase && item.message === event.message)
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
  const aliveTargets = snapshot.players.filter((player) =>
    snapshot.human_target_candidates.includes(player.id)
  );
  setSelectOptionsIfChanged(
    targetSelect,
    "target",
    aliveTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "无目标 / 跳过"
  );
  setSelectOptionsIfChanged(
    wolfTargetSelect,
    "wolfTarget",
    aliveTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "请选择狼队目标"
  );
  setSelectOptionsIfChanged(
    voteSelect,
    "vote",
    aliveTargets,
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
    aliveTargets,
    (player) => ({ value: String(player.id), label: `${player.name}（座位 ${player.id + 1}）` }),
    "请选择移交对象"
  );

  rebuildNightActionOptions(snapshot.human_allowed_night_actions || []);
}

function rebuildNightActionOptions(allowedActions) {
  const labels = {
    skip: "跳过",
    inspect: "查验",
    guard: "守护",
    save: "救人",
    poison: "毒人",
  };

  const actionItems = (allowedActions.length ? allowedActions : ["skip"]).map((action) => ({
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

  if (!humanAlive && phase !== "last_words" && phase !== "badge_transfer") {
    spectatorSection.classList.remove("hidden");
    return;
  }

  if (phase === "wolf_chat") {
    if (snapshot.human_is_wolf) {
      wolfChatSection.classList.remove("hidden");
      const myTurn = snapshot.current_speaker_id === snapshot.human_player_id;
      wolfChatBtn.disabled = !myTurn || state.lockedActions.wolf_chat;
      wolfChatInput.disabled = !myTurn || state.lockedActions.wolf_chat;
      wolfTargetSelect.disabled = !myTurn || state.lockedActions.wolf_chat;
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

  if (phase === "day_speech" && snapshot.available_speech_directions?.length) {
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

  if (phase === "day_vote") {
    const humanPlayer = snapshot.players.find((player) => player.id === snapshot.human_player_id);
    if (!humanPlayer?.can_vote) {
      spectatorSection.classList.remove("hidden");
      return;
    }
    voteSection.classList.remove("hidden");
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
  [
    wolfChatBtn,
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
  if (snapshot.pending_human_action !== "day_speech") {
    state.lockedActions.speech = false;
  }
  if (snapshot.pending_human_action !== "last_words") {
    state.lockedActions.last_words = false;
  }
  if (snapshot.phase !== "night") {
    state.lockedActions.night = false;
  }
  if (snapshot.phase !== "day_vote") {
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
    day: "白天发言",
    last_words: "遗言",
  };
  return map[type] || type;
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
    last_words: "遗言",
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
