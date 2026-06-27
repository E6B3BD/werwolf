const fs = require("fs");
const path = require("path");
const vm = require("vm");

class ClassList {
  constructor() {
    this.items = new Set();
  }
  add(...items) {
    items.forEach((item) => this.items.add(item));
  }
  remove(...items) {
    items.forEach((item) => this.items.delete(item));
  }
  toggle(item, force) {
    if (force === undefined) {
      if (this.items.has(item)) {
        this.items.delete(item);
        return false;
      }
      this.items.add(item);
      return true;
    }
    if (force) this.items.add(item);
    else this.items.delete(item);
    return Boolean(force);
  }
  contains(item) {
    return this.items.has(item);
  }
}

class FakeElement {
  constructor(id = "", tagName = "div") {
    this.id = id;
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.parentElement = null;
    this.classList = new ClassList();
    this.listeners = {};
    this.disabled = false;
    this.value = "";
    this.textContent = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this._innerHTML = "";
  }
  get innerHTML() {
    return this._innerHTML;
  }
  set innerHTML(value) {
    this._innerHTML = String(value);
    if (value === "") {
      this.children = [];
    }
  }
  get options() {
    return this.children;
  }
  appendChild(child) {
    child.parentElement = this;
    this.children.push(child);
    this._innerHTML += child.innerHTML || child.textContent || "";
    return child;
  }
  addEventListener(type, handler) {
    this.listeners[type] ||= [];
    this.listeners[type].push(handler);
  }
  async click() {
    for (const handler of this.listeners.click || []) {
      await handler({ target: this });
    }
  }
  querySelector(selector) {
    if (selector === "label[for='voteSelect']") {
      return document.getElementById("voteSelectLabel");
    }
    return null;
  }
  closest(selector) {
    let current = this;
    while (current) {
      if (selector === ".action-panel" && current.classList.contains("action-panel")) return current;
      current = current.parentElement;
    }
    return null;
  }
  focus() {
    document.activeElement = this;
  }
  setSelectionRange() {}
}

class HTMLTextAreaElement extends FakeElement {}
class HTMLInputElement extends FakeElement {}
class HTMLSelectElement extends FakeElement {}

const ids = [
  "playerCount", "createGameBtn", "refreshBtn", "speechBtn", "nightBtn", "voteBtn",
  "wolfChatBtn", "wolfConfirmBtn", "sheriffBtn", "lastWordsBtn", "badgeTransferBtn",
  "badgeTearBtn", "selfDestructBtn", "directionLeftBtn", "directionRightBtn",
  "speechInput", "sheriffSpeechInput", "lastWordsInput", "wolfChatInput",
  "targetSelect", "wolfTargetSelect", "voteSelect", "sheriffVoteSelect",
  "badgeTargetSelect", "nightActionType", "currentHint", "pendingActionText",
  "timerBadge", "humanRole", "wolfChatCard", "eventsBoard", "historyBoard",
  "speechSection", "wolfChatSection", "sheriffSection", "directionSection",
  "lastWordsSection", "nightSection", "voteSection", "badgeSection",
  "selfDestructSection", "spectatorSection", "wolfChatBoard", "speechFeedBoard",
  "runForSheriffCheckbox", "gameTitle", "gameMeta", "phaseBadge", "privateMessage", "privateContextCard",
  "winnerBadge", "playersBoard", "voteSelectLabel",
];

function buildElement(id) {
  if (id.endsWith("Input") || id === "wolfChatInput" || id === "speechInput" || id === "lastWordsInput" || id === "sheriffSpeechInput") {
    return new HTMLTextAreaElement(id, "textarea");
  }
  if (id.endsWith("Select") || id === "nightActionType" || id === "playerCount") {
    return new HTMLSelectElement(id, "select");
  }
  if (id === "runForSheriffCheckbox") {
    return new HTMLInputElement(id, "input");
  }
  return new FakeElement(id, "div");
}

const elements = new Map(ids.map((id) => [id, buildElement(id)]));
elements.get("playerCount").value = "12";
elements.get("voteSection").appendChild(elements.get("voteSelectLabel"));
const actionPanel = new FakeElement("actionPanel", "div");
actionPanel.classList.add("action-panel");
for (const id of ["speechInput", "sheriffSpeechInput", "lastWordsInput", "wolfChatInput", "targetSelect", "wolfTargetSelect", "voteSelect"]) {
  elements.get(id).parentElement = actionPanel;
}
const rightPanel = new FakeElement("rightPanel", "aside");
rightPanel.classList.add("right-panel");

const document = {
  activeElement: null,
  body: new FakeElement("body", "body"),
  getElementById(id) {
    if (!elements.has(id)) {
      elements.set(id, buildElement(id));
    }
    return elements.get(id);
  },
  querySelector(selector) {
    if (selector === ".right-panel") return rightPanel;
    return null;
  },
  createElement(tagName) {
    if (tagName === "option") return new FakeElement("", "option");
    return new FakeElement("", tagName);
  },
};

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function makePlayers(humanId = 0, humanRole = "狼人") {
  return Array.from({ length: 12 }, (_, id) => ({
    id,
    name: `玩家${id + 1}`,
    role: id === humanId ? humanRole : "平民",
    camp: id < 4 ? "werewolf" : "villager",
    is_human: id === humanId,
    alive: true,
    can_vote: true,
    is_sheriff: false,
  }));
}

function wolfSnapshot() {
  return {
    game_id: "domtest",
    snapshot_seq: 1,
    phase: "wolf_chat",
    day: 1,
    night_id: 1,
    sheriff_enabled: false,
    guard_enabled: false,
    human_player_id: 0,
    human_role: "狼人",
    human_alive: true,
    human_is_wolf: true,
    winner: null,
    human_private_context: "你是 1 号位，身份：狼人\n狼队友：2号 玩家2、3号 玩家3、4号 玩家4",
    human_private_message: "",
    current_hint: "狼人协商阶段：轮到你发言并给出刀口建议。",
    pending_human_action: "wolf_chat",
    human_allowed_night_actions: ["wolf_chat", "wolf_confirm"],
    human_target_candidates: [4, 5, 6],
    wolf_teammate_ids: [1, 2, 3],
    wolf_chat_records: [],
    wolf_history_summaries: ["过往夜晚复盘：上夜已形成狼队共识，只保留策略教训，不复述旧夜具体刀口。"],
    wolf_night_plan: { night_id: 1, current_target_id: null, locked: false },
    sheriff_id: null,
    sheriff_candidates: [],
    players: makePlayers(0, "狼人"),
    speeches: [],
    votes: [],
    night_summaries: [],
    events: [
      { phase: "wolf_chat", visibility: "wolf", day: 1, night_id: 0, message: "旧夜最终刀口：玩家9", occurrence_key: "wolf_chat_final:0" },
    ],
    visible_timeline: [
      { item_id: "event:1", kind: "event", phase: "wolf_chat", visibility: "wolf", day: 1, night_id: 1, content: "狼人开始夜聊：先交换刀口收益。", occurrence_key: "wolf_chat_start:1" },
    ],
    current_speaker_id: 0,
    speech_order: [0, 1, 2, 3],
    can_self_destruct: false,
    available_speech_directions: [],
    timer_label: "",
    time_limit_seconds: 0,
    remaining_seconds: 0,
    deadline_ts: null,
  };
}

function nightSnapshot() {
  return {
    ...wolfSnapshot(),
    snapshot_seq: 2,
    phase: "night",
    current_hint: "夜晚阶段：你的角色今晚没有主动技能，系统会自动按跳过处理。",
    pending_human_action: null,
    human_allowed_night_actions: ["skip"],
    human_target_candidates: [],
    wolf_chat_records: [
      { day: 1, night_id: 1, round_id: 1, player_id: 0, speaker_seat_no: 1, player_name: "玩家1", content: "我确认刀5号，先拆带队位置。", proposed_target_id: 4, proposed_target_seat_no: 5, stance_to_previous: "proposal" },
    ],
    wolf_night_plan: { night_id: 1, current_target_id: 4, final_confirmer_id: 0, locked: true },
    visible_timeline: [
      { item_id: "event:2", kind: "event", phase: "night", visibility: "public", day: 1, night_id: 1, content: "进入夜晚行动。", occurrence_key: "night:1" },
    ],
  };
}

function nonWolfSnapshot() {
  return {
    ...wolfSnapshot(),
    snapshot_seq: 3,
    phase: "day_speech",
    human_player_id: 8,
    human_role: "平民",
    human_is_wolf: false,
    human_private_context: "你是 9 号位，身份：平民\n阵营：好人阵营",
    pending_human_action: null,
    human_allowed_night_actions: [],
    human_target_candidates: [],
    wolf_teammate_ids: [],
    wolf_chat_records: [],
    wolf_night_plan: null,
    players: makePlayers(8, "平民"),
    visible_timeline: [
      { item_id: "event:3", kind: "event", phase: "speech", visibility: "public", day: 1, night_id: 1, content: "第 1 天白天发言开始。", occurrence_key: "speech:1" },
    ],
  };
}

let lastRequest = null;
const fetch = async (url, options = {}) => {
  lastRequest = { url, options };
  if (String(url).endsWith("/wolf-chat")) {
    return { ok: true, json: async () => nightSnapshot() };
  }
  return { ok: true, json: async () => wolfSnapshot() };
};

const context = {
  console,
  document,
  fetch,
  setInterval: () => 1,
  clearInterval: () => {},
  setTimeout: (fn) => { fn(); return 1; },
  Date,
  Number,
  String,
  Boolean,
  Array,
  JSON,
  Math,
  WeakMap,
  Set,
  Map,
  HTMLTextAreaElement,
  HTMLInputElement,
  HTMLSelectElement,
};
context.window = context;

const source = fs.readFileSync(path.join(process.cwd(), "app/static/app.js"), "utf8");
vm.runInNewContext(source, context, { filename: "app/static/app.js" });

(async () => {
  await elements.get("createGameBtn").click();
  assert(elements.get("humanRole").textContent === "你的身份：狼人", "human role should render");
  assert(elements.get("privateMessage").textContent.includes("狼队友"), "wolf private context should render teammates");
  assert(!elements.get("wolfChatSection").classList.contains("hidden"), "wolf action section should be visible on own wolf turn");
  assert(elements.get("sheriffSection").classList.contains("hidden"), "sheriff section should stay hidden when extension off");
  assert(elements.get("eventsBoard").innerHTML.includes("狼人开始夜聊"), "current wolf event should render");
  assert(!elements.get("eventsBoard").innerHTML.includes("旧夜最终刀口"), "stale wolf event should not render");
  assert(elements.get("wolfChatBoard").innerHTML.includes("狼队历史摘要"), "wolf history summary should render for wolf");
  assert(elements.get("wolfChatBoard").innerHTML.includes("过往夜晚"), "wolf history content should render for wolf");
  assert(!elements.get("nightActionType").options.some((option) => option.value === "guard"), "guard option should be filtered when extension off");

  elements.get("wolfTargetSelect").value = "4";
  await elements.get("wolfConfirmBtn").click();
  const payload = JSON.parse(lastRequest.options.body);
  assert(payload.action_type === "wolf_confirm", "wolf confirm should post final confirm action");
  assert(payload.target_id === 4, "wolf confirm should post selected target");
  assert(elements.get("phaseBadge").textContent === "夜晚", "phase should update after wolf confirm");
  assert(elements.get("wolfChatSection").classList.contains("hidden"), "wolf section should hide after phase changes");
  const lockedWolfChat = elements.get("wolfChatBoard").innerHTML;
  assert(lockedWolfChat.includes("最终确认"), "locked wolf chat should show the final confirmation record");
  assert(!lockedWolfChat.includes("当前狼队计划"), "locked wolf chat should not duplicate final target as current plan");
  assert(!lockedWolfChat.includes("已锁定"), "locked wolf chat should avoid a second locked-plan target label");

  context.applySnapshot({ ...wolfSnapshot(), snapshot_seq: 1 });
  assert(elements.get("phaseBadge").textContent === "夜晚", "older wolf snapshot must not overwrite newer night snapshot");

  context.applySnapshot(nonWolfSnapshot());
  assert(elements.get("humanRole").textContent === "你的身份：平民", "non-wolf role should render");
  assert(elements.get("privateMessage").textContent.includes("身份：平民"), "non-wolf private context should render");
  assert(!elements.get("privateMessage").textContent.includes("狼队友"), "non-wolf private context must not show wolf teammates");
  assert(elements.get("wolfChatCard").classList.contains("hidden"), "wolf chat card should hide for non-wolf");
  assert(elements.get("wolfChatBoard").innerHTML.includes("你不是狼人"), "non-wolf wolf board should explain hidden chat");
  assert(!elements.get("wolfChatBoard").innerHTML.includes("过往夜晚"), "non-wolf wolf board must not show wolf history");

  context.applySnapshot({ ...nonWolfSnapshot(), phase: "sheriff_election", current_hint: "扩展流程" });
  assert(elements.get("sheriffSection").classList.contains("hidden"), "sheriff UI should stay hidden when disabled by rules");
  assert(!elements.get("spectatorSection").classList.contains("hidden"), "disabled sheriff phase should fall back to spectator state");

  console.log("PASS frontend DOM harness");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
