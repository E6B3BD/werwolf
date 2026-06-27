# Werwolf

基于 `OpenAI Agents SDK` + `FastAPI` 的 Web 狼人杀项目。  
这个版本参考了同目录下的 `lying_man`，但做了几个关键调整：

- 使用 Web 页面游玩，而不是终端交互。
- 每局随机选择一个真人玩家位置，不是全 AI 对战。
- AI 玩家决策层改成 OpenAI Agents SDK 结构，规则引擎仍然由 Python 代码确定性执行。
- 目录按 `配置 / Agent / 规则引擎 / API / 前端` 分层，便于继续扩展角色与玩法。

## 1. 当前版本能力

当前默认主规则支持：

- 12 人预女猎白竞技规则
- 真人随机获得一个角色
- 其余玩家由 AI 驱动
- 狼人夜晚协商阶段
  - 真人是狼人时，可以和 AI 狼夜聊并提交刀人目标
  - 纯 AI 狼也会先协商再落刀
- 白天发言
- 白天投票
- 夜晚基础行动
  - 狼人协商后刀人
  - 预言家查验
  - 女巫救人 / 毒人
- 猎人出局开枪
- 白痴被白天放逐时翻牌免出局并失去投票权
- 白天放逐 PK
- 狼人白天 / 警上自爆
- Web 页面查看玩家状态、事件日志、常驻发言记录、投票记录、狼人夜聊记录

扩展规则能力不属于默认主规则，不能混入标准板：

- 守卫、警长、警徽、丘比特、狼美人、白狼王等
- 多真人联机
- 复盘分析、赛后报告

## 2. 项目结构

```text
werwolf/
  .env                     # OpenAI 配置与本地运行配置
  pyproject.toml           # Python 依赖
  requirements.txt         # pip 依赖清单
  README.md
  app/
    __init__.py
    main.py                # FastAPI 入口
    core/
      config.py            # 配置加载
    agents/
      prompts.py           # AI 玩家提示词
      runtime.py           # OpenAI Agents SDK 运行时封装
    engine/
      models.py            # 核心数据模型
      game.py              # 规则引擎与游戏管理
    api/
      routes.py            # HTTP API
    templates/
      index.html           # 页面模板
    static/
      style.css            # 页面样式
      app.js               # 前端交互逻辑
  tests/
```

## 3. 环境要求

按你的要求，建议使用 `conda GE` 环境。

示例：

```bash
conda activate GE
cd /Users/e6b3bd/work/狼人杀/werwolf
python -m pip install -U pip
python -m pip install -e .
```

如果你更习惯 `requirements.txt`：

```bash
conda activate GE
cd /Users/e6b3bd/work/狼人杀/werwolf
python -m pip install -r requirements.txt
```

如果你更习惯非 editable 安装，也可以：

```bash
conda activate GE
cd /Users/e6b3bd/work/狼人杀/werwolf
python -m pip install .
```

## 4. 配置 .env

项目根目录已经预留了 `.env` 文件：

```env
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
HOST=127.0.0.1
PORT=8008
```

你需要至少补充：

```env
OPENAI_API_KEY=你的真实密钥
```

说明：

- `OPENAI_BASE_URL` 已预留接口，默认是官方地址。
- 如果你后面要接代理网关或兼容服务，可以直接改这个值。
- 如果没有配置 `OPENAI_API_KEY`，项目仍能运行，但 AI 会走本地兜底逻辑，只适合界面联调，不适合作为正式体验。

## 5. 启动方式

```bash
conda activate GE
cd /Users/e6b3bd/work/狼人杀/werwolf
uvicorn app.main:app --reload --host 127.0.0.1 --port 8008
```

然后打开浏览器：

```text
http://127.0.0.1:8008
```

健康检查接口：

```text
http://127.0.0.1:8008/health
```

## 6. 为什么这里用 OpenAI Agents SDK

这个项目不是把全部规则交给 LLM，而是只把“玩家决策”交给 Agent。

### 保留 Python 规则引擎的原因

狼人杀本身是强规则、强阶段、强结算游戏：

- 谁能行动
- 谁能被投
- 谁能被救
- 谁胜谁负

这些都应该由确定性的 Python 代码负责，不应该交给模型自由发挥。

### 用 Agents SDK 负责 AI 玩家决策的原因

它比原来 `lying_man` 里那种“自然语言输出 + 正则抽取 ID”更稳：

- 更容易做结构化输出
- 更容易后续增加 guardrail
- 更容易加 tracing / 调试
- 更适合未来扩展多 Agent 协商、赛后复盘 Agent

## 7. 当前对局流程

当前版本的基础流程是：

1. 狼人协商阶段 `wolf_chat`
   - 狼队先讨论今晚刀谁
   - 真人如果是狼人，可以输入夜聊内容并提交建议目标
2. 夜晚结算阶段 `night`
   - 守卫、预言家、女巫等角色执行技能
3. 公布死讯并处理遗言
4. 白天发言阶段 `day_speech`
5. 白天投票阶段 `day_vote`
6. 白痴翻牌、遗言、猎人开枪等衍生结算
7. 进入下一天，重新开始狼人协商

## 8. 后续建议

如果你准备继续做第二版，我建议优先补这些：

1. 扩展更多角色和板子
2. 给 AI 决策增加严格 JSON schema 和更强校验
3. 完善死亡原因、公开身份等复盘字段
4. 引入持久化存档
5. 增加赛后复盘页
6. 增加多真人房间模式

## 9. 注意

当前版本是一个“可运行的重构起点”，不是完全复刻 `lying_man` 全规则的终版。  
它已经把目录、Web 交互、OpenAI 接入边界、规则引擎与 Agent 层分离开了，后续继续做会比在 `lying_man` 上硬改更稳。
