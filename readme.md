# Deep Research (Battery Domain)

一个面向电池故障分析场景的 Deep Research 项目，支持：

- 多智能体/单智能体任务编排（LangGraph）
- 本地知识库检索（Chroma + BM25 融合）
- 业务数据库查询分析（电池告警数据）
- 图表生成（MCP Chart Server）
- Web UI（Gradio）与 HTTP API（FastAPI）两种访问方式

---

## 1. 核心能力

- **快速模式（Simple）**：适合数据库查询、知识库检索、快速回答。
- **深度模式（Deep Research）**：包含复杂度判断、检索、分析、草稿与最终报告生成。
- **可观测性**：运行日志、节点事件、节点输出、最终报告均可落盘到 `log/`。
- **本地知识库**：支持 `rag_data/all` 与 `rag_data/plan` 双库检索。

---

## 2. 项目结构

```text
.
├─ deep_research/                  # 核心包（agent、prompt、state、工具）
│  ├─ research_agent_full.py       # 深度研究主流程
│  ├─ single_agent_supervisor.py   # 简单任务流程
│  ├─ multi_agent_supervisor.py    # 多智能体监督器
│  ├─ research_agent_scope.py      # 任务澄清、预检索、草稿节点
│  ├─ research_agent_analyze.py    # 数据库分析节点
│  ├─ research_agent_draw.py       # 图表生成节点
│  ├─ local_db.py                  # 本地向量库/BM25检索
│  ├─ utils.py                     # 工具聚合与通用逻辑
│  ├─ gemini_chat.py               # LLM调用封装
│  └─ embedding_client.py          # Embedding调用封装
├─ rag_data/
│  ├─ all/                         # 主知识库（chroma + kv + vector）
│  └─ plan/                        # 计划知识库（chroma + kv + vector）
├─ dbdata/                         # 业务数据与转换脚本（xlsx/db）
├─ log/                            # 运行日志与报告输出
├─ server.py                       # Web UI 入口（FastAPI + Gradio）
├─ api_server.py                   # API 服务入口（FastAPI）
├─ step1_build_konwledge.py        # 文档解析 + 切分 + embedding构建
├─ step2.5_json2chroma.py          # 从kv json重建 Chroma
└─ docker-compose.yml              # 容器编排（app + chart server）
```

---

## 3. 环境要求

- Python 3.11+
- 建议使用虚拟环境（conda 或 venv）
- 如需图表服务（MCP），建议配合 Docker

依赖文件：

- `requirements.txt`
- `environment.yml`

---

## 4. 配置说明（.env）

请在项目根目录准备 `.env`，至少包含以下关键项（按你的实际网关/平台填写）：

```bash
# LLM
MIDEA_API_KEY=xxx
MIDEA_AIGC_USER=your_user
GEMINI_AIMP_BIZ_ID=gemini-2.5-flash
GEMINI_MODEL=gemini-2.5-flash

# RPO（可选，流式最终报告通道）
MIDEA_API_KEY_RPO=xxx
GEMINI_AIMP_BIZ_ID_RPO=xxx
GEMINI_MODEL_RPO=xxx

# Embedding
EMBED_API_KEY=xxx
EMBED_BASE_URL=https://aimpapi.midea.com/t-aigc/aimp-text-embedding/v1
EMBED_MODEL=Qwen3-Embedding-4B

# 检索库
CHROMA_PERSIST_DIR=./rag_data/all
CHROMA_PERSIST_PLAN_DIR=./rag_data/plan
CHROMA_COLLECTION_NAME=raptor_kb

# 外部搜索（可选）
TAVILY_API_KEY=xxx
EXTERNAL_SEARCH_ENABLED=1

# API 服务鉴权
PUBLIC_API_TOKEN=change_me

# 图表MCP（默认本地）
MCP_CHART_SERVER_URL=http://localhost:1122
```

---

## 5. 启动方式

### 5.1 启动 Web UI（推荐体验）

```bash
python server.py
```

默认访问（见控制台输出）：

- `http://localhost:50221`

---

### 5.2 启动 API 服务

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

接口示例：

- `POST /api/v1/run`：提交任务（支持 `deep_mode`）
- `GET /api/v1/runs`：查询历史运行
- `GET /api/v1/runs/{run_dir}/final_report`：获取最终报告
- `GET /api/v1/runs/{run_dir}/events`：获取节点事件
- `GET /api/v1/runs/{run_dir}/nodes/{node_name}`：获取节点输出

---

### 5.3 Docker 一键启动（含图表服务）

```bash
docker compose up -d --build
```

默认包含两个服务：

- `deep_research_app`（8000）
- `mcp_chart_server`（1122）

---

## 6. 知识库构建流程

### Step 1：解析文档并生成知识分片与向量

```bash
python step1_build_konwledge.py
```

输出到 `rag_data/*` 下的 `kv_store_text_chunks.json` / `vdb_chunks.json`。

### Step 2：从 kv json 重建 Chroma（可选）

```bash
python step2.5_json2chroma.py
```

---

## 7. 日志与输出

每次运行会在 `log/thinkdepth_run_*` 下生成：

- `main_run.log`
- `nodes/`（节点事件）
- `node_outputs/`（节点输出快照）
- `final_report.txt`

---

## 8. 常见问题

- **Q: 图表生成失败？**  
  A: 检查 `MCP_CHART_SERVER_URL` 和 `mcp-server-chart` 是否已启动。

- **Q: 本地知识库检索为空？**  
  A: 检查 `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION_NAME` 是否正确，确认 `rag_data` 已构建。

- **Q: API 返回 401？**  
  A: 请求里传入的 token 需与 `.env` 中 `PUBLIC_API_TOKEN` 一致。

---

## 9. 开发建议

- 推荐先跑通 `server.py`，确认 LLM / Embedding / Chroma 配置。
- 再根据业务需要接入 `api_server.py`。
- 提交代码时请勿上传 `.env` 与敏感凭证。

