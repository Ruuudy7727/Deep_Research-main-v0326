# RAG 图文一体处理说明（Deep_Research v0326）

本文档聚焦本项目 RAG 子系统，尤其是“图文一体证据”能力：离线如何构建，在线如何检索与回传，为什么会出现“图片数量差异”，以及如何调参。

---

## 一句话总结

本项目采用“两段式 RAG”：

- **Step1**：把文档解析为可检索的文本分片，并保留分片到图片的关联关系；
- **Step2.5**：把分片向量化写入 Chroma，在线检索时恢复图片元数据并与文本证据一起返回。

---

## 1. 设计目标与特色

### 1.1 为什么做“图文一体”

纯文本检索在设备故障、标准规范、流程示意图场景下有明显短板：关键证据可能在图中（流程图、结构图、波形图、阈值图）。  
因此本项目把“文本证据”和“图片证据”绑定在同一个 chunk 的 metadata 中，确保：

- 检索排序基于文本语义；
- 证据展示可同时给出关联图片；
- 最终回答阶段可选择多模态模型读取图片，减少“只看文字”的信息损失。

### 1.2 核心思路

- **向量只嵌入清洗后的文本**，避免图片 hash 路径污染语义；
- **图片引用单独存 metadata**（`image_paths`），检索命中后再恢复；
- **前后端共用同一套图片路径规范**（`images/doc_id/file.jpg` -> `/kb_images/...`）。

---

## 2. 离线构建流程（Step1 + Step2.5）

## 2.1 Step1：解析、切分、图文绑定

入口：`step1_build_konwledge.py`

主要流程：

1. **文档解析**
   - 使用 `MineruParser.parse_document()` 解析输入目录文件；
   - Office/文本会先转 PDF，再交给 MinerU；
   - 图片文件强制走 OCR。

2. **按目录策略切分**
   - `h1` 目录：按一级标题切分；
   - `newline` 目录：按空行切分；
   - 其他目录：使用默认切分策略（可配置）。

3. **提取图片并清洗文本**
   - 从 chunk 中提取 Markdown 图片语法 `![...](...)`；
   - 同时删除该语法，得到 `cleaned_content`（用于 embedding）。

4. **chunk 过滤并写 KV**
   - 过滤条件：`cleaned_content` 非空且 token 数 >= `min_chunk_tokens`（默认 25）；
   - 合格 chunk 写入 `rag_data/all/kv_store_text_chunks.json`；
   - metadata 包含 `image_paths`、`file_path`、`full_doc_id` 等字段。

5. **图片归档复制**
   - 把文档中收集到的图片引用统一复制到 `rag_data/all/images/<doc_id>/`；
   - 复制动作按文档级集合执行，不依赖某个 chunk 是否最终入库。

产物：

- `rag_data/all/kv_store_text_chunks.json`
- `rag_data/all/images/<doc_id>/*`

---

## 2.2 Step2.5：向量化并写入 Chroma

入口：`step2.5_json2chroma.py`

主要流程：

1. 读取 `kv_store_text_chunks.json`；
2. 对每条 `content` 调 embedding（批处理 + 限速 + 重试）；
3. 清洗 metadata 后写入 Chroma 集合 `raptor_kb`。

注意：

- Chroma metadata 只支持标量，`image_paths` 列表会被压成分号字符串；
- 在线检索阶段会再把该字符串还原为列表。

---

## 3. 在线检索与图文证据回传

### 3.1 检索主链路

入口：`deep_research/utils.py` 中 `unified_local_search()`

- 自动初始化本地检索器；
- 执行混合检索（向量 + BM25）；
- 分数标准化后返回 Top-K 结果。

### 3.2 图文证据增强（server 层 patch）

入口：`server_plus.py` 的 RAG image-capture patch

做了三件关键事：

1. **补齐 metadata**  
   原检索返回可能只带 `source/score/type`，patch 会把 `image_paths/file_path/full_doc_id` 等补回；

2. **抓取图片证据**  
   从命中结果中抽取有效图片，写入 `SHARED_STATE["retrieved_images"]`；

3. **抓取文本证据块**  
   把高分 chunk（可附带图片 URL）写入 `SHARED_STATE["evidence_chunks"]`，供前端展示与报告引用。

### 3.3 最终回答阶段多模态

在满足阶段与模式条件时，系统会把检索到的图片一并注入模型调用（multimodal），并记录 `answer_images`，用于：

- 最终答案正文按编号引用图片；
- 前端统一展示“答案使用到的图”。

---

## 4. 为什么会出现“图片数量差异”

常见现象：`rag_data/all/images` 里的图片数 > `kv_store` 里可追溯到的图片数。

这不是丢图，而是口径差异：

- `images` 目录统计的是“文档中被引用并归档的图”；
- `kv_store.image_paths` 统计的是“进入合格 chunk 的图”。

某些图片所在 chunk 在清洗后文本过短（例如只剩标题），会被 `min_chunk_tokens` 过滤掉：

- 图片仍会复制到 `rag_data/all/images`；
- 但该图不会出现在 `kv_store` 对应 chunk 的 `image_paths` 中。

---

## 5. 调参与实践建议

### 5.1 想让更多图片参与检索证据

- 降低 `--min-chunk-tokens`（例如从 25 降到 8~15）；
- 或优化切分策略，避免“图单独成很短分片”。

### 5.2 想降低噪声图

- 保持较高 token 阈值；
- 对 markdown 图片提取规则增加白名单或路径约束（按文档类型筛图）。

### 5.3 想提升在线稳定性

- 保留 embedding 批处理限速和重试配置；
- 监控 `retrieved_images` 与 `evidence_chunks` 数量，防止前端过载。

---

## 6. 推荐执行顺序

```bash
# Step1: 生成 kv + 归档图片
python step1_build_konwledge.py

# Step2.5: 基于 kv 重建 Chroma
python step2.5_json2chroma.py
```

---

## 7. 自检清单（图文一体是否生效）

- `rag_data/all/kv_store_text_chunks.json` 中存在 `image_paths` 字段；
- `rag_data/all/images/<doc_id>/` 下存在对应图片文件；
- 在线检索后状态中可见 `retrieved_images` / `evidence_chunks`；
- 最终回答阶段可见图片证据引用（文本+图一致）。

---

## 8. 结论

本项目“图文一体 RAG”并不是把图片直接向量化检索，而是采用更稳妥的工程化路径：

- 文本负责召回与排序；
- 图片作为证据上下文随 chunk 回传；
- 在最终回答阶段按需多模态增强。

这样兼顾了召回质量、系统复杂度与可观测性，适合当前标准文档/故障分析类知识库场景。

