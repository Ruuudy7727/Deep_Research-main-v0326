# Figure 1 — AI Polish Prompt

Use this prompt with **GPT-4o image / DALL·E / Midjourney / Figma AI / Napkin.ai / Canva AI / draw.io AI**.
Upload `figure/system_architecture.png` as the layout reference.

---

## Prompt (copy from here)

Create a clean, publication-quality system architecture diagram for an academic paper (IEEE-style). Use the attached reference layout as the ground truth — do not change topology or module names.

**Title:** Do not render a title inside the figure (caption goes in the paper as "Figure 1: Overall architecture of the proposed LLM-agent assistant.").

**Layout (left → right, top → bottom):**

1. **Entry:** Box "User Question" (subtitle: maintenance query) → arrow → Box "Complexity-aware Router" (subtitle: routine: alarms, measurements, lookup, charts / complex: root-cause, cross-document reasoning)
2. **Router outputs:**
   - Label **"routine"** → upper lane **"Fast Path: Single-Agent Tool Routing"** (blue theme #2563EB)
   - Label **"complex"** → lower lane **"Deep-Research Path: Multi-Agent Diagnostic Reasoning"** (purple theme #7C3AED)
3. **Fast Path (inside blue rounded container):**
   - "Single-agent Supervisor" (selects next action) → fans out to four tool boxes: **DIRECT** (general response), **RETRIEVE** (knowledge lookup), **DATABASE** (schema-constrained SQL), **CHART** (visualization) → all converge to **"Answer Synthesis"** (concise response, SQL result, or chart)
4. **Deep-Research Path (inside purple rounded container, left-to-right pipeline):**
   - Pre-retrieval (collect initial evidence) → Draft generation (first diagnostic sketch) → Supervisor–researcher coordination (iterative evidence seeking) → Finding compression (merge key findings) → Final report synthesis (diagnosis, evidence, actions)
   - Arrow from Finding compression down to Final report synthesis, labeled **"compressed evidence"**
5. **Data sources (left side, stacked):**
   - Green box: **Structured Operating Data** — alarm events, telemetry, diagnostic tables, cell-level time series
   - Orange/amber box: **Private Maintenance Knowledge Base** — standards, manuals, failure cases, image-linked chunks
   - **Dashed arrows** from Structured Data → DATABASE tool and Deep Path Pre-retrieval; from Knowledge Base → RETRIEVE and Pre-retrieval
6. **Output (bottom-right, green):** **Final Answer / Report** — answer, chart, or diagnostic report; solid arrows from Answer Synthesis and Final report synthesis
7. **Footer note:** Both paths are grounded by structured operating data and the private maintenance knowledge base.

**Style requirements:**

- White/light gray background (#F8FAFC), flat vector, no 3D, no photos
- Rounded rectangles, consistent padding, sans-serif (Arial/Helvetica)
- Solid arrows = control flow; dashed arrows = data grounding
- High contrast for print; minimum 2200×1180 px or vector SVG
- Academic figure: no decorative icons unless subtle (database cylinder, book icon OK)
- English labels only

**Do NOT add:** API names, file paths, LangGraph, Gemini, or implementation details — keep architecture-level only.

---

## Code mapping (for your own edits, not for the figure)

| Figure box | Code module |
|------------|-------------|
| Complexity-aware Router | `clarify_with_user` in `deep_research/research_agent_scope.py` |
| Single-agent Supervisor | `deep_research/single_agent_supervisor.py` |
| Pre-retrieval / Draft | `pre_brief_retrieval`, `write_draft_report` |
| Supervisor–researcher | `deep_research/multi_agent_supervisor.py` |
| Finding compression | `compress_research` in `deep_research/research_agent.py` |
| Final report synthesis | `final_report_generation` in `deep_research/research_agent_full.py` |

---

## Regenerate the draft locally

```powershell
python figure/generate_system_architecture.py
```

Outputs: `figure/system_architecture.{svg,png,pdf}`.
