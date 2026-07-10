"""Open an interactive browser viewer for eval results.jsonl files."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0f1117;
      --panel: #171b26;
      --border: #2a3142;
      --text: #e6eaf2;
      --muted: #8b93a7;
      --accent: #6ea8fe;
      --ok: #3dd68c;
      --warn: #f5c542;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      background: var(--panel);
    }}
    header h1 {{
      margin: 0;
      font-size: 16px;
      font-weight: 600;
    }}
    header .meta {{ color: var(--muted); font-size: 13px; }}
    #search {{
      flex: 1;
      min-width: 220px;
      max-width: 420px;
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
    }}
    .btn {{
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      cursor: pointer;
    }}
    .btn:hover {{ border-color: var(--accent); }}
    main {{
      flex: 1;
      display: grid;
      grid-template-columns: 320px 1fr;
      min-height: 0;
    }}
    #list {{
      border-right: 1px solid var(--border);
      overflow: auto;
      background: var(--panel);
    }}
    .item {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
    }}
    .item:hover {{ background: #1d2333; }}
    .item.active {{ background: #24304a; border-left: 3px solid var(--accent); }}
    .item .id {{ font-weight: 600; font-size: 13px; }}
    .item .sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .badge {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 999px;
      font-size: 11px;
      margin-right: 6px;
      background: #2a3142;
    }}
    .badge.done {{ color: var(--ok); }}
    #detail {{
      overflow: auto;
      padding: 16px;
    }}
    pre.json {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: Consolas, "Cascadia Code", monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
    .key {{ color: #9cdcfe; }}
    .str {{ color: #ce9178; }}
    .num {{ color: #b5cea8; }}
    .bool {{ color: #569cd6; }}
    .null {{ color: #808080; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      #list {{ max-height: 35vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <span class="meta">{count} records</span>
    <input id="search" type="search" placeholder="搜索 id / subset / question..." />
    <button class="btn" id="expandAll">全部展开</button>
    <button class="btn" id="collapseAll">全部折叠</button>
    <button class="btn" id="copyJson">复制当前 JSON</button>
  </header>
  <main>
    <div id="list"></div>
    <div id="detail"><pre class="json" id="jsonOut"></pre></div>
  </main>
  <script>
    const DATA = {data};
    let active = 0;
    let filtered = DATA.map((_, i) => i);

    function esc(s) {{
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }}

    function summary(row) {{
      const q = row.question || "";
      return q.length > 48 ? q.slice(0, 48) + "..." : q;
    }}

    function renderList() {{
      const list = document.getElementById("list");
      list.innerHTML = filtered.map((idx) => {{
        const row = DATA[idx];
        const status = row.status || "?";
        return `<div class="item ${{idx === active ? "active" : ""}}" data-idx="${{idx}}">
          <div class="id">${{esc(row.id || "(no id)")}}</div>
          <div class="sub">
            <span class="badge">${{esc(row.subset || "-")}}</span>
            <span class="badge ${{status}}">${{esc(status)}}</span>
            ${{esc(summary(row))}}
          </div>
        </div>`;
      }}).join("");
      list.querySelectorAll(".item").forEach((el) => {{
        el.addEventListener("click", () => select(Number(el.dataset.idx)));
      }});
    }}

    function highlightJson(obj, depth = 0, expanded = true) {{
      const pad = "  ".repeat(depth);
      if (obj === null) return `<span class="null">null</span>`;
      const t = typeof obj;
      if (t === "string") return `<span class="str">"${{esc(obj)}}"</span>`;
      if (t === "number") return `<span class="num">${{obj}}</span>`;
      if (t === "boolean") return `<span class="bool">${{obj}}</span>`;
      if (Array.isArray(obj)) {{
        if (!obj.length) return "[]";
        const id = "n" + Math.random().toString(36).slice(2);
        const inner = obj.map((v) => pad + "  " + highlightJson(v, depth + 1, expanded)).join(",\\n");
        return `<span class="collapsible" data-id="${{id}}">${{expanded ? "▼" : "▶"}} [</span>\\n${{expanded ? inner + "\\n" + pad : ""}}<span>] (${{obj.length}})</span>`;
      }}
      if (t === "object") {{
        const keys = Object.keys(obj);
        if (!keys.length) return "{{}}";
        const id = "n" + Math.random().toString(36).slice(2);
        const inner = keys.map((k) => {{
          const v = highlightJson(obj[k], depth + 1, expanded);
          return `${{pad}}  <span class="key">"${{esc(k)}}"</span>: ${{v}}`;
        }}).join(",\\n");
        return `<span class="collapsible" data-id="${{id}}">${{expanded ? "▼" : "▶"}} {{</span>\\n${{expanded ? inner + "\\n" + pad : ""}}<span>}} (${{keys.length}} keys)</span>`;
      }}
      return esc(String(obj));
    }}

    let expandState = true;

    function renderDetail() {{
      const row = DATA[active];
      document.getElementById("jsonOut").innerHTML = highlightJson(row, 0, expandState);
      document.querySelectorAll(".collapsible").forEach((el) => {{
        el.style.cursor = "pointer";
        el.addEventListener("click", (e) => {{
          e.stopPropagation();
          expandState = !expandState;
          renderDetail();
        }});
      }});
    }}

    function select(idx) {{
      active = idx;
      expandState = true;
      renderList();
      renderDetail();
    }}

    document.getElementById("search").addEventListener("input", (e) => {{
      const q = e.target.value.trim().toLowerCase();
      filtered = DATA.map((row, i) => ({{
        i,
        blob: [row.id, row.subset, row.status, row.question, row.observed_action, row.gold_action]
          .filter(Boolean).join(" ").toLowerCase()
      }})).filter((x) => !q || x.blob.includes(q)).map((x) => x.i);
      if (!filtered.includes(active)) active = filtered[0] ?? 0;
      renderList();
      if (filtered.length) renderDetail();
      else document.getElementById("jsonOut").textContent = "无匹配记录";
    }});

    document.getElementById("expandAll").onclick = () => {{ expandState = true; renderDetail(); }};
    document.getElementById("collapseAll").onclick = () => {{ expandState = false; renderDetail(); }};
    document.getElementById("copyJson").onclick = async () => {{
      await navigator.clipboard.writeText(JSON.stringify(DATA[active], null, 2));
    }};

    renderList();
    renderDetail();
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="View results.jsonl in the browser.")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=ROOT / "eval_outputs" / "v2" / "paper_v2" / "full" / "results.jsonl",
        help="Path to results.jsonl (default: eval_outputs/v2/paper_v2/full/results.jsonl)",
    )
    parser.add_argument("--no-open", action="store_true", help="Only write HTML, do not open browser")
    args = parser.parse_args()

    path = args.file.resolve()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    rows = load_jsonl(path)
    out = path.with_suffix(path.suffix + ".viewer.html")
    title = path.name
    html = HTML.format(
        title=title,
        count=len(rows),
        data=json.dumps(rows, ensure_ascii=False),
    )
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} records)")

    if not args.no_open:
        webbrowser.open(out.as_uri())
        print("Opened in browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
