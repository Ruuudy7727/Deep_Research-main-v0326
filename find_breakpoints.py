import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def find_lines(filepath, patterns):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = {}
        for i, line in enumerate(lines, 1):
            for name, pattern in patterns.items():
                if pattern in line and name not in results:
                    results[name] = i
        return results

supervisor_file = r"e:\OGE\cobot\Deep_Research-main-v0326\deep_research\single_agent_supervisor.py"
server_file = r"e:\OGE\cobot\Deep_Research-main-v0326\server_plus.py"

print("=" * 80)
print("BREAKPOINT LOCATIONS")
print("=" * 80)

print("\n[FILE: single_agent_supervisor.py]")
sup_patterns = {
    "single_supervisor_node": "async def single_supervisor_node(state: AgentState):",
    "execute_tools_node": "async def execute_tools_node(state: AgentState):",
    "solve_simple_task": "async def solve_simple_task(state: AgentState",
    "DATABASE_branch": 'if route == "DATABASE":',
    "CHART_branch": 'elif route == "CHART":',
    "RETRIEVE_branch": 'elif route == "RETRIEVE":',
}
sup_results = find_lines(supervisor_file, sup_patterns)
for name, line_num in sorted(sup_results.items(), key=lambda x: x[1]):
    print(f"  Line {line_num:4d} - {name}")

print("\n[FILE: server_plus.py]")
srv_patterns = {
    "api_chat": '@app.post("/api/chat")',
    "background_graph_runner": "async def background_graph_runner(",
    "astream_events": "async for event in graph.astream_events",
}
srv_results = find_lines(server_file, srv_patterns)
for name, line_num in sorted(srv_results.items(), key=lambda x: x[1]):
    print(f"  Line {line_num:4d} - {name}")

print("\n" + "=" * 80)
print("RECOMMENDED DEBUG FLOW")
print("=" * 80)
print("""
Step 1: server_plus.py Line {api_chat}
  -> User request enters API

Step 2: server_plus.py Line {background_graph_runner}
  -> Agent execution starts

Step 3: single_agent_supervisor.py Line {single_supervisor_node}
  -> LLM decides route (DATABASE/CHART/RETRIEVE/DIRECT)

Step 4: single_agent_supervisor.py Line {execute_tools_node}
  -> Tool execution dispatcher

Step 5: Branch breakpoints based on route:
  - Line {DATABASE_branch} -> Database query
  - Line {CHART_branch} -> Chart generation
  - Line {RETRIEVE_branch} -> Knowledge base search

Step 6: single_agent_supervisor.py Line {solve_simple_task}
  -> Final answer generation with streaming

Step 7: server_plus.py Line {astream_events}
  -> LangGraph event stream processing
""".format(**{**sup_results, **srv_results}))
