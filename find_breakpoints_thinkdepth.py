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

thinkdepth_file = r"e:\OGE\cobot\Deep_Research-main-v0326\thinkdepth_test.py"

print("=" * 80)
print("THINKDEPTH_TEST.PY BREAKPOINT LOCATIONS")
print("=" * 80)

patterns = {
    "run_full_agent": "async def run_full_agent(",
    "run_simple_agent": "async def run_simple_agent(",
    "interactive_mode": "async def interactive_mode():",
    "astream_events_loop": "async for event in full_agent.astream_events",
    "single_agent_ainvoke": "final_state = await single_agent.ainvoke",
    "build_inputs": "def build_inputs_for_graph(",
    "gemini_chat_once": "def gemini_chat_once(",
    "main_entry": 'if __name__ == "__main__":',
}

results = find_lines(thinkdepth_file, patterns)

print("\n[KEY BREAKPOINT LOCATIONS]")
print("-" * 80)
for name, line_num in sorted(results.items(), key=lambda x: x[1]):
    print(f"  Line {line_num:4d} - {name}")

print("\n" + "=" * 80)
print("RECOMMENDED DEBUG FLOW FOR THINKDEPTH_TEST.PY")
print("=" * 80)
print("""
=== ENTRY POINT ===
Line {main_entry} - if __name__ == "__main__"
  -> Program starts here

Line {interactive_mode} - interactive_mode()
  -> User interaction and mode selection

=== EXECUTION PATHS ===

PATH 1: SIMPLE MODE (Fast Response)
------------------------
Line {run_simple_agent} - run_simple_agent()
  -> Entry point for simple/fast mode
  
Line {build_inputs} - build_inputs_for_graph()
  -> Prepare input state
  
Line {single_agent_ainvoke} - await single_agent.ainvoke(inputs)
  -> *** KEY BREAKPOINT *** Calls single_agent_supervisor graph
  -> After this, execution goes to:
     - single_agent_supervisor.py Line 239 (supervisor node)
     - single_agent_supervisor.py Line 335 (execute_tools_node)
     - single_agent_supervisor.py Line 581 (solve_simple_task)

PATH 2: DEEP RESEARCH MODE (Full Analysis)
------------------------
Line {run_full_agent} - run_full_agent()
  -> Entry point for deep research mode
  
Line {astream_events_loop} - async for event in full_agent.astream_events
  -> *** KEY BREAKPOINT *** Event stream processing
  -> Monitors all graph node executions
  -> After this, execution goes through multiple nodes:
     - clarify_with_user
     - pre_brief_retrieval
     - write_draft_report
     - final_report_generation

=== UTILITY FUNCTIONS ===
Line {gemini_chat_once} - gemini_chat_once()
  -> LLM API call (can set breakpoint to see all prompts)

=== DEBUGGING STRATEGY ===

For Simple Mode:
1. Set breakpoint at Line {run_simple_agent}
2. Set breakpoint at Line {single_agent_ainvoke}
3. Step into -> goes to single_agent_supervisor.py Line 239

For Deep Research Mode:
1. Set breakpoint at Line {run_full_agent}
2. Set breakpoint at Line {astream_events_loop}
3. Watch event['name'] and event['event'] variables in loop

To trace all LLM calls:
- Set breakpoint at Line {gemini_chat_once}
- Inspect user_text and system_instruction parameters
""".format(**results))

print("\n" + "=" * 80)
print("CROSS-FILE EXECUTION FLOW")
print("=" * 80)
print("""
thinkdepth_test.py                    single_agent_supervisor.py
==================                    ==========================

Line {single_agent_ainvoke}  ------>  Line 239: single_supervisor_node
(await single_agent.ainvoke)          (LLM decides route)
                                      |
                                      v
                                      Line 335: execute_tools_node
                                      (Execute DATABASE/CHART/RETRIEVE)
                                      |
                                      v
                                      Line 581: solve_simple_task
                                      (Generate final answer)
                                      |
                                      v
                              Returns to thinkdepth_test.py
                              final_state contains result
""".format(**results))
