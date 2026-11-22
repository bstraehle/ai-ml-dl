# References:

# https://docs.crewai.com/introduction

from pathlib import Path

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from agents.models.llms import (
    CONTEXT_WINDOW_SIZE,
    LLM_CREW_PLANNING,
    LLM_MANAGER,
    LLM_AGENT
)
from agents.tools.ai_tools import AITools
from agents.tools.deterministic_tools import DeterministicTools
from agents.tools.mcp_tools import MCPTools
from utils.utils import read_file_json, is_ext

# Configuration

PLANNING_CREW                  = True
MEMORY_CREW                    = False
VERBOSE_CREW                   = False

MAX_ITER_MANAGER               = 15
REASONING_MANAGER              = True
MAX_REASONING_ATTEMPTS_MANAGER = 3
VERBOSE_MANAGER                = False

MAX_ITER_AGENT                 = 15
REASONING_AGENT                = True
MAX_REASONING_ATTEMPTS_AGENT   = 2
VERBOSE_AGENT                  = False

CONFIG_DIR = Path(__file__).parent / "config"
AGENTS_CONFIG = yaml.safe_load((CONFIG_DIR / "agents.yaml").read_text(encoding="utf-8"))
TASKS_CONFIG = yaml.safe_load((CONFIG_DIR / "tasks.yaml").read_text(encoding="utf-8"))

# Coworker agents (config name in agents.yaml) and their tools, in delegation order
AGENT_TOOLS = {
    "web_search_agent":                    ["web_search_tool"],
    "web_browser_agent":                   ["web_browser_tool"],
    "chess_analysis_agent":                ["img_to_fen_tool", "best_move_tool", "algebraic_notation_tool"],
    "image_analysis_agent":                ["image_analysis_tool"],
    "audio_analysis_agent":                ["audio_analysis_tool"],
    "video_analysis_agent":                ["video_analysis_tool"],
    "youtube_analysis_agent":              ["youtube_analysis_tool"],
    "document_analysis_agent":             ["document_analysis_tool"],
    "arithmetic_agent":                    ["add_tool", "subtract_tool", "multiply_tool", "divide_tool", "modulus_tool"],
    "code_generation_and_execution_agent": ["code_generation_and_execution_tool"],
    "code_execution_agent":                ["code_execution_tool"],
}

DETERMINISTIC_TOOLS = {
    "add_tool":      DeterministicTools.add_tool,
    "subtract_tool": DeterministicTools.subtract_tool,
    "multiply_tool": DeterministicTools.multiply_tool,
    "divide_tool":   DeterministicTools.divide_tool,
    "modulus_tool":  DeterministicTools.modulus_tool,
}


def _get_llm(model: str, api_key: str) -> LLM:
    # GPT-6.2 models only support tool calling via the Responses API
    return LLM(model=model, api_key=api_key, api="responses", context_window_size=CONTEXT_WINDOW_SIZE)


def _print_agent_config(name: str, llm: str, reasoning: bool, max_reasoning_attempts: int, max_iter: int, tools=None):
    print(f"🤖 {name}: llm={llm}, reasoning={reasoning} (max attempts {max_reasoning_attempts}), max_iter={max_iter}"
          + (f", tools={', '.join(tools)}" if tools else ""))


def get_crew(openai_api_key: str, ai_tools: AITools) -> Crew:
    tools = ai_tools.get_tools() | DETERMINISTIC_TOOLS | {"best_move_tool": MCPTools.best_move_tool}

    print("")
    agents = []
    for name, tool_names in AGENT_TOOLS.items():
        _print_agent_config(name, LLM_AGENT, REASONING_AGENT, MAX_REASONING_ATTEMPTS_AGENT, MAX_ITER_AGENT, tool_names)
        agents.append(Agent(
            config=AGENTS_CONFIG[name],
            allow_delegation=False,
            llm=_get_llm(LLM_AGENT, openai_api_key),
            max_iter=MAX_ITER_AGENT,
            tools=[tools[t] for t in tool_names],
            reasoning=REASONING_AGENT,
            max_reasoning_attempts=MAX_REASONING_ATTEMPTS_AGENT,
            verbose=VERBOSE_AGENT
        ))

    _print_agent_config("manager_agent", LLM_MANAGER, REASONING_MANAGER, MAX_REASONING_ATTEMPTS_MANAGER, MAX_ITER_MANAGER)
    manager = Agent(
        config=AGENTS_CONFIG["manager_agent"],
        allow_delegation=True,
        llm=_get_llm(LLM_MANAGER, openai_api_key),
        max_iter=MAX_ITER_MANAGER,
        reasoning=REASONING_MANAGER,
        max_reasoning_attempts=MAX_REASONING_ATTEMPTS_MANAGER,
        verbose=VERBOSE_MANAGER
    )

    manager_task_config = {k: v for k, v in TASKS_CONFIG["manager_task"].items() if k != "agent"}

    print(f"🤖 Crew: planning={PLANNING_CREW} (llm={LLM_CREW_PLANNING}), memory={MEMORY_CREW}, verbose={VERBOSE_CREW}")

    return Crew(
        agents=[*agents, manager],
        tasks=[Task(config=manager_task_config, agent=manager)],
        process=Process.sequential,
        planning_llm=_get_llm(LLM_CREW_PLANNING, openai_api_key),
        planning=PLANNING_CREW,
        memory=MEMORY_CREW,
        verbose=VERBOSE_CREW
    )


def _get_final_question(question, file_path):
    if not file_path:
        return question

    if any(is_ext(file_path, ext) for ext in (".csv", ".xls", ".xlsx", ".json", ".jsonl")):
        return f"{question} JSON data:\n{read_file_json(file_path)}."

    return f"{question} File path: {file_path}."


def run_crew(question, file_path, openai_api_key, gemini_api_key, anthropic_api_key):
    print("")
    print(f"🤖 Question: {question}")

    ai_tools = AITools(gemini_api_key, anthropic_api_key)
    final_question = _get_final_question(question, file_path)

    print("")
    print(f"🤖 Crew execution started")

    answer = get_crew(openai_api_key, ai_tools).kickoff(inputs={"question": final_question})

    print("")
    print(f"🤖 Crew execution completed")

    final_answer = ai_tools.final_answer(question, str(answer))

    print("")
    print(f"🤖 Answer: {final_answer}")

    return final_answer
