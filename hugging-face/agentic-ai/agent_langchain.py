from langchain.agents import create_agent
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common import MODEL, REASONING_EFFORT, TIMEOUT, openweathermap_api_key, today

# Each agent step is a model call plus a tool call, so this allows about 25 steps
RECURSION_LIMIT = 50


async def agent_langchain(openai_api_key: str, prompt: str) -> str:
    llm = ChatOpenAI(
        model=MODEL,
        api_key=openai_api_key,
        use_responses_api=True,
        reasoning={"effort": REASONING_EFFORT},
        timeout=TIMEOUT,
    )
    weather = OpenWeatherMapQueryRun(api_wrapper=OpenWeatherMapAPIWrapper(openweathermap_api_key=openweathermap_api_key()))
    agent = create_agent(llm, tools=[weather, tool(today)])
    result = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, {"recursion_limit": RECURSION_LIMIT})
    return result["messages"][-1].text
