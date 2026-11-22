from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAIResponses
from llama_index.tools.weather import OpenWeatherMapToolSpec

from common import MODEL, REASONING_EFFORT, TIMEOUT, openweathermap_api_key, today

MAX_ITERATIONS = 10


async def agent_llamaindex(openai_api_key: str, prompt: str) -> str:
    llm = OpenAIResponses(
        model=MODEL,
        api_key=openai_api_key,
        reasoning_options={"effort": REASONING_EFFORT},
        timeout=TIMEOUT,
    )
    weather = OpenWeatherMapToolSpec(key=openweathermap_api_key()).to_tool_list(spec_functions=["weather_at_location"])
    agent = FunctionAgent(tools=[*weather, FunctionTool.from_defaults(fn=today)], llm=llm, streaming=False)
    response = await agent.run(user_msg=prompt, max_iterations=MAX_ITERATIONS)
    return str(response)
