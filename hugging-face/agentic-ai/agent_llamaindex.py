import os

from datetime import date
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.tools.weather import OpenWeatherMapToolSpec

def today_tool(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date. 
       The input should always be an empty string, and this function will always return today's date. 
       Any date mathematics should occur outside this function."""
    return str(date.today())

def agent_llamaindex(config, prompt):
    llm = OpenAI(
        model = config["model"],
        temperature = config["temperature"])

    tool_spec = OpenWeatherMapToolSpec(key = os.environ["OPENWEATHERMAP_API_KEY"])
    tools = tool_spec.to_tool_list()
    
    date_tool = FunctionTool.from_defaults(fn = today_tool)
            
    agent = OpenAIAgent.from_tools(
        [tools[0],   # built-in tools
         date_tool], # custom tools
        llm = llm,
        max_iterations = 10,
        max_execution_time = 60,
        verbose = True
    )

    return str(agent.chat(prompt))
