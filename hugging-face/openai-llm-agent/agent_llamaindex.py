import os

from datetime import date
from llama_hub.tools.weather import OpenWeatherMapToolSpec
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool

def date_tool(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date. 
       The input should always be an empty string, and this function will always return today's date. 
       Any date mathematics should occur outside this function."""
    return str(date.today())

def agent_llamaindex(model, temperature, prompt):
    llm = OpenAI(
        model = model,
        temperature = temperature)

    tool_spec = OpenWeatherMapToolSpec()
    tools = tool_spec.to_tool_list()
    
    dt_tool = FunctionTool.from_defaults(fn = date_tool)
            
    agent = OpenAIAgent.from_tools(
        [tools[0], # built-in tools
         dt_tool], # custom tools
        llm = llm, 
        verbose = True)

    return agent.chat(prompt)
