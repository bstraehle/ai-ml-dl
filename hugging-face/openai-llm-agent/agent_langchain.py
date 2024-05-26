import os

from datetime import date
from langchain.agents import AgentType, initialize_agent, load_tools, tool
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]    = "openai-llm-agent"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

@tool
def today_tool(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date. 
       The input should always be an empty string, and this function will always return today's date. 
       Any date mathematics should occur outside this function."""
    return str(date.today())

def agent_langchain(config, prompt):
    llm = ChatOpenAI(
        model_name = config["model"],
        temperature = config["temperature"])

    OPENWEATHERMAP_API_KEY = os.environ["OPENWEATHERMAP_API_KEY"]
    
    tools = load_tools(["openweathermap-api"])

    agent = initialize_agent(
        tools +       # built-in tools
        [today_tool], # custom tools
        llm,
        agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True,
        max_iterations = 10,
        max_execution_time = 60,
        verbose = True
    )

    with get_openai_callback() as callback:
        completion = agent(prompt)

    return completion, callback
