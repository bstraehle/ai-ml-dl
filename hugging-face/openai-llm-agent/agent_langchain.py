import os

from datetime import date
from langchain.agents import AgentType, initialize_agent, load_tools, tool
from langchain.chat_models import ChatOpenAI

@tool
def date_tool(text: str) -> str:
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
        tools +      # built-in tools
        [date_tool], # custom tools
        llm,
        agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True,
        verbose = True
    )

    return agent(prompt)
