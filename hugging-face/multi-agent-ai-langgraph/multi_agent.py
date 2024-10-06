import functools, operator

from datetime import date

from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

@tool
def today_tool(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date. 
       The input should always be an empty string, and this function will always return today's date. 
       Any date mathematics should occur outside this function."""
    return (str(date.today()) + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.")
    
def create_graph(model, topic):
    tavily_tool = TavilySearchResults(max_results=10)
    
    members = ["Researcher"]
    options = ["FINISH"] + members
   
    system_prompt = (
        "You are a Manager tasked with managing a conversation between the "
        "following agent(s): {members}. Given the following user request, "
        "respond with the agent to act next. Each agent will perform a "
        "task and respond with their results and status. When finished, "
        "respond with FINISH."
    )

    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}.",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    
    llm = ChatOpenAI(model=model, max_tokens=10000, temperature=0)
    
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    researcher_agent = create_agent(llm, [tavily_tool, today_tool], system_prompt=
                                    "You are the chief research scientist at an ivy league university. "
                                    "1. Research content on topic: " + topic + ", prioritizing academic papers. "
                                    "2. Based on your research, write a 5000-word detailed paper on the topic in markdown format. "
                                    "3. At the beginning of the article, add current date and author: Multi-Agent AI System. "
                                    "4. At the end of the article, add a references section with links to academic papers.")
    researcher_node = functools.partial(agent_node, agent=researcher_agent, name="Researcher")

    workflow = StateGraph(AgentState)

    workflow.add_node("Manager", supervisor_chain)
    workflow.add_node("Researcher", researcher_node)

    for member in members:
        workflow.add_edge(member, "Manager")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("Manager", lambda x: x["next"], conditional_map)
    
    workflow.set_entry_point("Manager")
    
    return workflow.compile()

def run_multi_agent(model, topic):
    graph = create_graph(model, topic)
    
    result = graph.invoke({
        "messages": [
            HumanMessage(content=topic)
        ]
    })
    
    article = result['messages'][-1].content
    
    print("===")
    print(article)
    print("===")
    
    return article
