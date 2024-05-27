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

LLM = "gpt-4o"

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
    
def create_graph(topic):
    tavily_tool = TavilySearchResults(max_results=10)
    
    members = ["Researcher", "Writer"]
    
    system_prompt = (
        "You are a manager tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    options = ["FINISH"] + members

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
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    
    llm = ChatOpenAI(model=LLM, max_tokens=4096)
    
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    researcher_agent = create_agent(llm, [tavily_tool], system_prompt=f"Prioritizing research papers, research content on topic: {topic}.")
    researcher_node = functools.partial(agent_node, agent=researcher_agent, name="Researcher")

    writer_agent = create_agent(llm, [today_tool], system_prompt=f"Write a 1000-word article on topic: {topic}, including a reference section with research papers. At the top, add current date and author: Multi-AI-Agent System based on GPT-4o.")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer")

    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Writer", writer_node)
    workflow.add_node("Manager", supervisor_chain)

    for member in members:
        workflow.add_edge(member, "Manager")

    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges("Manager", lambda x: x["next"], conditional_map)
    workflow.set_entry_point("Manager")
    
    return workflow.compile()

def run_multi_agent(topic):
    graph = create_graph(topic)
    result = graph.invoke({
        "messages": [
            HumanMessage(content=topic)
        ]
    })
    return result['messages'][-1].content
