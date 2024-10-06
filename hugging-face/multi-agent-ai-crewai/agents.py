from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import scrape_tool, search_tool, today_tool

def get_researcher_agent(model, verbose):
    return Agent(
        role="Researcher",
        goal="Research content on topic: {topic}.",
        backstory="You're working on researching content on topic: {topic}. "
                  "Your work is the basis for the Writer to write on this topic.",
        llm=ChatOpenAI(model=model),
        tools = [search_tool(), scrape_tool()],
        allow_delegation=False,
    	verbose=verbose
    )

def get_writer_agent(model, verbose):
    return Agent(
        role="Writer",
        goal="Write an article on topic: {topic}.",
        backstory="You're working on writing an article on topic: {topic}. "
                  "You base your writing on the work of the Researcher, who provides context on this topic.",
        llm=ChatOpenAI(model=model),
        tools = [today_tool()],
        allow_delegation=False,
        verbose=verbose
    )
