from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import scrape_tool, search_tool, today_tool

def get_researcher_agent(model):
    return Agent(
        role="Researcher",
        goal="Research content on topic: {topic}.",
        backstory="You're working on researching content on topic {topic}. "
                  "Your work is the basis for the Writer to write on this topic.",
        llm=ChatOpenAI(model=model, temperature=0.7),
        tools = [search_tool(), scrape_tool()],
        allow_delegation=False,
    	verbose=False
    )

def get_writer_agent(model):
    return Agent(
        role="Writer",
        goal="Write an article on topic: {topic}.",
        backstory="You're working on writing a 2000-word article on topic {topic}. "
                  "You base your writing on the work of the Researcher, who provides context on this topic.",
        llm=ChatOpenAI(model=model, temperature=0.7),
        tools = [today_tool()],
        allow_delegation=False,
        verbose=False
    )
