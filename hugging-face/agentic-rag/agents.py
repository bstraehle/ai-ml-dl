from crewai import Agent
from langchain_openai import ChatOpenAI
from tools import scrape_tool, search_tool

def get_researcher_agent(model):
    return Agent(
        role="Researcher",
        goal="Research content on topic {topic}",
        backstory="You're working on researching content on topic {topic}. "
                  "Your work is the basis for the Blogger to post on this topic.",
        llm=ChatOpenAI(model=model, temperature=0.7),
        tools = [search_tool(), scrape_tool()],
        allow_delegation=False,
    	verbose=False
    )

def get_blogger_agent(model):
    return Agent(
        role="Blogger",
        goal="Write a {word_count}-word blog post on topic {topic}",
        backstory="You're working on writing a {word_count}-word blog post on topic {topic}. "
                  "You base your writing on the work of the Researcher, who provides context on this topic.",
        llm=ChatOpenAI(model=model, temperature=0.7),
        allow_delegation=False,
        verbose=False
    )
