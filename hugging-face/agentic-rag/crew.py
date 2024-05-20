from crewai import Crew, Process
from langchain_openai import ChatOpenAI

from agents import get_researcher_agent, get_blogger_agent
from tasks import get_research_task, get_blog_task

def get_crew(model):
    return Crew(
        agents=[get_researcher_agent(model), get_blogger_agent(model)],
        tasks=[get_research_task(model), get_blog_task(model)],
        manager_llm=ChatOpenAI(model=model, temperature=0.7),
        process=Process.sequential,
        verbose=True
    )
