from crewai import Crew, Process
from langchain_openai import ChatOpenAI

from agents import get_researcher_agent, get_writer_agent
from tasks import get_research_task, get_write_task

def get_crew(model, verbose):
    return Crew(
        agents=[get_researcher_agent(model, verbose), 
                get_writer_agent(model, verbose)],
        tasks=[get_research_task(model, verbose), 
               get_write_task(model, verbose)],
        manager_llm=ChatOpenAI(model=model),
        process=Process.sequential,
        verbose=verbose
    )
