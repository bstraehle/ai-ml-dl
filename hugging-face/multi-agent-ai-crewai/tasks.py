from crewai import Task

from agents import get_researcher_agent, get_writer_agent

def get_research_task(model):
    return Task(
        description=(
            "1. Search the web for content on topic: {topic}, prioritizing research papers.\n"
            "2. Scrape the 10 most relevant web sites for content."
        ),
        expected_output="Content on topic: {topic}.",
        agent=get_researcher_agent(model),
    )

def get_write_task(model):
    return Task(
        description=(
            "1. Use the context to write a 2000-word article on topic: {topic}.\n"
            "2. At the beginning of the article, add current date and author: Multi-AI-Agent System.\n"
            "3. At the end of the article, add a references section with research papers."
        ),
        expected_output="An article on topic {topic}.",
        agent=get_writer_agent(model),
    )