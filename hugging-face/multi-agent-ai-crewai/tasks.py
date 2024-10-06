from crewai import Task

from agents import get_researcher_agent, get_writer_agent

def get_research_task(model, verbose):
    return Task(
        description=(
            "1. Search the web for content on topic: {topic}.\n"
            "2. Scrape the 10 most relevant web sites for content."
        ),
        expected_output="Content on topic: {topic}.",
        agent=get_researcher_agent(model, verbose),
    )

def get_write_task(model, verbose):
    return Task(
        description=(
            "1. Use the context to write an in-depth article on topic: {topic}.\n"
            "2. The output must be in markdown format (omit the triple backticks).\n"
            "3. At the beginning of the article, add the current date and author: Multi-Agent AI System.\n"
            "4. Also at the beginning of the article, add a references section with links to relevant content."
        ),
        expected_output="An article on topic {topic}.",
        agent=get_writer_agent(model, verbose),
    )
