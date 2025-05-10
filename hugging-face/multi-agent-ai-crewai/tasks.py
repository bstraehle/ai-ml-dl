from crewai import Task

from agents import get_researcher_agent, get_writer_agent

def get_research_task(model, verbose):
    return Task(
        description=(
            "1. Search the web for content on topic: {topic}.\n"
            "2. Scrape the 25 most relevant web sites for content."
        ),
        expected_output="Content on topic: {topic}.",
        agent=get_researcher_agent(model, verbose),
    )

def get_write_task(model, verbose):
    return Task(
        description=(
            "1. Use the context to write a long and in-depth article on topic: {topic}.\n"
            "2. The output must be in markdown format (omit the triple backticks), including lists.\n"
            "3. At the beginning of the article below the title, add the current date and author: Multi-Agent AI System.\n"
            "4. At the end of the article, add a references section with distinct links to content provided by the Researcher.\n"
            "5. Include inline references to the links, for example: [1]."
        ),
        expected_output="An article on topic {topic}.",
        agent=get_writer_agent(model, verbose),
    )
