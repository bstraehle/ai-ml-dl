from crewai import Task

from agents import get_researcher_agent, get_blogger_agent

def get_research_task(model):
    return Task(
        description=(
            "1. Search the web for content on topic {topic}.\n"
            "2. Scrape the 10 most relevant web sites for content, prioritizing research papers."
        ),
        expected_output="Content on topic {topic}.",
        agent=get_researcher_agent(model),
    )

def get_blog_task(model):
    return Task(
        description=(
            "1. Use the context to write a {word_count}-word blog post on topic {topic}.\n"
            "2. If applicable, add a references section with research papers."
        ),
        expected_output="A {word_count}-word blog post on topic {topic}.",
        agent=get_blogger_agent(model),
    )
