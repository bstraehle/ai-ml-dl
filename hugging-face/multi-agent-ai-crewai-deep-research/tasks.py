from datetime import datetime

from crewai import Task


def get_research_task(agent):
    return Task(
        description=(
            "1. Search the web for content on topic: {topic}.\n"
            "2. Scrape the 10 most relevant web sites for content."
        ),
        expected_output="Content on topic: {topic}.",
        agent=agent,
    )


def get_author_task(agent):
    return Task(
        description=(
            "1. Use the content to write an in-depth article on topic: {topic}.\n"
            "2. At the end of the article, add a references section with links to content provided by the Researcher.\n"
            "3. Include inline references.\n"
            "4. The output must be in markdown format (omit the triple backticks)."
        ),
        expected_output="An article on topic {topic}.",
        agent=agent,
    )


def get_edit_task(agent):
    return Task(
        description=(
            "1. Use the APA Style Guide to edit an article on topic: {topic}.\n"
            "2. Add the current date, " + datetime.today().strftime("%m/%d/%Y") + ", and author, Multi-Agent AI Assistant.\n"
            "3. The output must be in markdown format (omit the triple backticks)."
        ),
        expected_output="An edited article on topic {topic}.",
        agent=agent,
    )
