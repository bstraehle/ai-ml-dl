from crewai import Crew, Process

from agents import get_author_agent, get_editor_agent, get_researcher_agent
from tasks import get_author_task, get_edit_task, get_research_task


def get_crew(model, api_key, verbose):
    researcher = get_researcher_agent(model, api_key, verbose)
    author = get_author_agent(model, api_key, verbose)
    editor = get_editor_agent(model, api_key, verbose)

    return Crew(
        agents=[researcher, author, editor],
        tasks=[get_research_task(researcher), get_author_task(author), get_edit_task(editor)],
        process=Process.sequential,
        verbose=verbose,
    )
