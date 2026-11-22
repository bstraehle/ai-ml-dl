from crewai import LLM, Agent
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

MAX_COMPLETION_TOKENS = 32768
# crewAI assumes 8,192 tokens for models it doesn't know (like gpt-6.2-sol), which
# makes it defer every scraped page and endlessly compact the conversation.
# 272K is OpenAI's standard-price input limit; above it input is billed at 2x.
CONTEXT_WINDOW_SIZE = 272_000


def get_llm(model, api_key, **kwargs):
    # GPT-6.2 models only support tool calling via the Responses API
    return LLM(model=model, api_key=api_key, api="responses", context_window_size=CONTEXT_WINDOW_SIZE, **kwargs)


def get_researcher_agent(model, api_key, verbose):
    return Agent(
        role="Researcher",
        goal="Research content on topic: {topic}.",
        backstory="You're working on researching content on topic: {topic}. "
                  "Your work is the basis for the Writer to write on this topic.",
        llm=get_llm(model, api_key),
        tools=[SerperDevTool(), ScrapeWebsiteTool()],
        allow_delegation=False,
        verbose=verbose,
    )


def get_author_agent(model, api_key, verbose):
    return Agent(
        role="Author",
        goal="Write an article on topic: {topic}.",
        backstory="You're working on writing an article on topic: {topic}. "
                  "You base your writing on the work of the Researcher, who provides content on this topic. "
                  "Your work is the basis for the Editor to edit the article on this topic.",
        llm=get_llm(model, api_key, max_completion_tokens=MAX_COMPLETION_TOKENS),
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )


def get_editor_agent(model, api_key, verbose):
    return Agent(
        role="Editor",
        goal="Edit an article on topic: {topic}.",
        backstory="You're working on editing an article on topic: {topic}. "
                  "You base your editing on the work of the Author, who provides an article on this topic.",
        llm=get_llm(model, api_key, max_completion_tokens=MAX_COMPLETION_TOKENS),
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )
