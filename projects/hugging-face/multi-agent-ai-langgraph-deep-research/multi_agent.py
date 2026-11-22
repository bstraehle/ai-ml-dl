from datetime import datetime
from typing import TypedDict

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

MAX_OUTPUT_TOKENS = 32768
RECURSION_LIMIT = 25  # Researcher model and tool steps


class ResearchState(TypedDict):
    topic: str
    research: str
    draft: str
    article: str


def system_prompt(role: str, goal: str, backstory: str) -> str:
    return f"You are the {role}. {backstory}\nYour goal: {goal}"


def create_graph(openai_api_key: str, model: str, topic: str):
    def llm(**kwargs) -> ChatOpenAI:
        # GPT-6.2 models only support tool calling via the Responses API
        return ChatOpenAI(model=model, api_key=openai_api_key, use_responses_api=True, **kwargs)

    researcher = create_agent(
        llm(),
        tools=[{"type": "web_search"}],
        system_prompt=system_prompt(
            "Researcher",
            f"Research content on topic: {topic}.",
            f"You're working on researching content on topic: {topic}. Your work is the basis for the Author to write on this topic.",
        ),
    )
    author = llm(max_tokens=MAX_OUTPUT_TOKENS)
    editor = llm(max_tokens=MAX_OUTPUT_TOKENS)

    async def research(state: ResearchState) -> dict:
        task = (
            f"1. Search the web for content on topic: {topic}.\n"
            "2. Use the 10 most relevant web sites for content, and include the URL of each source."
        )
        result = await researcher.ainvoke({"messages": [{"role": "user", "content": task}]}, {"recursion_limit": RECURSION_LIMIT})
        return {"research": required(result["messages"][-1].text, "Researcher")}

    async def write(state: ResearchState) -> dict:
        messages = [
            ("system", system_prompt(
                "Author",
                f"Write an article on topic: {topic}.",
                f"You're working on writing an article on topic: {topic}. "
                "You base your writing on the work of the Researcher, who provides content on this topic. "
                "Your work is the basis for the Editor to edit the article on this topic.",
            )),
            ("user",
             f"1. Use the content to write an in-depth article on topic: {topic}.\n"
             "2. At the end of the article, add a references section with links to content provided by the Researcher.\n"
             "3. Include inline references.\n"
             "4. The output must be in markdown format (omit the triple backticks).\n\n"
             f"Content from the Researcher:\n{state['research']}"),
        ]
        return {"draft": required((await author.ainvoke(messages)).text, "Author")}

    async def edit(state: ResearchState) -> dict:
        messages = [
            ("system", system_prompt(
                "Editor",
                f"Edit an article on topic: {topic}.",
                f"You're working on editing an article on topic: {topic}. "
                "You base your editing on the work of the Author, who provides an article on this topic.",
            )),
            ("user",
             f"1. Use the APA Style Guide to edit an article on topic: {topic}.\n"
             f"2. Add the current date, {datetime.today().strftime('%m/%d/%Y')}, and author, Multi-Agent AI Assistant.\n"
             "3. The output must be in markdown format (omit the triple backticks).\n\n"
             f"Article from the Author:\n{state['draft']}"),
        ]
        return {"article": required((await editor.ainvoke(messages)).text, "Editor")}

    graph = StateGraph(ResearchState)
    graph.add_node("researcher", research)
    graph.add_node("author", write)
    graph.add_node("editor", edit)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "author")
    graph.add_edge("author", "editor")
    graph.add_edge("editor", END)
    return graph.compile()


def required(text: str, agent: str) -> str:
    if not text:
        raise RuntimeError(f"The {agent} returned no text.")
    return text


async def run_multi_agent(openai_api_key: str, model: str, topic: str) -> str:
    """Research the topic on the web, write an article, and edit it, returning the edited Markdown article."""
    result = await create_graph(openai_api_key, model, topic).ainvoke({"topic": topic})
    return result["article"]
