import base64
import mimetypes
import re
from datetime import date
from typing import Annotated

import yfinance as yf
from agent_framework import Agent, Content, FunctionInvocationConfiguration, Message, tool
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI
from pydantic import Field

MAX_ROUNDS = 3  # Code Writer -> Code Reviewer rounds before giving up
MAX_TOOL_ITERATIONS = 10
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
# Links to files in the code interpreter container don't resolve outside it; images are shown inline instead
SANDBOX_LINK = re.compile(r"\[[^\]]*\]\(sandbox:[^)]*\)\n?")
VERDICT = re.compile(r"VERDICT:\s*(APPROVED|CHANGES REQUESTED)", re.IGNORECASE)

# Shared by both agents so the Writer meets the Reviewer's checklist up front and the Reviewer judges the work
# against what the setup allows
SETUP = (
    "Setup: the Code Writer runs Python in OpenAI's code interpreter, a sandbox without internet access. "
    "Today's date comes from the today() tool and historical stock data from the yf_download() tool, which "
    "downloads it from Yahoo Finance with yfinance outside the sandbox. The Writer saves that data to a file in the "
    "sandbox and the analysis code reads the file; this is the intended way to use real data, not hard-coding. "
    "Tool calls and their results are shown to the Code Reviewer as evidence of where the data came from."
)
CHECKLIST = (
    "Checklist: 1. The output meets every requirement of the task (data, labels, file names). "
    "2. The code is Pylint-compliant (including a module docstring), readable, and computes results from the "
    "data rather than hard-coding them. 3. Plots are non-empty, correctly labeled, and plausible."
)
WRITER_INSTRUCTIONS = (
    "You are the Code Writer, a Sr. Python Programmer and an expert in writing Pylint-compliant code. "
    f"{SETUP} {CHECKLIST} "
    "Write Python code to solve the task and run it with the code interpreter. Save any figures as PNG files. "
    "Before finishing, check your work against the checklist, since the Code Reviewer will use it. "
    "If the Code Reviewer requests changes, revise the code, run it again, and summarize what you changed."
)
REVIEWER_INSTRUCTIONS = (
    "You are the Code Reviewer. Review the Code Writer's work on the task against the checklist. "
    f"{SETUP} {CHECKLIST} "
    "Request changes only for problems that make the output wrong or fail the task, not for style preferences, "
    "and never for things the setup rules out, such as live downloads inside the sandbox or further proof of data "
    "provenance beyond the tool calls. Mention minor suggestions, but still approve. Be concise. End with "
    "'VERDICT: APPROVED' if the work is acceptable, or 'VERDICT: CHANGES REQUESTED' followed by the changes the "
    "Writer must make."
)


@tool
def today() -> str:
    """Returns today's date."""
    return str(date.today())


@tool
def yf_download(
    tickers: Annotated[list[str], Field(description='Stock tickers, e.g. ["NVDA", "MSFT"].')],
    start_date: Annotated[str, Field(description="Start date as YYYY-MM-DD.")],
    end_date: Annotated[str, Field(description="End date as YYYY-MM-DD (exclusive).")],
) -> str:
    """Returns historical daily stock data as CSV for the given tickers from start date to end date, using yfinance."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    return data.to_csv() if not data.empty else "No data found."


async def render_writer(response, openai: AsyncOpenAI) -> tuple[list[str], list[str], dict]:
    """Return the Writer's Markdown sections, the code and output as plain text for the Reviewer, and the images
    it generated (file name -> bytes)."""
    sections, work, images = [], [], {}
    for message in response.messages:
        for content in message.contents:
            if content.type == "function_call":
                sections.append(f"**Code Writer** called `{content.name}({content.arguments or ''})`")
                work.append(f"Tool call: {content.name}({content.arguments or ''})")
            elif content.type == "function_result":
                result = str(content.result)
                work.append(f"Tool result ({len(result.splitlines())} lines):\n{result[:500]}")
            elif content.type == "code_interpreter_tool_call":
                code = "\n".join(item.text for item in content.inputs or [] if item.text)
                sections.append(f"**Code Writer** (code interpreter)\n```python\n{code}\n```")
                work.append(f"Code:\n```python\n{code}\n```")
            elif content.type == "code_interpreter_tool_result":
                logs = "\n".join(item.text for item in content.outputs or [] if item.type == "text" and item.text)
                if logs:
                    sections.append(f"```\n{logs}\n```")
                    work.append(f"Output:\n```\n{logs}\n```")
            elif content.type == "text" and content.text:
                text = SANDBOX_LINK.sub("", content.text).strip()
                sections.append(f"**Code Writer**\n{text}")
                work.append(f"Summary:\n{text}")
                for annotation in content.annotations or []:
                    file_id, filename = annotation.get("file_id"), annotation.get("url") or ""
                    container_id = (annotation.get("additional_properties") or {}).get("container_id")
                    if file_id and container_id and filename.lower().endswith(IMAGE_EXTENSIONS):
                        file = await openai.containers.files.content.retrieve(file_id, container_id=container_id)
                        images[filename] = file.read()
    return sections, work, images


def markdown_image(filename: str, data: bytes) -> str:
    mime_type = mimetypes.guess_type(filename)[0] or "image/png"
    return f"![{filename}](data:{mime_type};base64,{base64.b64encode(data).decode()})"


async def run_multi_agent(openai_api_key: str, model: str, task: str) -> str:
    """Have the Code Writer write and run code and the Code Reviewer review it, revising until approved, and return
    Markdown with the final images followed by the conversation."""
    openai = AsyncOpenAI(api_key=openai_api_key)
    client = OpenAIChatClient(
        model=model,
        async_client=openai,
        function_invocation_configuration=FunctionInvocationConfiguration(max_iterations=MAX_TOOL_ITERATIONS),
    )
    writer = Agent(
        client=client,
        name="Code Writer",
        instructions=WRITER_INSTRUCTIONS,
        tools=[client.get_code_interpreter_tool(), today, yf_download],
    )
    reviewer = Agent(client=client, name="Code Reviewer", instructions=REVIEWER_INSTRUCTIONS)
    writer_session, reviewer_session = writer.create_session(), reviewer.create_session()

    images, transcript = {}, []
    writer_prompt = task
    for _ in range(MAX_ROUNDS):
        sections, work, round_images = await render_writer(await writer.run(writer_prompt, session=writer_session), openai)
        transcript += sections
        images = round_images or images  # show the latest version of the plots

        review_input = "\n\n".join([f"Task: {task}", "Code Writer's work:", *work])
        plots = [Content.from_data(data, mimetypes.guess_type(name)[0] or "image/png", name=name) for name, data in round_images.items()]
        review = await reviewer.run(Message(role="user", contents=[review_input, *plots]), session=reviewer_session)
        transcript.append(f"**Code Reviewer**\n{review.text.strip()}")

        verdict = VERDICT.findall(review.text or "")
        if verdict and verdict[-1].upper() == "APPROVED":
            break
        writer_prompt = f"The Code Reviewer requested changes:\n{review.text}"
    else:
        transcript.append(f"*Stopped after {MAX_ROUNDS} rounds without the Code Reviewer's approval.*")

    return "\n\n".join([*(markdown_image(name, data) for name, data in images.items()), *transcript])
