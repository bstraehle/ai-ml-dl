# References:

# https://docs.crewai.com/introduction
# https://ai.google.dev/gemini-api/docs

import mimetypes
import os
import re
import time

from crewai.tools import tool
from crewai_tools import StagehandTool
from google import genai
from google.genai import types

from agents.models.llms import (
    LLM_WEB_SEARCH,
    LLM_WEB_BROWSER,
    LLM_IMAGE_ANALYSIS,
    LLM_AUDIO_ANALYSIS,
    LLM_VIDEO_ANALYSIS,
    LLM_YOUTUBE_ANALYSIS,
    LLM_DOCUMENT_ANALYSIS,
    LLM_CODE_GENERATION,
    LLM_CODE_EXECUTION,
    LLM_IMAGE_TO_FEN,
    LLM_ALGEBRAIC_NOTATION,
    LLM_FINAL_ANSWER,
    LLM_FALLBACK,

    THINKING_LEVEL_WEB_SEARCH,
    THINKING_LEVEL_MEDIA_ANALYSIS,
    THINKING_LEVEL_YOUTUBE_ANALYSIS,
    THINKING_LEVEL_DOCUMENT_ANALYSIS,
    THINKING_LEVEL_CODE_GENERATION,
    THINKING_LEVEL_CODE_EXECUTION,
    THINKING_LEVEL_IMAGE_TO_FEN,
    THINKING_LEVEL_ALGEBRAIC_NOTATION,
    THINKING_LEVEL_FINAL_ANSWER
)
from agents.models.prompts import (
    PROMPT_IMG_TO_FEN,
    PROMPT_ALGEBRAIC_NOTATION,
    PROMPT_FINAL_ANSWER
)
from utils.utils import (
    FEN_PATTERN,
    read_docx_text,
    read_pptx_text,
    is_ext
)

FILE_PROCESSING_TIMEOUT = 300


class AITools:
    """Gemini- and Stagehand-backed tools bound to one request's API keys."""

    def __init__(self, gemini_api_key: str, anthropic_api_key: str):
        self._client = genai.Client(api_key=gemini_api_key)
        self._anthropic_api_key = anthropic_api_key

    # Helpers

    @staticmethod
    def _is_rate_limit_error(exception):
        error_str = str(exception)
        return "429" in error_str and "RESOURCE_EXHAUSTED" in error_str

    def _generate(self, tool_name: str, model: str, thinking_level: str, contents, tools=None):
        """Call Gemini, falling back to LLM_FALLBACK once if the daily rate limit is hit."""
        for attempt in range(2):
            try:
                config_params = {"tools": tools} if tools else {}

                if model != LLM_FALLBACK:
                    config_params["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_params)
                )

                print(f"🛠️ AITools: {tool_name}: model={model}")
                if model != LLM_FALLBACK:
                    print(f"🛠️ AITools: {tool_name}: thinking_level={thinking_level}")

                return response
            except Exception as e:
                if attempt == 0 and self._is_rate_limit_error(e):
                    print(f"⚠️ AITools: {tool_name}: Daily rate limit hit with {model}, falling back to {LLM_FALLBACK}")
                    model = LLM_FALLBACK
                    continue
                print(f"⚠️ AITools: {tool_name}: exception={str(e)}")
                raise RuntimeError(f"Processing failed: {str(e)}") from e

    def _upload(self, file_path: str):
        """Upload a file to Gemini and wait until it can be used."""
        file = self._client.files.upload(file=file_path)
        deadline = time.monotonic() + FILE_PROCESSING_TIMEOUT

        while file.state == types.FileState.PROCESSING:
            if time.monotonic() > deadline:
                raise RuntimeError(f"File processing timed out after {FILE_PROCESSING_TIMEOUT}s")
            time.sleep(1)
            file = self._client.files.get(name=file.name)

        if file.state == types.FileState.FAILED:
            raise RuntimeError("File processing failed")

        return file

    @staticmethod
    def _image_part(file_path: str):
        mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
        with open(file_path, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=mime_type)

    @staticmethod
    def _first_text(response):
        return next((part.text for part in response.parts or [] if part.text is not None), None)

    @staticmethod
    def _execution_result(response):
        return next(
            (part.code_execution_result.output for part in response.candidates[0].content.parts
             if part.code_execution_result is not None),
            None
        )

    def _media_analysis(self, tool_name: str, model: str, question: str, file_path: str) -> str:
        print("")
        print(f"🛠️ AITools: {tool_name}: question={question}, file_path={file_path}")

        def generate():
            return self._generate(tool_name, model, THINKING_LEVEL_MEDIA_ANALYSIS, [self._upload(file_path), question])

        result = self._run(tool_name, generate).text
        print(f"🛠️ AITools: {tool_name}: result={result}")
        return result

    @staticmethod
    def _run(tool_name: str, fn):
        """Run fn, turning any failure outside _generate (file I/O, upload) into a RuntimeError."""
        try:
            return fn()
        except RuntimeError:
            raise
        except Exception as e:
            print(f"⚠️ AITools: {tool_name}: exception={str(e)}")
            raise RuntimeError(f"Processing failed: {str(e)}") from e

    # Tools

    def get_tools(self) -> dict:
        """Create the crewAI tools for this request, keyed by function name."""
        ai = self

        @tool("Web Search Tool")
        def web_search_tool(question: str) -> str:
            """Given a question only, search the web to answer the question.

            Args:
                question (str): Question to answer

            Returns:
                str: Answer to the question

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: web_search_tool: question={question}")

            response = ai._generate("web_search_tool", LLM_WEB_SEARCH, THINKING_LEVEL_WEB_SEARCH, question,
                                    tools=[types.Tool(google_search=types.GoogleSearch())])
            result = response.text
            print(f"🛠️ AITools: web_search_tool: result={result}")
            return result

        @tool("Web Browser Tool")
        def web_browser_tool(question: str, url: str) -> str:
            """Given a question and URL, load the URL and act, extract, or observe to answer the question.

            Args:
                question (str): Question about a URL
                url (str): The target URL (must be http/https). "http://"/"https://" will be auto-added if missing.

            Returns:
                str: Answer to the question

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: web_browser_tool: question={question}, url={url}")

            try:
                url_str = url.strip()

                if not url_str.lower().startswith(("http://", "https://")):
                    url_str = f"https://{url_str}"

                with StagehandTool(
                    api_key=os.environ["BROWSERBASE_API_KEY"],
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                    model_api_key=ai._anthropic_api_key,
                    model_name=LLM_WEB_BROWSER,
                    dom_settle_timeout_ms=5000,
                    headless=True,
                    self_heal=True,
                    wait_for_captcha_solves=True,
                    verbose=3
                ) as stagehand_tool:
                    result = stagehand_tool.run(
                        instruction=question,
                        url=url_str,
                        command_type="act" # TODO: act, extract, observe
                    )

                print(f"🛠️ AITools: web_browser_tool: model={LLM_WEB_BROWSER}")
                print(f"🛠️ AITools: web_browser_tool: command_type=act")
                print(f"🛠️ AITools: web_browser_tool: result={result}")

                return result
            except Exception as e:
                print(f"⚠️ AITools: web_browser_tool: exception={str(e)}")
                raise RuntimeError(f"Processing failed: {str(e)}") from e

        @tool("Image Analysis Tool")
        def image_analysis_tool(question: str, file_path: str) -> str:
            """Given a question and image file, analyze the image to answer the question.

            Args:
                question (str): Question about an image file
                file_path (str): The image file path

            Returns:
                str: Answer to the question about the image file

            Raises:
                RuntimeError: If processing fails
            """
            return ai._media_analysis("image_analysis_tool", LLM_IMAGE_ANALYSIS, question, file_path)

        @tool("Audio Analysis Tool")
        def audio_analysis_tool(question: str, file_path: str) -> str:
            """Given a question and audio file, analyze the audio to answer the question.

            Args:
                question (str): Question about an audio file
                file_path (str): The audio file path

            Returns:
                str: Answer to the question about the audio file

            Raises:
                RuntimeError: If processing fails
            """
            return ai._media_analysis("audio_analysis_tool", LLM_AUDIO_ANALYSIS, question, file_path)

        @tool("Video Analysis Tool")
        def video_analysis_tool(question: str, file_path: str) -> str:
            """Given a question and video file, analyze the video to answer the question.

            Args:
                question (str): Question about a video file
                file_path (str): The video file path

            Returns:
                str: Answer to the question about the video file

            Raises:
                RuntimeError: If processing fails
            """
            return ai._media_analysis("video_analysis_tool", LLM_VIDEO_ANALYSIS, question, file_path)

        @tool("YouTube Analysis Tool")
        def youtube_analysis_tool(question: str, url: str) -> str:
            """Given a question and YouTube URL, analyze the video to answer the question.

            Args:
                question (str): Question about a YouTube video
                url (str): The YouTube URL

            Returns:
                str: Answer to the question about the YouTube video

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: youtube_analysis_tool: question={question}, url={url}")

            contents = types.Content(parts=[types.Part(file_data=types.FileData(file_uri=url)), types.Part(text=question)])
            result = ai._generate("youtube_analysis_tool", LLM_YOUTUBE_ANALYSIS, THINKING_LEVEL_YOUTUBE_ANALYSIS, contents).text
            print(f"🛠️ AITools: youtube_analysis_tool: result={result}")
            return result

        @tool("Document Analysis Tool")
        def document_analysis_tool(question: str, file_path: str) -> str:
            """Given a question and document file, analyze the document to answer the question.

            Args:
                question (str): Question about a document file
                file_path (str): The document file path

            Returns:
                str: Answer to the question about the document file

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: document_analysis_tool: question={question}, file_path={file_path}")

            def generate():
                if is_ext(file_path, ".docx") or is_ext(file_path, ".pptx"):
                    text_data = read_docx_text(file_path) if is_ext(file_path, ".docx") else read_pptx_text(file_path)
                    print(f"🛠️ Text data:\n{text_data}")
                    contents = [f"{question}\n{text_data}"]
                else:
                    contents = [ai._upload(file_path), question]
                return ai._generate("document_analysis_tool", LLM_DOCUMENT_ANALYSIS, THINKING_LEVEL_DOCUMENT_ANALYSIS, contents)

            result = ai._run("document_analysis_tool", generate).text
            print(f"🛠️ AITools: document_analysis_tool: result={result}")
            return result

        @tool("Code Generation and Execution Tool")
        def code_generation_and_execution_tool(question: str, json_data: str) -> str:
            """Given a question and JSON data, generate and execute code to answer the question.

            Args:
                question (str): Question to answer
                json_data (str): The JSON data

            Returns:
                str: Answer to the question

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: code_generation_and_execution_tool: question={question}, json_data={json_data}")

            response = ai._generate("code_generation_and_execution_tool", LLM_CODE_GENERATION, THINKING_LEVEL_CODE_GENERATION,
                                    [f"{question}\n{json_data}"], tools=[types.Tool(code_execution=types.ToolCodeExecution())])
            result = ai._execution_result(response)
            print(f"🛠️ AITools: code_generation_and_execution_tool: result={result}")
            return result

        @tool("Code Execution Tool")
        def code_execution_tool(question: str, file_path: str) -> str:
            """Given a question and Python file, execute the file to answer the question.

            Args:
                question (str): Question to answer
                file_path (str): The Python file path

            Returns:
                str: Answer to the question

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: code_execution_tool: question={question}, file_path={file_path}")

            def generate():
                return ai._generate("code_execution_tool", LLM_CODE_EXECUTION, THINKING_LEVEL_CODE_EXECUTION,
                                    [ai._upload(file_path), question], tools=[types.Tool(code_execution=types.ToolCodeExecution())])

            result = ai._execution_result(ai._run("code_execution_tool", generate))
            print(f"🛠️ AITools: code_execution_tool: result={result}")
            return result

        @tool("Image to FEN Tool")
        def img_to_fen_tool(question: str, file_path: str, active_color: str) -> str:
            """Given a chess question, image file, and active color, return the FEN.

            Args:
                question (str): The chess question
                file_path (str): The image file path
                active_color (str): The active color

            Returns:
                str: FEN of the chess position

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: img_to_fen_tool: question={question}, file_path={file_path}, active_color={active_color}")

            def generate():
                prompt = PROMPT_IMG_TO_FEN.format(question=question, active_color=active_color)
                content = types.Content(parts=[types.Part(text=prompt), ai._image_part(file_path)])
                return ai._generate("img_to_fen_tool", LLM_IMAGE_TO_FEN, THINKING_LEVEL_IMAGE_TO_FEN, [content])

            result = ai._first_text(ai._run("img_to_fen_tool", generate)) or ""

            is_match = re.search(FEN_PATTERN, result)

            if is_match:
                result = is_match.group(0)
            else:
                result = next(
                    (line.strip() for line in result.strip().split("\n")
                     if "/" in line and (" w " in line or " b " in line)),
                    result
                )

            print(f"🛠️ AITools: img_to_fen_tool: result={result}")
            return result

        @tool("Algebraic Notation Tool")
        def algebraic_notation_tool(question: str, file_path: str, position_evaluation: str) -> str:
            """Given a chess question, image file, and position evaluation in UCI notation, answer the question in algebraic notation.

            Args:
                question (str): The chess question
                file_path (str): The image file path
                position_evaluation (str): The position evaluation in UCI notation

            Returns:
                str: Answer to the question in algebraic notation

            Raises:
                RuntimeError: If processing fails
            """
            print("")
            print(f"🛠️ AITools: algebraic_notation_tool: question={question}, file_path={file_path}, position_evaluation={position_evaluation}")

            def generate():
                prompt = PROMPT_ALGEBRAIC_NOTATION.format(question=question, position_evaluation=position_evaluation)
                content = types.Content(parts=[types.Part(text=prompt), ai._image_part(file_path)])
                return ai._generate("algebraic_notation_tool", LLM_ALGEBRAIC_NOTATION, THINKING_LEVEL_ALGEBRAIC_NOTATION, [content])

            result = ai._first_text(ai._run("algebraic_notation_tool", generate))
            print(f"🛠️ AITools: algebraic_notation_tool: result={result}")
            return result

        return {
            "web_search_tool": web_search_tool,
            "web_browser_tool": web_browser_tool,
            "image_analysis_tool": image_analysis_tool,
            "audio_analysis_tool": audio_analysis_tool,
            "video_analysis_tool": video_analysis_tool,
            "youtube_analysis_tool": youtube_analysis_tool,
            "document_analysis_tool": document_analysis_tool,
            "code_generation_and_execution_tool": code_generation_and_execution_tool,
            "code_execution_tool": code_execution_tool,
            "img_to_fen_tool": img_to_fen_tool,
            "algebraic_notation_tool": algebraic_notation_tool,
        }

    def final_answer(self, question: str, answer: str) -> str:
        """Given a question and initial answer, get the final answer.

        Args:
            question (str): The question
            answer (str): The initial answer

        Returns:
            str: Final answer

        Raises:
            RuntimeError: If processing fails
        """
        print("")
        print(f"🛠️ AITools: final_answer_tool: question={question}, answer={answer}")

        prompt = PROMPT_FINAL_ANSWER.format(question=question, answer=answer)
        result = self._generate("final_answer_tool", LLM_FINAL_ANSWER, THINKING_LEVEL_FINAL_ANSWER, [prompt]).text.strip()
        print(f"🛠️ AITools: final_answer_tool: result={result}")
        return result
