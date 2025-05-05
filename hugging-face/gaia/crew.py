# References:

# https://docs.crewai.com/introduction
# https://ai.google.dev/gemini-api/docs

import os
import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from google import genai
from google.genai import types
from openinference.instrumentation.crewai import CrewAIInstrumentor
from phoenix.otel import register
from util import read_file, get_final_answer

## LLMs

MANAGER_MODEL           = "gpt-4.5-preview"
AGENT_MODEL             = "gpt-4.1-mini"

FINAL_ANSWER_MODEL      = "gemini-2.5-pro-preview-03-25"

WEB_SEARCH_MODEL        = "gemini-2.5-flash-preview-04-17"
IMAGE_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
AUDIO_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
VIDEO_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
YOUTUBE_ANALYSIS_MODEL  = "gemini-2.5-flash-preview-04-17"
DOCUMENT_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
CODE_GENERATION_MODEL   = "gemini-2.5-flash-preview-04-17"
CODE_EXECUTION_MODEL    = "gemini-2.5-flash-preview-04-17"

# LLM evaluation

PHOENIX_API_KEY = os.environ["PHOENIX_API_KEY"]

os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer_provider = register(
    auto_instrument=True,
    project_name="gaia"
)

CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

def run_crew(question, file_path):
    # Tools

    @tool("Web Search Tool")
    def web_search_tool(question: str) -> str:
        """Given a question only, search the web to answer the question.
    
           Args:
               question (str): Question to answer
                
           Returns:
               str: Answer to the question
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            
            response = client.models.generate_content(
                model=WEB_SEARCH_MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearchRetrieval())]
                )
            )

            return response.text
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
    
    @tool("Image Analysis Tool")
    def image_analysis_tool(question: str, file_path: str) -> str:
        """Given a question and image file, analyze the image to answer the question.
    
           Args:
               question (str): Question about an image file
               file_path (str): The image file path
                
           Returns:
               str: Answer to the question about the image file
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            
            file = client.files.upload(file=file_path)

            response = client.models.generate_content(
                model=IMAGE_ANALYSIS_MODEL,
                contents=[file, question]
            )
          
            return response.text
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")

    @tool("Audio Analysis Tool")
    def audio_analysis_tool(question: str, file_path: str) -> str:
        """Given a question and audio file, analyze the audio to answer the question.
    
           Args:
               question (str): Question about an audio file
               file_path (str): The audio file path
                
           Returns:
               str: Answer to the question about the audio file
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            file = client.files.upload(file=file_path)
            
            response = client.models.generate_content(
                model=AUDIO_ANALYSIS_MODEL, 
                contents=[file, question]
            )
          
            return response.text
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")

    @tool("Video Analysis Tool")
    def video_analysis_tool(question: str, file_path: str) -> str:
        """Given a question and video file, analyze the video to answer the question.
    
           Args:
               question (str): Question about a video file
               file_path (str): The video file path
                
           Returns:
               str: Answer to the question about the video file
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            file = client.files.upload(file=file_path)
            
            response = client.models.generate_content(
                model=VIDEO_ANALYSIS_MODEL, 
                contents=[file, question]
            )

            return response.text
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")

    @tool("Document Analysis Tool")
    def document_analysis_tool(question: str, file_path: str) -> str:
        """Given a question and document file, analyze the document to answer the question.
    
           Args:
               question (str): Question about a document file
               file_path (str): The document file path
                
           Returns:
               str: Answer to the question about the document file
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            
            file = client.files.upload(file=file_path)

            response = client.models.generate_content(
                model=DOCUMENT_ANALYSIS_MODEL,
                contents=[file, question]
            )
          
            return response.text
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
            
    @tool("YouTube Analysis Tool")
    def youtube_analysis_tool(question: str, url: str) -> str:
        """Given a question and YouTube URL, analyze the video to answer the question.
    
           Args:
               question (str): Question about a YouTube video
               url (str): The YouTube video URL
                
           Returns:
               str: Answer to the question about the YouTube video
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            return client.models.generate_content(
                model=YOUTUBE_ANALYSIS_MODEL,
                contents=types.Content(
                    parts=[types.Part(file_data=types.FileData(file_uri=url)),
                           types.Part(text=question)]
                )
            )
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
    
    @tool("Code Generation Tool")
    def code_generation_tool(question: str, file_path: str) -> str:
        """Given a question and data file, generate and execute code to answer the question.

           Args:
               question (str): Question to answer
                file_path (str): The data file path

           Returns:
               str: Answer to the question
               
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            file_data = read_file(file_path)
            
            response = client.models.generate_content(
                model=CODE_GENERATION_MODEL,
                contents=[f"{question}\n{file_data}"],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                ),
            )
            
            for part in response.candidates[0].content.parts:
                if part.code_execution_result is not None:
                    return part.code_execution_result.output
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
            
    @tool("Code Execution Tool")
    def code_execution_tool(question: str, file_path: str) -> str:
        """Given a question and Python file, execute the file to answer the question.
    
           Args:
               question (str): Question to answer
               file_path (str): The Python file path
                
           Returns:
               str: Answer to the question
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

            file = client.files.upload(file=file_path)

            response = client.models.generate_content(
                model=CODE_EXECUTION_MODEL,
                contents=[file, question],
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution)]
                ),
            )
            
            for part in response.candidates[0].content.parts:
                if part.code_execution_result is not None:
                    return part.code_execution_result.output
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")
            
    # Agents

    web_search_agent = Agent(
        role="Web Search Agent",
        goal="Given a question only, search the web and answer the question: {question}",
        backstory="As an expert web search assistant, you search the web to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[web_search_tool],
        verbose=True
    )

    image_analysis_agent = Agent(
        role="Image Analysis Agent",
        goal="Given a question and image file, analyze the image and answer the question: {question}",
        backstory="As an expert image analysis assistant, you analyze the image to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[image_analysis_tool],
        verbose=True
    )

    audio_analysis_agent = Agent(
        role="Audio Analysis Agent",
        goal="Given a question and audio file, analyze the audio and answer the question: {question}",
        backstory="As an expert audio analysis assistant, you analyze the audio to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[audio_analysis_tool],
        verbose=True
    )

    video_analysis_agent = Agent(
        role="Video Analysis Agent",
        goal="Given a question and video file, analyze the video and answer the question: {question}",
        backstory="As an expert video analysis assistant, you analyze the video file to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[video_analysis_tool],
        verbose=True
    )
    
    youtube_analysis_agent = Agent(
        role="YouTube Analysis Agent",
        goal="Given a question and YouTube URL, analyze the video and answer the question: {question}",
        backstory="As an expert YouTube analysis assistant, you analyze the video to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[youtube_analysis_tool],
        verbose=True
    )

    document_analysis_agent = Agent(
        role="Document Analysis Agent",
        goal="Given a question and document file, analyze the document and answer the question: {question}",
        backstory="As an expert document analysis assistant, you analyze the document to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[document_analysis_tool],
        verbose=True
    )
    
    code_generation_agent = Agent(
        role="Code Generation Agent",
        goal="Given a question and data file, generate and execute code to answer the question: {question}",
        backstory="As an expert Python code generation assistant, you generate and execute code to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=3,
        tools=[code_generation_tool],
        verbose=True
    )
    
    code_execution_agent = Agent(
        role="Code Execution Agent",
        goal="Given a question and Python file, execute the file to answer the question: {question}",
        backstory="As an expert Python code execution assistant, you execute code to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=3,
        tools=[code_execution_tool],
        verbose=True
    )

    manager_agent = Agent(
        role="Manager Agent",
        goal="Answer the following question. If needed, delegate to one of your coworkers. Question: {question}",
        backstory="As an expert manager assistant, you answer the question.",
        allow_delegation=True,
        llm=MANAGER_MODEL,
        max_iter=5,
        verbose=True
    )

    # Task

    manager_task = Task(
        agent=manager_agent,
        description="Answer the following question. If needed, delegate to one of your coworkers:\n"
                    "- Web Search Agent requires a question only.\n"
                    "- Image Analysis Agent requires a question and **.png, .jpeg, .webp, .heic, or .heif image file**.\n"
                    "- Audio Analysis Agent requires a question and **.wav, .mp3, .aiff, .aac, .ogg, or .flac audio file**.\n"
                    "- Video Analysis Agent requires a question and **.mp4, .mpeg, .mov, .avi, .x-flv, .mpg, .webm, .wmv, or .3gpp video file**.\n"
                    "- Document Analysis Agent requires a question and **.pdf, .txt, .html, css, .js, .md, .xml, or .rtf document file**.\n"
                    "- YouTube Analysis Agent requires a question and **YouTube URL**.\n"
                    "- Code Generation Agent requires a question and **.csv, .xls, .xlsx, .json, or .jsonl data file**.\n"
                    "- Code Execution Agent requires a question and **.py Python file**.\n"
                    "Question: {question}",
        expected_output="The answer to the question."
    )
    
    # Crew
    
    crew = Crew(
        agents=[web_search_agent, 
                image_analysis_agent, 
                audio_analysis_agent, 
                video_analysis_agent, 
                youtube_analysis_agent, 
                document_analysis_agent, 
                code_generation_agent, 
                code_execution_agent],
        manager_agent=manager_agent,
        tasks=[manager_task],
        verbose=True
    )

    # Process

    if file_path:
        question = f"{question} File path: {file_path}."

    initial_answer = crew.kickoff(inputs={"question": question})
    final_answer = get_final_answer(FINAL_ANSWER_MODEL, question, str(initial_answer))

    print("###")
    print(f"Question: {question}")
    print(f"Initial answer: {initial_answer}")
    print(f"Final answer: {final_answer}")
    print("###")
    
    return final_answer
