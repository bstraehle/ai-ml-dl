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
from tools import add, subtract, multiply, divide, modulus
from utils import read_file_json, read_docx_text, read_pptx_text, is_ext

## LLMs

MANAGER_MODEL           = "gpt-4.5-preview"
AGENT_MODEL             = "gpt-4.1-mini"

FINAL_ANSWER_MODEL      = "gemini-2.5-pro-preview-03-25"

WEB_SEARCH_MODEL        = "gemini-2.5-flash-preview-04-17"
IMAGE_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
AUDIO_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
VIDEO_ANALYSIS_MODEL    = "gemini-2.5-flash-preview-04-17"
YOUTUBE_ANALYSIS_MODEL  = "gemini-2.5-flash-preview-04-17"
DOCUMENT_ANALYSIS_MODEL = "gemini-2.5-flash-preview-04-17"
ARITHMETIC_MODEL        = "gemini-2.5-flash-preview-04-17"
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

#CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

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

            contents = []
            
            if is_ext(file_path, ".docx"):
                text_data = read_docx_text(file_path)
                contents = [f"{question}\n{text_data}"]
                print(f"=> Text data:\n{text_data}")
            elif is_ext(file_path, ".pptx"):
                text_data = read_pptx_text(file_path)
                contents = [f"{question}\n{text_data}"]
                print(f"=> Text data:\n{text_data}")
            else:
                file = client.files.upload(file=file_path)
                contents = [file, question]
            
            response = client.models.generate_content(
                model=DOCUMENT_ANALYSIS_MODEL,
                contents=contents
            )
          
            return response.text
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")

    @tool("Arithmetic Tool")
    def arithmetic_tool(question: str, a: float, b: float) -> float:
        """Given a question and two numbers, perform the calculation to answer the question.
    
           Args:
               question (str): Question to answer
               a (float): First number
               b (float): Second number
                
           Returns:
               float: Result number
                
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            
            response = client.models.generate_content(
                model=ARITHMETIC_MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[add, subtract, multiply, divide, modulus]
                )
            )

            return response.text
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

    @tool("Code Generation Tool")
    def code_generation_tool(question: str, json_data: str) -> str:
        """Given a question and JSON data, generate and execute code to answer the question.

           Args:
               question (str): Question to answer
                file_path (str): The JSON data

           Returns:
               str: Answer to the question
               
           Raises:
               RuntimeError: If processing fails"""
        try:
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
                    
            response = client.models.generate_content(
                model=CODE_GENERATION_MODEL,
                contents=[f"{question}\n{json_data}"],
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

    arithmetic_agent = Agent(
        role="Arithmetic Agent",
        goal="Given a question and two numbers, perform the calculation and answer the question: {question}",
        backstory="As an expert arithmetic assistant, you perform the calculation to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=2,
        tools=[arithmetic_tool],
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

    code_generation_agent = Agent(
        role="Code Generation Agent",
        goal="Given a question and JSON data, generate and execute code to answer the question: {question}",
        backstory="As an expert Python code generation assistant, you generate and execute code to answer the question.",
        allow_delegation=False,
        llm=AGENT_MODEL,
        max_iter=3,
        tools=[code_generation_tool],
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
                    "- YouTube Analysis Agent requires a question and **YouTube URL**.\n"
                    "- Document Analysis Agent requires a question and **.docx, .pptx, .pdf, .txt, .html, css, .js, .md, .xml, or .rtf document file**.\n"
                    "- Arithmetic Agent requires a question and **two numbers to add, subtract, multiply, divide, or get the modulus**. "
                    "  In case there are more than two numbers, use the Code Generation Agent instead.\n"
                    "- Code Execution Agent requires a question and **Python file**.\n"
                    "- Code Generation Agent requires a question and **JSON data**.\n"
                    "In case you cannot answer the question and there is not a good coworker, delegate to the Code Generation Agent.\n"
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
                arithmetic_agent, 
                code_execution_agent, 
                code_generation_agent],
        manager_agent=manager_agent,
        tasks=[manager_task],
        verbose=True
    )

    # Process

    final_question = question
    
    if file_path:
        if is_ext(file_path, ".csv") or is_ext(file_path, ".xls") or is_ext(file_path, ".xlsx") or is_ext(file_path, ".json") or is_ext(file_path, ".jsonl"):
            json_data = read_file_json(file_path)
            final_question = f"{question} JSON data:\n{json_data}."
        else:
            final_question = f"{question} File path: {file_path}."
    
    answer = crew.kickoff(inputs={"question": final_question})
    final_answer = get_final_answer(FINAL_ANSWER_MODEL, question, str(answer))

    print(f"=> Initial question: {question}")
    print(f"=> Final question: {final_question}")
    print(f"=> Initial answer: {answer}")
    print(f"=> Final answer: {final_answer}")
    
    return final_answer

def get_final_answer(model, question, answer):
    prompt_template = """
        You are an expert question answering assistant. Given a question and an initial answer, your task is to provide the final answer.
        Your final answer must be a number and/or string OR as few words as possible OR a comma-separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as USD, $, percent, or % unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (for example cities), and write the digits in plain text unless specified otherwise.
        If you are asked for a comma-separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        If the final answer is a number, use a number not a word.
        If the final answer is a string, start with an uppercase character.
        If the final answer is a comma-separated list of numbers, use a space character after each comma.
        If the final answer is a comma-separated list of strings, use a space character after each comma and start with a lowercase character.
        Do not add any content to the final answer that is not in the initial answer.

        **Question:** """ + question + """
        
        **Initial answer:** """ + answer + """
        
        **Example 1:** What is the biggest city in California? Los Angeles
        **Example 2:** How many 'r's are in strawberry? 3
        **Example 3:** What is the opposite of black? White
        **Example 4:** What are the first 5 numbers in the Fibonacci sequence? 0, 1, 1, 2, 3
        **Example 5:** What is the opposite of bad, worse, worst? good, better, best
        
        **Final answer:** 

        """

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    response = client.models.generate_content(
        model=model, 
        contents=[prompt_template]
    )
    
    return response.text
