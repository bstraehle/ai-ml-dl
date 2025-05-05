import os
import pandas as pd
from google import genai

def get_questions(file_path, level):
    df = pd.read_json(file_path, lines=True)
    df = df[df["Level"] == level]
    
    result=[]
    
    for index, row in df.iterrows():
        result.append([row["Level"], row["Question"], row["file_name"], row["Final answer"]])

    return result

def read_file(file_path):
    df = None

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(file_path)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(file_path)

    return "" if df is None else df.to_json()

def get_final_answer(model, question, answer):
    prompt_template = """
        You are an expert question answering assistant. Given a question and an initial answer, your task is to provide the final answer.
        Your final answer must be a number and/or string OR as few words as possible OR a comma-separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as USD, $, percent, or % unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (for example cities), and write the digits in plain text unless specified otherwise.
        If you are asked for a comma-separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        If the final answer is a number, use a number not a word.
        If the final answer is a word, start with an uppercase character.
        If the final answer is a comma-separated list of numbers, use a space character after each comma.
        If the final answer is a comma-separated list of strings, use a space character after each comma and start with a lowercase character.
        Do not add any content to the final answer that is not in the initial answer.

        **Question:** """ + question + """
        
        **Initial answer:** """ + answer + """
        
        **Example 1:** How many 'r's are in strawberry? 3
        **Example 2:** What is the opposite of black? White
        **Example 3:** What is the biggest city in California? Los Angeles
        **Example 4:** What is the superlative of good? Best
        **Example 5:** What are the first 10 numbers in the Fibonacci sequence? 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        
        **Final answer:** 

        """

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    response = client.models.generate_content(
        model=model, 
        contents=[prompt_template]
    )
    
    return response.text
