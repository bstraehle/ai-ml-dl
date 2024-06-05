import base64, json, os

from autogen import ConversableAgent, AssistantAgent
from autogen.coding import LocalCommandLineCodeExecutor

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def read_image_file(image_file_path: str) -> str:
    with open(image_file_path, "rb") as image_file:
        image_data = image_file.read()
        return base64.b64encode(image_data).decode("utf-8")

def generate_markdown_image(image_data: str) -> str:
    return f"![Image](data:image/png;base64,{image_data})"

def format_as_markdown(code: str) -> str:
    markdown_code = '```\n'
    markdown_code += code
    markdown_code += '\n```'
    return markdown_code
    
def run_multi_agent(llm, message):
    llm_config = {"model": llm}
    
    executor = LocalCommandLineCodeExecutor(
        timeout=60,
        work_dir="coding",
    )

    code_executor_agent = ConversableAgent(
        name="code_executor_agent",
        llm_config=False,
        code_execution_config={"executor": executor},
        human_input_mode="NEVER",
        default_auto_reply="TERMINATE",
    )

    code_writer_agent = AssistantAgent(
        name="code_writer_agent",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    
    chat_result = code_executor_agent.initiate_chat(
        code_writer_agent,
        message=message,
        max_turns=10
    )

    chat = ""
    
    for message in chat_result.chat_history:
        chat += f"**{message['role'].replace('assistant', 'Code Executor').replace('user', 'Code Writer')}**\n{message['content']}\n\n"
    
    image_data = read_image_file("/home/user/app/coding/ytd_stock_gains.png")
    markdown_code_png = generate_markdown_image(image_data)

    '''
    file_name_py = ""
    file_name_sh = ""
    
    for file in os.listdir("coding"):
        if file:
            _, file_extension = os.path.splitext(file)
            if file_extension == ".py":
                file_name_py = file
            if file_extension == ".sh":
                file_name_sh = file
    
    try:
        file_path_py = "coding/" + file_name_py
        code_py = read_file(file_path_py)
        markdown_code_py = format_as_markdown(code_py)

        file_path_sh = "coding/" + file_name_sh
        code_sh = read_file(file_path_sh)
        markdown_code_sh = format_as_markdown(code_sh)
    except FileNotFoundError:
        print(f"Error: File '{file_path_sh}' not found.")
    except IOError as e:
        print(f"Error reading file '{file_path_sh}': {e.strerror}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    '''
     
    #result = f"{markdown_code_png}\n\n{markdown_code_sh}\n\n{markdown_code_py}\n\n{chat}"
    result = f"{markdown_code_png}\n\n{chat}"

    #print("===")
    #print(result)
    #print("===")
    
    return result
