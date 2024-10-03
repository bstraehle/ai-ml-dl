import base64, datetime, json, os

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

def get_latest_file(directory, file_extension):
    latest_file = None
    latest_date = datetime.datetime.min

    for file in os.listdir(directory):
        if file:
            _, file_ext = os.path.splitext(file)

            if file_ext == file_extension:
                file_path = os.path.join(directory, file)
                file_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_date > latest_date:
                    latest_date = file_date
                    latest_file = file

    return latest_file
    
def run_multi_agent(llm, task):
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
        message=task,
        max_turns=10
    )

    chat = ""
    first_message = True
    
    for message in chat_result.chat_history:
        if not first_message:
            chat += f"**{message['role'].replace('assistant', 'Code Executor').replace('user', 'Code Writer')}**\n{message['content']}\n\n"
        first_message = False

    file_name_png = get_latest_file("coding", ".png")
    
    image_data = read_image_file(f"/home/user/app/coding/{file_name_png}")
    markdown_code_png = generate_markdown_image(image_data)

    result = f"{markdown_code_png}\n\n{chat}"

    print("===")
    print(result)
    print("===")
    
    return result
