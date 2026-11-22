# The code below should be added to a util.py file

import requests
import json

from dotenv import load_dotenv
import os
_ = load_dotenv() #loads 'TOGETHER_API_KEY'


  # The right API to pass in a prompt (of type string) is the completions API https://docs.together.ai/reference/completions-1
  # The right API to pass in a messages (of type of list of message) is The chat completions API https://docs.together.ai/reference/chat-completions-1

def llama31(prompt_or_messages, model_size=8, temperature=0, raw=False, debug=False):
    model = f"meta-llama/Meta-Llama-3.1-{model_size}B-Instruct-Turbo"
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "prompt": prompt
        }
    else:
        messages = prompt_or_messages
        url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/chat/completions"
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }

    if debug:
        print(payload)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()  # Raises HTTPError for bad responses
        res = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")

    if 'error' in res:
        raise Exception(f"API Error: {res['error']}")

    if raw:
        return res

    if isinstance(prompt_or_messages, str):
        return res['choices'][0].get('text', '')
    else:
        return res['choices'][0].get('message', {}).get('content', '')

# pretty print JSON with syntax highlighting
import json
from pygments import highlight, lexers, formatters
def cprint(response):
    formatted_json = json.dumps(response, indent=4)
    colorful_json = highlight(formatted_json,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print(colorful_json)

# pretty print JSON with syntax highlighting
import json
from pygments import highlight, lexers, formatters

def cprint(response):
    formatted_json = json.dumps(response, indent=4)
    colorful_json = highlight(formatted_json,
                              lexers.JsonLexer(),
                              formatters.TerminalFormatter())
    print(colorful_json)


def html_tokens(tokens):
  # simulate the color values used in https://tiktokenizer.vercel.app
  on_colors = ["#ADE0FC", "#FCE278", "#B2D1FE", "#AFF7C6", "#FDCE9B", "#97F1FB", "#DEE1E7", "#E3C9FF", "#BBC6FD", "#D1FB8C"]

  # Create an HTML string with colored spans
  html_string = ""
  for i, t in enumerate(tokens):
      if t == "\n":
            t = "\\n"
      elif t == "\n\n":
            t = "\\n\\n"
      on_col = on_colors[i % len(on_colors)]
      html_string += f'<span style="color: black; background-color: {on_col}; padding: 2px;">{t}</span>'

  return html_string


# The code below should be added to a util.py file

import requests
import json

def llamaguard3(prompt, debug=False):
  model = "meta-llama/Meta-Llama-Guard-3-8B"
  url = f"{os.getenv('DLAI_TOGETHER_API_BASE', 'https://api.together.xyz')}/v1/completions"
  payload = {
    "model": model,
    "temperature": 0,
    "prompt": prompt,
    "max_tokens": 4096,
  }

  headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TOGETHER_API_KEY"]
  }
  res = json.loads(requests.request("POST", url, headers=headers, data=json.dumps(payload)).content)

  if 'error' in res:
    raise Exception(res['error'])

  if debug:
    print(res)
  return res['choices'][0]['text']



