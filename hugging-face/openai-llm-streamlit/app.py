import streamlit as st
import os

from openai import OpenAI

config = {
    "max_tokens": 1000,
    "model": "gpt-4",
    "temperature": 0
}

def invoke(openai_api_key, prompt):
    if (openai_api_key == ""):
        st.error("OpenAI API Key is required.", icon = "🚨")
        return ""
    if (prompt == ""):
        st.error("Prompt is required.", icon = "🚨")
        return ""

    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    content = ""
    
    try:
        client = OpenAI()
    
        completion = client.chat.completions.create(
            max_tokens = config["max_tokens"],
            messages = [{"role": "user", "content": prompt}],
            model = config["model"],
            temperature = config["temperature"])
    
        content = completion.choices[0].message.content
    except Exception as e:
        err_msg = e

        st.error(e, icon = "🚨")
        return ""

    return content

description = """[Streamlit](https://streamlit.io/) UI using the [OpenAI](https://openai.com/) SDK 
                 with [gpt-4](https://openai.com/research/gpt-4) model."""

st.title("Generative AI - LLM")
st.write(description)
completion = ""

with st.form("myform"):
    openai_api_key = st.text_input("OpenAI API Key", type = "password")
    prompt = st.text_input("Prompt")
    submitted = st.form_submit_button("Submit")

    if submitted:
        with st.spinner(""):
            completion = invoke(openai_api_key, prompt)

if completion != "":
    st.info(completion)
