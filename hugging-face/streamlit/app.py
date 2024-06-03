import streamlit as st
import os, threading

from openai import OpenAI

lock = threading.Lock()

config = {
    "max_tokens": 1000,
    "model": "gpt-4o",
    "temperature": 0
}

def invoke(openai_api_key, prompt):
    if not openai_api_key:
        st.error("OpenAI API Key is required.", icon = "🚨")
        return ""
    
    if not prompt:
        st.error("Prompt is required.", icon = "🚨")
        return ""

    with lock:
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
        finally:
            del os.environ["OPENAI_API_KEY"]
    
        return content

st.title("Generative AI - LLM")
st.write(os.environ["DESCRIPTION"])
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
