import os
from gpt_index import GPTSimpleVectorIndex
from streamlit_chat import message

import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai_secret"]

st.set_page_config(
    page_title="Details Chatbot",
    page_icon=":robot:",
    layout="wide"
)

with st.sidebar:
    st.subheader("Details")
    placeholder_sidebar = st.empty()

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

st.header("Details Chatbot")
message("Hi, I am details chatbot, how can i help?")
placeholder = st.empty() 

def ask_bot(input_index = 'index.json', query=""):
    index = GPTSimpleVectorIndex.load_from_disk(input_index)
    # query = input('What do you want to ask the bot?   \n')
    response = index.query(query, response_mode="compact")
    return response.response
    # response.response
    # print ("\nBot says: \n\n" + response.response + "\n\n\n")

def get_text():
    input_text = st.text_input(" ","", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output = ask_bot('indexes/zendesk_index.json', f"{user_input} with source url")
    # st.session_state["input"] = ""
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with placeholder.container():
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

with placeholder_sidebar.container():
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            st.write(st.session_state['past'][i])