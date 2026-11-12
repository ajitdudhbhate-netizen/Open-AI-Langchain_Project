
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Question: {question}")
])


# Generate response
def generate_response(query, llm, temperature, max_tokens, api_key):

    model = ChatOpenAI(
        model=llm,
        temperature=temperature,
        api_key=api_key,
        model_kwargs={"max_tokens": max_tokens}
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"question": query})

    return result


# Streamlit application
st.title("Q & A Chatbot With OpenAI")

query = st.text_input("What is your query?")

st.sidebar.title("Settings")

api = st.sidebar.text_input(
    "Provide an API key",
    type="password"
)

llm = st.sidebar.selectbox(
    "Select the LLM:",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
)

temperature = st.sidebar.slider(
    "Select Temperature:",
    min_value=0.0,
    max_value=2.0,
    value=0.8
)

max_tokens = st.sidebar.slider(
    "Select Max Tokens:",
    min_value=50,
    max_value=2000,
    value=500
)

if st.button("Answer"):

    if query and api:

        with st.spinner("Generating response..."):

            response = generate_response(
                query,
                llm,
                temperature,
                max_tokens,
                api
            )

        st.write(response)

    else:
        st.warning("Please enter your query and API key.")
