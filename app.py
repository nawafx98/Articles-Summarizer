import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

api_key = os.environ.get('OPENAI_API_KEY')

# App framework
st.title('🦜🔗 Articles Summarizer')

# User input with placeholder
user_input = st.text_area("Enter your article text here:", placeholder="Paste your article content here for summarization...")

# Check if the user input is not empty
if not user_input:
    st.warning("Please enter the article text to generate a summary.")
else:
    # Button to generate summary
    if st.button("Generate Summary"):
        # Check if the input is too short
        if len(user_input) < 50:
            st.warning("The article text is too short. Please provide a longer article for better summarization.")
        else:
            # Prompt templates
            summary_template = PromptTemplate(
                input_variables=['user_input'],
                template="Please provide a comprehensive and detailed summary of the following text: {user_input}. Ensure that your summary is clear, accurate, and complete."
            )

            # Llms
            llm = OpenAI(temperature=0.7)
            summary_chain = LLMChain(llm=llm, prompt=summary_template, verbose=True, output_key='summary')

            # Memory
            summary_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history')

            chunk_size = 4000  # Adjust this value as needed
            chunks = [user_input[i:i+chunk_size] for i in range(0, len(user_input), chunk_size)]

            with st.spinner("Generating summary..."):
                full_summary = ""

                for i, chunk in enumerate(chunks):
                    summary = summary_chain.run(user_input=chunk)
                    full_summary += summary 

            st.subheader('Summary:')
            st.write(full_summary)
