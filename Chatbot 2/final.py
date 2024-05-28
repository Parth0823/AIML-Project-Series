# Importing Libraries
import os
from constants import huggingface_token
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import chainlit as cl

# Setting up Hugging face and efining the model
os.environ["HUGGINGFACEHUB_API_TOKEN"]=huggingface_token
llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3",model_kwargs={"max_length":128,"token":huggingface_token},temperature=0.3)

# Using DuckDuckGo for backend
search = DuckDuckGoSearchRun()

# Setting up toolsfor the agent
tools=[]
duckduckgo_tool = Tool(
    name='DuckDuckGo Search',
    func= search.run,
    description="Useful for when you need to do a search on the internet to find college realted information. be specific with your input."
)
tools.append(duckduckgo_tool)

# Setting up the Agent with memory
conversational_agent = initialize_agent(
    agent="conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=ConversationBufferMemory(memory_key="chat_history")
)

@cl.on_message
async def chat(message):
    """
    Handles incoming user messages and generates responses using the agent.
    """
    # Send a loader message
    loader_message = await cl.Message(content="Processing...").send()

    # Get the response from the agent
    response = conversational_agent.run(input=message.content)
    
    # Update the loader message with the actual response
    loader_message.content=response
    await loader_message.update()

# Run the Chainlit app
if __name__ == "__main__":
    cl.run()