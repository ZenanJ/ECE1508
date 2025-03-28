import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from retrieval.retriever import Retriever
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from retrieval.retriever import Retriever
from generation.prompt_templates import PromptTemplates
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from utils.config import DEEPSEEK_API_KEY, LOCAL_MODEL_NAME

 
class Generator:
    def __init__(self):
        
        # Load environment variables
        load_dotenv()  
        
        # Ask user to choose between local or remote model
        print("Choose between local or remote model:")
        print("1. Local")
        print("2. Remote")
        choice = input("Enter 1 or 2: ")
        if choice == "1":
            self.llm = ChatOllama(model=LOCAL_MODEL_NAME).bind_tools([Retriever().search_movie])
        elif choice == "2":
            self.llm = ChatDeepSeek(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)
   
        # Create the tools list
        retriever = Retriever()
        
        # bind the search_movie tool to the llm
        tools = [
            Tool(
                name="static_search_movie",
                func=retriever.static_search_movie,
                description="""Search for movies based on criteria. 
                Args: 
                    query_text (str): Search text (e.g., 'sci-fi')
                    avg_rating (float, optional): Minimum rating (default: 4.0)
                    num_ratings (int, optional): Minimum number of ratings (e.g., 500)
                    top_k (int, optional): Number of results to return (default: 50)
                """
            )
        ]

        # Get tool names
        tool_names = [tool.name for tool in tools]

        # Create proper memory object
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        # Create the agent prompt with all required variables and chat_history
        agent_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tool_names", "tools", "chat_history"],
            template=PromptTemplates.USER_INPUT
        )

        # Create the agent with proper configuration
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=agent_prompt
        )

        # Create the agent executor with properly initialized memory
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            return_intermediate_steps=True
        )
    