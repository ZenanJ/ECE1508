import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from retrieval.retriever import Retriever
from generation.prompt_templates import PromptTemplates
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from utils.config import DEEPSEEK_API_KEY, LOCAL_MODEL_NAME, OPENAI_API_KEY

 
class Generator:
    def __init__(self, model_choice="remote", llm_choice="deepseek"):
        
        # Load environment variables
        load_dotenv()

        # Set up the LLM based on model choice
        retriever = Retriever()
        
        # Ask user to choose between local or remote model
        if model_choice == "local":
            self.llm = ChatOllama(model=LOCAL_MODEL_NAME).bind_tools([retriever.search_movie])
        elif model_choice == "remote":
            if llm_choice == "deepseek":
                self.llm = ChatDeepSeek(model="deepseek-chat", api_key=DEEPSEEK_API_KEY)
            elif llm_choice == "openai":
                self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
        else:
            raise ValueError("model_choice must be either 'local' or 'remote'")

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
    