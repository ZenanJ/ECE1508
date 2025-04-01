import argparse
from generation.generator import Generator
from pprint import pprint
import gradio as gr


# original command line interface
def launch_cli(agent_executor):
    print("Movie Recommendation Assistant (type 'exit' to quit)")
    while True:
        user_query = input("\nWhat kind of movie are you looking for? ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the Movie Recommendation Assistant. Goodbye!")
            break

        response = agent_executor.invoke({"input": user_query})
        print("\n----- RECOMMENDATION RESULTS -----")
        if "output" in response:
            print(response["output"])
        else:
            pprint(response)
        print("---------------------------------\n")


# using gradio chatbot interface
def launch_gradio_ui(agent_executor):

    # Define the chatbot function with agent trace (Final Answer + Agent Reasoning)
    # def chat_with_agent(message, history):
    #     try:
    #         result = agent_executor.invoke({"input": message})
    #         final_answer = result["output"]
    #
    #         # Collect agent thoughts and actions
    #         thoughts = []
    #         for step in result.get("intermediate_steps", []):
    #             action = step[0]
    #             observation = step[1]
    #             thoughts.append(
    #                 f"Thought: {action.log.strip()}\n"
    #                 f"Action: {action.tool}\n"
    #                 f"Action Input: {action.tool_input}\n"
    #                 f"Observation: {observation}"
    #             )
    #         chain_of_thought = "\n\n".join(thoughts)
    #
    #         # Append detailed trace to final answer (optional, can hide it too)
    #         full_response = f"{final_answer}\n\n---\n🧠 **Agent Reasoning**\n{chain_of_thought}"
    #         return full_response
    #
    #     except Exception as e:
    #         return f"❌ Error: {str(e)}"

    # Define the function for ChatInterface (only show Final Answer)
    def chat_with_agent(message, history):
        try:
            result = agent_executor.invoke({"input": message})
            return result["output"]  # Just return the Final Answer
        except Exception as e:
            return f"❌ Error: {str(e)}"

    demo = gr.ChatInterface(
        fn=chat_with_agent,
        type="messages",
        title="🎬 Movie Recommendation Assistant",
        description="Ask for movie suggestions and see how the agent reasons with tools!",
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="What kind of movie are you looking for?", submit_btn="🚀 Submit"),
        theme="soft",
        )

    demo.launch()


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Launch Movie Recommendation Assistant")
    parser.add_argument('--mode', choices=['cli', 'ui'], default='ui', help='Mode to run: cli or ui')
    parser.add_argument('--model', choices=['local', 'remote'], default='remote', help="Choose the LLM model, 'local' or 'remote'")
    parser.add_argument('--llm', choices=['deepseek', 'openai'], default='deepseek', help="Choose the LLM model, 'deepseek' or 'openai'")
    args = parser.parse_args()
    
    # Initialize generator with model choice
    generator = Generator(model_choice=args.model, llm_choice=args.llm)
    agent_executor = generator.agent_executor
    
    # Launch the selected mode
    if args.mode == 'cli':
        launch_cli(agent_executor)
    else:
        launch_gradio_ui(agent_executor)