from generation.generator import Generator
from pprint import pprint

if __name__ == "__main__":
    
    # Initialize components
    generator = Generator()
    
    # Get the agent executor from the generator
    agent_executor = generator.agent_executor
    
    # Start a conversation loop
    print("Movie Recommendation Assistant (type 'exit' to quit)")
    
    while True:
        # Rename 'input' to 'user_query' to avoid conflict with Python's built-in function
        user_query = input("\nWhat kind of movie are you looking for? ")
        
        # Check for exit command
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the Movie Recommendation Assistant. Goodbye!")
            break
            
        # Process the user's query - only pass the input parameter for memory to work correctly
        response = agent_executor.invoke({
            "input": user_query
        })
        # Print the response in a more readable format
        print("\n----- RECOMMENDATION RESULTS -----")
        if "output" in response:
            print(response["output"])
        else:
            pprint(response)
        print("---------------------------------\n")