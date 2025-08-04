import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv

load_dotenv()

gemini_api_key= os.getenv("GEMINI_API_KEY")

external_client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=external_client,
    
)

config = RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled= True
)

# Define two agents with different jobs
math_agent = Agent(
    name= "Math Expert", 
    instructions= "You are a helpful agent.Solve any Math related problems"
)

english_agent = Agent(
    name= "English Expert",
    instructions= "You are a helpful agent.Summarize any text solve English related problems."
)

# Handoffs method
distributer_agent = Agent(
    name= "Distributer Agent",
    instructions="You are a smart distributor. Based on the user's input, forward the task "
        "to either the Math Expert for math problems or English Expert for English-related queries.",
    handoffs=[english_agent, math_agent]
    )

user_query= input("Enter your question(Math or English related): ")

# Run the swarm 
results = Runner.run_sync(
    distributer_agent,
    user_query,
    run_config= config
)

print(results.final_output)