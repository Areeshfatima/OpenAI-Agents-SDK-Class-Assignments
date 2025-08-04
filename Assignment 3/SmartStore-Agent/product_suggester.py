import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled= True
)

smart_agent = Agent(
    name= "Smart Store Agent",
    instructions="You are a helpful smart store agent.Suggest a product based on user need.If the user say 'I have a headache'.It should suggest a medicine and explain why."
)

prompt = input("Describe your need or symptom: ")

response = Runner.run_sync(
    smart_agent,
    prompt,
    run_config=config
)

print(response.final_output)
