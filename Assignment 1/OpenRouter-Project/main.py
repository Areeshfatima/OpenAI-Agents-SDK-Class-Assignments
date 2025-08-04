import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load .env file and get API key
load_dotenv()

openrouter_api_key= os.getenv("OPENROUTER_API_KEY")


# Check if key exists
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.Please ensure it is defined in your .env file.")

# Setup OpenRouter client(like OpenAI. but via OpenRouter)
external_client = AsyncOpenAI(
    api_key= openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",

)

# Choose any OpenRouter-supported model
model = OpenAIChatCompletionsModel(
    model= "google/gemini-2.0-flash-exp:free",   # example model, replace if needed
    openai_client=external_client
)

# Setup config
config = RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled= True
)

agent = Agent(
    name= "Writer Agent",
    instructions= "You are a helpful writer agent. Generate stories, poems, essay etc."
)

prompt = "Write a short essay about Pakistan current situation in simple English."

result = Runner.run_sync(
    agent,
    prompt,
    run_config= config
)

print(result.final_output)