import os
import requests
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
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


@function_tool
def get_capital(country:str)-> str:
    """Fetch the capital of the given country from user"""
    response = requests.get(f"https://restcountries.com/v3.1/name/{country}?fullText=True&fields=capital")
    data = response.json()
    capital = data[0]['capital'][0]
    return f"The capital of {country} is {capital}"

@function_tool
def get_language(country: str)-> str:
    """Fetch the language of the given country from user"""
    response = requests.get(f"https://restcountries.com/v3.1/name/{country}?fullText=True&fields=languages")
    data = response.json()
    language = data[0]['languages']
    lang_str = ','.join(language.values())
    return f"The spoken language of {country} is {lang_str}"
    

@function_tool
def get_population(country: str)-> str:
    """Fetch the population of the given country from user."""
    response = requests.get(f"https://restcountries.com/v3.1/name/{country}?fullText=True&fields=population")
    data = response.json()
    population = data[0]['population']
    return f"The current population of {country} is approximately {population}"
    
agent = Agent(
    name= "Orchestractor Agent",
    instructions= "You are a helpful agent.you takes the country name from user and use all tools to give complete info about that country.",
    tools=[get_capital,get_language, get_population]
)

prompt = input("Do you want to know about any country: ")

result = Runner.run_sync(
    agent,
    prompt,
    run_config= config
)

print(result.final_output)
