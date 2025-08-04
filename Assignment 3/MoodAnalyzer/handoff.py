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

Mood_analyzer = Agent(
    name= "Mood Analyzer Agent",
    instructions="You are a helpful mood analyzer agent.You have to check user's mood from their message."
)

Suggest_activity = Agent(
    name= "Suggest Activity Agent",
    instructions="You are a helful suggest activity agent. You have to suggest best activity according to user's mood."
)

Triage_agent = Agent(
    name= "Triage Agent",
    instructions="You are a helpful agent that can triage questions according to user's input ansd suggest activity according to their mood's.",
    handoffs=[Mood_analyzer, Suggest_activity]
)

prompt = input("How you feel today?/Should I suggest anything to you?")

result = Runner.run_sync(
    Triage_agent,
    prompt,
    run_config=config
)

# print('last agent:', result.last_agent)
print(result.final_output)
