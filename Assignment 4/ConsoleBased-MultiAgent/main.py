import os
from dotenv import load_dotenv
from typing import Optional
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, RunContextWrapper, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered, function_tool, output_guardrail
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from pydantic import BaseModel


# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.Please ensure it is defined in your .env file.")

external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure the model
model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled= True,
)

# Define context model
class UserInfo(BaseModel):
    name: str
    is_premium_user: bool
    issue_type: Optional[str] = None

# Output guardrail model
class NoApologyOutput(BaseModel):
    contains_banned_words: bool
    reason: str

# Guardrail agent to prevent apologies
guardrail_agent = Agent(
    name= "Guardrail Agent",
    instructions="You are the Guardrail Agent. Validate the assistant's output before it is shown to the user.\n"
                 "Rules:\n"
                 "1) The text must not contain apology words such as 'sorry', 'apologize', or 'apologies'.\n"
                 "2) If found, set contains_banned_words=True and explain why.\n"
                 "3) Otherwise set contains_banned_words=False and explain why it is safe.\n",
    output_type= NoApologyOutput,
    model=model,
)

@output_guardrail
async def apology_guardrail(ctx: RunContextWrapper, agent:Agent, output):
    response = await Runner.run(
        guardrail_agent,
        input= output,
        context= ctx.context,
        run_config= config,
     )
    return GuardrailFunctionOutput(
        output_info= response.final_output,
        tripwire_triggered= response.final_output.contains_banned_words
    )


# General info tool
@function_tool
def general_info(query: str)-> str:
    """General lookup stub. In a real app, you could call docs or knowledge base."""
    return f"I found some useful general information about '{query}'. It may help you with your question."

# Refund tool with conditional enablement
def is_refund_enabled(wrapper: RunContextWrapper[UserInfo], agent:Agent[UserInfo])-> bool:
    return wrapper.context.is_premium_user


@function_tool(is_enabled= is_refund_enabled)
def refund(wrapper: RunContextWrapper[UserInfo])-> str:
    """Handle refunds. Enabled only for premium users via is_enabled."""
    ctx = wrapper.context
    if not ctx.is_premium_user:
        return f"Refunds are only available for premium users, {ctx.name}."
    return f"Refund successfully processed for {ctx.name}."

# Restart services tool with conditional enablement
def is_restart_enabled(wrapper: RunContextWrapper[UserInfo], agent:Agent[UserInfo])-> bool:
    return wrapper.context.issue_type == "technical"
    
@function_tool(is_enabled= is_restart_enabled)
def restart_services(service: str)-> str:
    """Restart a named service (only when issue_type == 'technical')."""
    return f"Your {service} is restarting"

# Define Agent
general_agent = Agent(
    name= "General Agent",
    instructions=(
        "You are a concise and helpful general-support agent.\n"
        "Answer general questions clearly. Use the general_info tool when it adds value.\n"
        "Do not ask for apologies."
    ),
    tools=[general_info],
    model=model,
)

billing_agent = Agent(
    name= "Billing Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are the billing specialist. Handle billing and refunds only.\n"
        "If user mentions refund, ALWAYS call the refund tool.\n"
        "After tool output, provide a short confirmation.\n"
        "Do not continue after confirming."
    ),
    tools=[refund],
    model=model,
)

technical_agent = Agent(
    name= "Technical Agent",
    instructions=(
        "You are the technical-support specialist.\n"
        "If user asks to restart something or reports errors/problems, use the restart_service tool when appropriate.\n"
        "Give clear, step-by-step guidance and end succinctly."

    ),
    tools=[restart_services],
    model=model,
)

# Triage agent with issue type determination
triage_agent = Agent[UserInfo](
    name= "Triage Agent",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a routing agent. Tasks:\n"
        "1) Determine issue_type from the user's message:\n"
        " If it contains 'refund' or 'bill' → issue_type='billing'\n"
        " If it contains 'restart', 'not working', 'error', 'issue', 'problem' → issue_type='technical'\n"
        " Otherwise → issue_type='general'\n"
        "2) Update the shared context with the issue_type.\n"
        "3) Handoff to the appropriate agent based on issue_type.\n"
        "4) After calling the handoff, do NOT produce any additional content. Terminate the turn."
    ),
    tools=[general_info, refund, restart_services],
    handoffs=[general_agent, technical_agent, billing_agent],
    output_guardrails=[apology_guardrail],
    model=model,
)

print("\n=== Console-Based Support Agent System ===\n")

name = input("Enter your name: ").strip()
is_premium = input("Are you a premium user(yes/no): ").strip().lower() == "yes"
prompt = input("Enter your query: ").strip()

user_info = UserInfo(
    name= name,
    is_premium_user= is_premium,
    issue_type= None
)

try:
    response = Runner.run_sync(
    triage_agent,
    prompt,
    context= user_info,
    run_config= config,
    )

    print("\n======= Final Output =======\n")

    print(response.final_output)

except OutputGuardrailTripwireTriggered as e:
    print("\n ===== Output =====\n")

    print(f"Trip output \n {e}")