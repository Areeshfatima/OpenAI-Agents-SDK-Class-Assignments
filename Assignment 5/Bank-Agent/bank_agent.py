import os
import re
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, RunContextWrapper, function_tool, input_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from dotenv import load_dotenv

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

# Simulated account data (for demo purposes)
accounts = {
    "9876543": {
        "balance" : 100000000,
        "name" : "Arisha"
    },
    "3456789": {
        "balance" : 5000.00,
        "name" : "Abrish"
    },
}

# Pydantic Model
class Account(BaseModel):
    name: str
    pin: int

class Guardrail_output(BaseModel):
    is_not_bank_related: bool

class TransferRequest(BaseModel):
    from_account: str
    to_account: str
    amount: float

# Input Guardrail Agent
guardrial_agents = Agent(
    name="Guardrail Agent",
    instructions="Determine if the user query is related to banking tasks (balance checks, transfers).",
    output_type=Guardrail_output,
)

@input_guardrail
async def check_bank_related(ctx:RunContextWrapper[None], agent:Agent, input: str)-> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrial_agents,
        input, 
        context=ctx.context,
        run_config= config,
        )
    return GuardrailFunctionOutput(
        output_info= result.final_output,
        tripwire_triggered= result.final_output.is_not_bank_related
    )

# User Authentication
def check_user(ctx: RunContextWrapper[Account], agent:Agent)-> bool:
    """Validate user credentials."""
    return ctx.context.name == "Arisha" and ctx.context.pin == 1234

# Balance Inquiry Tool
@function_tool(is_enabled= check_user)
def check_balance(account_number: str) -> str:
    """Check the balance for a given account number."""
    if not re.match(r'^\d{7}$', account_number):
        return "Invalid account number. It must be a 7-digit number."
    if account_number not in accounts:
        return "Account not found."
    
    balance = accounts[account_number]["balance"]
    return f"Your account balance is ${balance:.2f}."

# Fund Transfer Tool
@function_tool(is_enabled= check_user)
def transfer_funds(from_account: str, to_account: str, amount: str) -> str:
    """Handle fund transfers between accounts."""
    if not(re.match(r'^\d{7}$', from_account) and re.match(r'^\d{7}$', to_account)):
        return "Invalid account number(s). Must be 7-digit numbers."
    try:
        amount = float(amount)
        if amount <= 0:
            return "Amount must be positive."
    except ValueError:
        return "Invalid amount. Please enter a numeric value."
    
    if from_account not in accounts and to_account not in accounts:
        return "One or both accounts not found."
    if accounts[from_account]["balance"] < amount:
        return "Insufficient funds for the transfer."
    
    accounts[from_account]["balance"] -= amount
    accounts[to_account]["balance"] += amount

    return f"Successfully transferred ${amount:.2f} from account {from_account} to account {to_account}."

# Output Guardrail
def sanitize_output(response: str) -> str:
    """Prevent leakage of sensitive internal data."""
    if "accounts" in response.lower() or "pin" in response.lower():
        return "Sorry, I cannot share sensitive internal data."
    return response

# Balance Inquiry Agent
balance_agent = Agent(
    name= "Balance Inquiry Agent",
    instructions="Handle customer requests to check their account balance. Ensure user authentication and valid account numbers.",
    tools=[check_balance],
    input_guardrails=[check_bank_related],
)

# Fund Transfer Agent
transfer_agent = Agent(
    name= "Fund Transfer Agent",
    instructions="Handle customer requests to transfer funds. Validate account numbers, amounts, and ensure sufficient funds.",
    tools=[transfer_funds],
    input_guardrails=[check_bank_related],  
)

# Main Bank Agent
bank_agents = Agent(
    name= "Bank Agent",
    instructions="Route customer queries to the appropriate agent (balance inquiry or fund transfer). Ensure user authentication and bank-related queries.",
    handoffs=[balance_agent, transfer_agent],
    tools=[check_balance, transfer_funds],
    input_guardrails= [check_bank_related],
)

user_context = Account(
    name= "Arisha",
    pin=1234
)

# Test 1: Balance inquiry
prompt_balance = "What is my balance? This is my account number 9876543"

result_balance = Runner.run_sync(
    balance_agent,
    prompt_balance,
    context=user_context,
    run_config= config,
)
print("\n ==== Balance Inquiry Response ==== \n")
print(sanitize_output(result_balance.final_output))

# Test 2: Fund transfer
prompt_transfer = "Transfer $1000 from account 9876543 to account 3456789."

result_transfer = Runner.run_sync(
    transfer_agent,
    prompt_transfer,
    context=user_context,
    run_config= config,
)

print("\n ==== Fund Transfer Response ==== \n")
print(sanitize_output(result_transfer.final_output))

print("\n ==== Accounts After Transfer ==== \n")
print(accounts)


# Test 3: Invalid input (non-bank-related)
prompt_invalid = "What's the weather today? "

try:
    result_invalid = Runner.run_sync(
    bank_agents,
    prompt_invalid,
    context=user_context,
    run_config= config,
)
    print("\n ==== Invalid Input Response ==== \n")
    print(sanitize_output(result_invalid.final_output))

except Exception as e:
    print("\n ==== Invalid Input Response  ==== \n")
    print("Sorry, I can only handle bank-related queries like balance checks or fund transfers.")


