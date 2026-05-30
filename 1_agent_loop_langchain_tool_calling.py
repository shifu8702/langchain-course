"""
1. uv init
2. load langchain, model, python-dotenv, black, isort
3. create .env file
4. set up all from '' import '' # importing tools, models, messages, traceable
5. create tools with @tool decorator
6. create agent loop
"""
from dotenv import load_dotenv
from langchain_core import messages
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 4
MODEL = "qwen3:1.7b"

#------Tools (LangChain @tool decorator)------
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a prduct in the catalog"""
    print(f"   >>Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 999.99,
        "smartphone": 499.95,
        "headphones": 199.50
    }
    return prices.get(product, "Product not found")

@tool
def apply_discount(price: float, discount_tier:str)-> float:
    """Apply a discount tier to a price and return the final price
    Available discount tiers: bronze, silver, gold."""
    print(f"   >>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")

    discounts_percentage = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }
    discount = discounts_percentage.get(discount_tier, 0)
    return round(price * (1 - discount/200), 2)



#------Agent Loop------
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = ChatGoogleGenerativeAI(temperature=0, model="gemma-4-31b-it")
    #llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools) 

    print(f"Question: {question}")
    print('=' * 60)

    messages =[
        SystemMessage(
           content=(
              "You are a helpful assistant shopping assistant. "
              "You have access to product catalog tool "
              "and a discount tool. \n\n"
              "STRICT RULES - you must follpow these exactly:\n"
              "1. NEVER guess or assume any product price. "
              "You must call get_product_price first to get real price.\n"
              "2.Onl call apply_discount after you have received a price from get_product_price. "
              "pass the exact price "
              "3. NEVER calculate discounts yourself using math. "
              "Always use the apply_discount tool. \n"
              "4. If the user don't specify a discount tier, "\
              "ask them which tier to use - Do NOT assume one. "
           )
        ),
        HumanMessage(content=question),
     ]

    for i in range(1, MAX_ITERATIONS + 1): 
        print(f"\n--- Iteration {i} ---")
    
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        #If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        #process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
    
        observation = tool_to_use.invoke(tool_args)
    
        print(f" [Tool Result] {observation}")

        messages.append(ai_message)
        #print( ai_message.content)
        messages.append(
        ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    
    print("ERROR: Max iterations reached without a final answer.")
    return None
    
        
if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")

