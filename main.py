import os
import json
import yfinance as yf
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
from typing import List, Dict, Any
from curl_cffi import requests

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Function Implementations
def get_surface_area(width,length,hight):
    width_info = width
    length_info = length
    hight_info = hight
    pocet_m2 = round((2* (float(width_info)*float(hight_info)) + 2*(float(length_info)*float(hight_info)) + float(width_info)*float(length_info)),2)
    return {"celkom m2": pocet_m2}

def get_price(price,surface_area):
    price_info = price
    surface_area_info = surface_area
    total_price = float(price_info) * float(surface_area_info)
    return {"celkova cena": total_price}


# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_surface_area",
            "description": "Použi túto funkciu na výpočet celkovej plochy izby na základe jej rozmerov.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {
                        "type": "string",
                        "description": "šírka plochy",
                    },
                    "length":{
                        "type": "string",
                        "description": "dĺžka plochy",
                    },
                    "hight":{
                        "type": "string",
                        "description": "výška plochy",
                    },
                },
                "required": ["width", "length", "hight"],
            
            }  ,
        },
    },  
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Použi túto funkciu na výpočet celkovej ceny",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "string",
                        "description": "cena za 1m2",
                    },
                    "surface_area": {
                        "type": "string",
                        "description": "celková plocha izby v m2",
                },
                
                },
                "required": ["price", "surface_area"],
            }
        },
    }
]

available_functions = {
    "get_surface_area": get_surface_area,
    "get_price": get_price,
}


class ReactAgent:
    """A ReAct (Reason and Act) agent that handles multiple tool calls."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.max_iterations = 10  # Prevent infinite loops

    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run the ReAct loop until we get a final answer.

        The agent will:
        1. Call the LLM
        2. If tool calls are returned, execute them
        3. Add results to conversation and repeat
        4. Continue until LLM returns only text (no tool calls)
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Call the LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
            )

            response_message = response.choices[0].message
            print(f"LLM Response: {response_message}")

            # Check if there are tool calls
            if response_message.tool_calls:
                # Add the assistant's message with tool calls to history
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in response_message.tool_calls
                        ],
                    }
                )

                # Process ALL tool calls (not just the first one)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    print(f"Executing tool: {function_name}({function_args})")

                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)

                    print(f"Tool result: {function_response}")

                    # Add tool response to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": function_name,
                            "content": json.dumps(function_response),
                        }
                    )

                # Continue the loop to get the next response
                continue

            else:
                # No tool calls - we have our final answer
                final_content = response_message.content

                # Add the final assistant message to history
                messages.append({"role": "assistant", "content": final_content})

                print(f"\nFinal answer: {final_content}")
                return final_content

        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    # Create a ReAct agent
    agent = ReactAgent()

    # Example 1: Simple query (single tool call)
    print("=== Example 1: Single Tool Call ===")
    messages1 = [
        {"role": "system", "content": "You are a helpful AI assistant. "},
        {"role": "user", "content": "Koľko je celková plocha izby s rozmermi šírka 3,8m, dĺžka 4,5m, výška 2,6m?"},
    ]

    result1 = agent.run(messages1.copy())
    print(f"\nResult: {result1}")

    # Example 2: Complex query requiring multiple tool calls
    print("\n\n=== Example 2: Multiple Tool Calls ===")
    messages2 = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Koľko je celková cena za maľovanie izby s rozmermi šírka 3,8m, dĺžka 4,5m, výška 2,6m, ak cena za 1m2 maľovania je 16,50€?",
        },
    ]

    result2 = agent.run(messages2.copy())
    print(f"\nResult: {result2}")

    # Example 3: Sequential reasoning
    print("\n\n=== Example 3: Sequential Reasoning ===")
    messages3 = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "Mám izbu s rozmermi šírka 3,8m, dĺžka 4,5m, výška 2,6m, ak cena za 1m2 maľovania je 16,50€. O koľko by sa znížila celková cena, ak by sme sa cena za m2 zmenila na 12,34€? Porovnaj a urob sumár.",
        },
    ]

    result3 = agent.run(messages3.copy())
    print(f"\nResult: {result3}")


if __name__ == "__main__":
    main()
