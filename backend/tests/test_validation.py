import asyncio
from pprint import pprint

import aiohttp


async def test_validation():
    async with aiohttp.ClientSession() as session:
        # Test case 1: Valid configuration
        valid_config = {
            "initial_node": "start",
            "nodes": {
                "start": {
                    "messages": [
                        {"role": "system", "content": "You are an order-taking assistant..."}
                    ],
                    "functions": [
                        {
                            "type": "function",
                            "function": {
                                "name": "choose_pizza",
                                "description": "User wants to order pizza",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                },
                "choose_pizza": {
                    "messages": [{"role": "system", "content": "Handle pizza order..."}],
                    "functions": [
                        {
                            "type": "function",
                            "function": {
                                "name": "end",
                                "description": "End the order",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                },
            },
        }

        print("\nTesting valid configuration:")
        async with session.post(
            "http://localhost:8000/api/validate", json=valid_config
        ) as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                print(await response.text())
            else:
                result = await response.json()
                pprint(result)

        # Test case 2: Invalid configuration
        invalid_config = {
            "initial_node": "start",
            "nodes": {
                "start": {
                    "messages": [
                        {"role": "system", "content": "You are an order-taking assistant..."}
                    ],
                    "functions": [
                        {
                            "type": "function",
                            "function": {
                                "name": "nonexistent_node",
                                "description": "This node doesn't exist",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                }
            },
        }

        print("\nTesting invalid configuration:")
        async with session.post(
            "http://localhost:8000/api/validate", json=invalid_config
        ) as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                print(await response.text())
            else:
                result = await response.json()
                pprint(result)


if __name__ == "__main__":
    asyncio.run(test_validation())
