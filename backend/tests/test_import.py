import asyncio
from pprint import pprint

import aiohttp


async def test_import():
    async with aiohttp.ClientSession() as session:
        # Your example flow configuration
        flow_config = {
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
                }
            },
        }

        async with session.post("http://localhost:8000/api/import", json=flow_config) as response:
            result = await response.json()
            pprint(result)


if __name__ == "__main__":
    asyncio.run(test_import())
