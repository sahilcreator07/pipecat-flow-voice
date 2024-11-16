import asyncio
from pprint import pprint

import aiohttp


async def test_templates():
    async with aiohttp.ClientSession() as session:
        print("\nFetching node templates:")
        async with session.get("http://localhost:8000/api/templates") as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                print(await response.text())
            else:
                result = await response.json()
                pprint(result)


if __name__ == "__main__":
    asyncio.run(test_templates())
