import asyncio
from openai import AsyncOpenAI

async def run():
    c = AsyncOpenAI(api_key='x')
    try:
        await c.chat.completions.create(
            model='x',
            messages=[{
                'role': 'system',
                'content': [{
                    'type': 'text',
                    'text': 'system prompt',
                    'cache_control': {'type': 'ephemeral'}
                }]
            }]
        )
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(run())
