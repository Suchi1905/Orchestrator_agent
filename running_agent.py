import asyncio
import json
import logging
from agent import (
    orchestrate_request,
)

logging.getLogger("google_genai.types").setLevel(logging.ERROR)


async def main():
    print("Agent started (Sarvam Orchestrator mode)\n")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
            
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue

        response = await orchestrate_request(user_input)
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
