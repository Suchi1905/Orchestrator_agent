import asyncio
import json
import logging
from google.adk.runners import InMemoryRunner
from google.genai import types
from agent import orchestrator_agent


def extract_json_text(response):
    if not response:
        return response

    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


logging.getLogger("google_genai.types").setLevel(logging.ERROR)


async def main():
    runner = InMemoryRunner(agent=orchestrator_agent)
    user_id = "local_user"
    session_id = "local_session"

    await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )

    print("Agent started (ADK mode)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        response = None
        responding_agent = None
        transfer_path = []
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            ),
        ):
            if event.actions.transfer_to_agent:
                transfer_path.append(event.actions.transfer_to_agent)
            if event.is_final_response() and event.content and event.content.parts:
                response = "\n".join(part.text for part in event.content.parts if part.text)
                responding_agent = event.author

        response = extract_json_text(response)

        try:
            parsed = json.loads(response)
            print(f"\nIntent: {parsed['intent']}")
            if responding_agent:
                print(f"Handled By: {responding_agent}")
            if transfer_path:
                print(f"Route: {' -> '.join(transfer_path)}")
            print(f"Response: {parsed['response']}\n")
        except Exception:
            print("\nThe agent returned a response : I could not read clearly.\n")
            if responding_agent:
                print(f"Handled By: {responding_agent}")
            if transfer_path:
                print(f"Route: {' -> '.join(transfer_path)}")
            if response:
                print(f"Response: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
