import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .agents.editor_agent import EditorAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv()
    model = os.getenv("JARVIS_AGENT_MODEL", "gpt-5-mini")
    agent = EditorAgent(model=model)
    logging.info("Jarvis local office REPL using model: %s", model)

    print("Jarvis local office REPL")
    print("Type a request and press Enter. Type 'exit' to quit. Type '/reset' to clear memory.\n")

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            return
        if query.lower() == "/reset":
            agent.clear_memory()
            print("Jarvis> Memory cleared.\n")
            continue

        try:
            answer = await agent.handle_query(query)
            print(f"Jarvis> {answer}\n")
        except Exception as exc:
            print(f"Jarvis> Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
