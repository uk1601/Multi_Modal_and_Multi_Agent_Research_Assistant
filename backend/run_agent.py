import asyncio
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from backend.agents.agents import DEFAULT_AGENT, agents  # noqa: E402

agent = agents[DEFAULT_AGENT]


async def main() -> None:
    inputs = {"messages": [("user", "Research on the Passive investing strategy")]}
    result = await agent.ainvoke(
        inputs,
        config={"recursion_limit": 50, "thread_id": uuid4()},
    )
    print("\n====== Final Result ======\n\n\n\n\n\n")
    print(result["final_report"])
    # process_markdown_string(result["final_report"])
    result["messages"][-1].pretty_print()


    agent.get_graph().draw_png("agent_diagram.png")


asyncio.run(main())
