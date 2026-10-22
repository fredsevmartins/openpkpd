import asyncio
import os
from pathlib import Path
from mcp import Client, StdioServerParameters


def test_stdio_server_discovers_four_tools():
    project = Path(__file__).resolve().parents[1]
    params = StdioServerParameters(
        command=str(project / ".venv" / "bin" / "openpkpd-mcp"),
        args=[], cwd=project,
        env={**os.environ, "PYTHONPATH": str(project / "src")},
    )

    async def check():
        async with Client(params) as client:
            result = await client.list_tools()
            return {tool.name for tool in result.tools}

    names = asyncio.run(check())
    assert names == {
        "simulate_pk", "calculate_nca", "simulate_pd",
        "simulate_multiple_doses",
    }
