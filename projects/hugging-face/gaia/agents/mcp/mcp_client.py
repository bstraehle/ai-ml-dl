# References:

# https://github.com/modelcontextprotocol/python-sdk

import asyncio
import os

import anyio
from mcp.client.session import ClientSession, DEFAULT_CLIENT_INFO
from mcp.client.sse import sse_client

# Chess position evaluation

SPACE_AUTHOR_NAME = os.getenv("SPACE_AUTHOR_NAME", "bstraehle")

MCP_SSE_URL_CHESS_POSITION_EVALUATION = f"https://{SPACE_AUTHOR_NAME}-chess.hf.space/gradio_api/mcp/sse"
MCP_TOOL_CHESS_POSITION_EVALUATION    = "chess_position_evaluation"


async def call_mcp_tool_async(mcp_url: str, tool_name: str, arguments: dict, timeout: int = 30) -> str:
    """Call an MCP tool asynchronously.

    Args:
        mcp_url (str): The MCP SSE URL
        tool_name (str): The name of the MCP tool to call
        arguments (dict): The arguments to pass to the tool
        timeout (int): Timeout in seconds (default: 30)

    Returns:
        str: The tool result as a string

    Raises:
        RuntimeError: If the tool returns an error
        Exception: If the MCP call fails
    """
    with anyio.fail_after(timeout):
        async with sse_client(mcp_url) as (read, write):
            async with ClientSession(read, write, client_info=DEFAULT_CLIENT_INFO) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)

    text = " ".join(item.text for item in result.content if hasattr(item, "text"))

    # Raised outside the client's task groups, which would wrap it in an ExceptionGroup
    if result.isError:
        raise RuntimeError(text or f"MCP tool {tool_name} failed")

    return text or str(result)


def call_mcp_tool(mcp_url: str, tool_name: str, arguments: dict, timeout: int = 30) -> str:
    """Call an MCP tool synchronously.

    Args:
        mcp_url (str): The MCP SSE URL
        tool_name (str): The name of the MCP tool to call
        arguments (dict): The arguments to pass to the tool
        timeout (int): Timeout in seconds (default: 30)

    Returns:
        str: The tool result as a string

    Raises:
        RuntimeError: If the tool returns an error
        Exception: If the MCP call fails
    """
    return asyncio.run(call_mcp_tool_async(mcp_url, tool_name, arguments, timeout))
