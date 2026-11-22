# References:

# https://docs.crewai.com/introduction

import ast
import json
import os

from crewai.tools import tool

from agents.mcp.mcp_client import (
    call_mcp_tool,
    MCP_SSE_URL_CHESS_POSITION_EVALUATION,
    MCP_TOOL_CHESS_POSITION_EVALUATION
)


def _parse_position_evaluation(text: str):
    """Parse the chess MCP server's result text without executing it.

    Gradio returns the evaluation dict as JSON or as a Python literal; older versions
    of the chess server returned a (result, error) tuple.
    """
    try:
        value = json.loads(text)
    except ValueError:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

    if isinstance(value, (tuple, list)) and value:
        if len(value) > 1 and value[1]:
            raise RuntimeError(str(value[1]))
        value = value[0]

    if isinstance(value, dict) and "continuation" in value:
        return value["continuation"]

    return value


class MCPTools:
    # Chess position evaluation

    @tool("Best Move Tool")
    def best_move_tool(fen: str) -> str:
        """Get best move with continuation in UCI notation for chess position in FEN.

        Args:
            fen (str): Chess position in FEN

        Returns:
            str: Best move with continuation in UCI notation (e.g., 'f7f2 e4e5 f2f1')

        Raises:
            RuntimeError: If processing fails

        """
        print("")
        print(f"🛠️ MCPTools: best_move_tool: fen={fen}")

        try:
            mcp_url = os.getenv("MCP_SSE_URL", MCP_SSE_URL_CHESS_POSITION_EVALUATION)

            raw_result = call_mcp_tool(
                mcp_url=mcp_url,
                tool_name=MCP_TOOL_CHESS_POSITION_EVALUATION,
                arguments={"fen": fen}
            )

            result = _parse_position_evaluation(raw_result)

            print(f"🛠️ MCPTools: best_move_tool: mcp_url={mcp_url}")
            print(f"🛠️ MCPTools: best_move_tool: tool_name={MCP_TOOL_CHESS_POSITION_EVALUATION}")
            print(f"🛠️ MCPTools: best_move_tool: result={result}")

            return result
        except Exception as e:
            print(f"⚠️ MCPTools: best_move_tool: exception={str(e)}")
            raise RuntimeError(f"Processing failed: {str(e)}") from e
