import operator
from collections.abc import AsyncIterator
from typing import Annotated, TypedDict

import chess
import chess.svg
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

MAX_ATTEMPTS = 3  # per move, if a player doesn't call make_move
AGENT_RECURSION_LIMIT = 20  # model and tool steps per attempt
BOARD_SIZE = 600

INSTRUCTIONS = (
    "You are a chess Grandmaster and you play as {color}. "
    "First call get_legal_moves() to get a list of legal moves. "
    "Then study the returned moves and call make_move(move) to make the best move. "
    "Finally analyze the move: **Analysis:** move in UCI format, emoji of piece, unordered list of 3 items."
)

PLAYERS = {chess.WHITE: "player_white", chess.BLACK: "player_black"}


class GameState(TypedDict):
    moves: Annotated[list[str], operator.add]  # Markdown per move
    last_analysis: str


class Game:
    def __init__(self):
        self.board = chess.Board()
        self.last_move: chess.Move | None = None

    def tools(self):
        @tool
        def get_legal_moves() -> str:
            """Returns the list of legal moves in UCI format."""
            return "Legal moves are: " + ",".join(str(move) for move in self.board.legal_moves) + "."

        @tool
        def make_move(move: str) -> str:
            """Makes a move. The input must be one legal move in UCI format, e.g. e2e4."""
            try:
                parsed = chess.Move.from_uci(move.strip())
            except ValueError:
                return f"Invalid move {move!r}: use UCI format, e.g. e2e4."
            if self.last_move is not None:
                return "You already made your move this turn."
            if parsed not in self.board.legal_moves:
                return f"Illegal move {move}. Call get_legal_moves() and pick one of those."
            self.board.push(parsed)
            self.last_move = parsed
            piece = self.board.piece_at(parsed.to_square)
            name = chess.piece_name(piece.piece_type)
            return (
                f"Moved {name.capitalize() if piece.color == chess.WHITE else name} ({piece.unicode_symbol()}) "
                f"from {chess.SQUARE_NAMES[parsed.from_square]} to {chess.SQUARE_NAMES[parsed.to_square]}."
            )

        return [get_legal_moves, make_move]

    def svg(self) -> str:
        move = self.last_move
        return chess.svg.board(
            self.board,
            arrows=[(move.from_square, move.to_square)] if move else [],
            fill={move.from_square: "gray"} if move else {},
            size=BOARD_SIZE,
        )


def create_graph(game: Game, openai_api_key: str, llm_white: str, llm_black: str, num_moves: int):
    tools = game.tools()
    agents = {
        color: create_agent(
            ChatOpenAI(model=model, api_key=openai_api_key, use_responses_api=True),
            tools=tools,
            system_prompt=INSTRUCTIONS.format(color="white" if color == chess.WHITE else "black"),
        )
        for color, model in ((chess.WHITE, llm_white), (chess.BLACK, llm_black))
    }

    def player_node(color: chess.Color):
        name = "Player White" if color == chess.WHITE else "Player Black"

        async def node(state: GameState) -> dict:
            game.last_move = None
            number = len(state["moves"]) + 1
            prompt = "Let's play chess! Make your move." if number == 1 else f"Your opponent played. {state['last_analysis']}\nMake your move."
            for _ in range(MAX_ATTEMPTS):
                result = await agents[color].ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]}, {"recursion_limit": AGENT_RECURSION_LIMIT}
                )
                if game.last_move is not None:
                    break
                prompt = "You didn't make a move. Call get_legal_moves(), then make_move(move)."
            else:
                raise RuntimeError(f"{name} didn't make a move after {MAX_ATTEMPTS} attempts.")
            analysis = result["messages"][-1].text
            return {"moves": [f"**{name}, Move {number}**\n{analysis}\n{game.svg()}\n\n"], "last_analysis": analysis}

        return node

    def next_player(state: GameState) -> str:
        if len(state["moves"]) >= num_moves or game.board.is_game_over():
            return END
        return PLAYERS[game.board.turn]

    graph = StateGraph(GameState)
    for color, node_name in PLAYERS.items():
        graph.add_node(node_name, player_node(color))
        graph.add_conditional_edges(node_name, next_player, [*PLAYERS.values(), END])
    graph.add_edge(START, PLAYERS[chess.WHITE])
    return graph.compile()


async def play(openai_api_key: str, llm_white: str, llm_black: str, num_moves: int) -> AsyncIterator[str]:
    """Play num_moves half-moves between two LLM players, yielding each move as Markdown with an SVG board."""
    game = Game()
    graph = create_graph(game, openai_api_key, llm_white, llm_black, num_moves)
    async for update in graph.astream({"moves": [], "last_analysis": ""}, {"recursion_limit": num_moves + 5}, stream_mode="updates"):
        for node_update in update.values():
            yield "".join(node_update.get("moves", []))

    if game.board.is_game_over():
        outcome = game.board.outcome()
        yield f"**Game over:** {game.board.result()} ({outcome.termination.name.replace('_', ' ').lower()})\n\n"
