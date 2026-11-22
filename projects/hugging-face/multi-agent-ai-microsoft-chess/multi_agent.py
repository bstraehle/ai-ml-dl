from collections.abc import AsyncIterator
from typing import Annotated

import chess
import chess.svg
from agent_framework import Agent, FunctionInvocationConfiguration, tool
from agent_framework.openai import OpenAIChatClient
from pydantic import Field

MAX_ATTEMPTS = 3  # per move, if a player doesn't call make_move
MAX_TOOL_ITERATIONS = 10
BOARD_SIZE = 600

INSTRUCTIONS = (
    "You are a chess Grandmaster and you play as {color}. "
    "First call get_legal_moves() to get a list of legal moves. "
    "Then study the returned moves and call make_move(move) to make the best move. "
    "Finally analyze the move: **Analysis:** move in UCI format, emoji of piece, unordered list of 3 items."
)


class Game:
    def __init__(self):
        self.board = chess.Board()
        self.last_move: chess.Move | None = None

    def get_legal_moves(self) -> str:
        return "Legal moves are: " + ",".join(str(move) for move in self.board.legal_moves) + "."

    def make_move(self, move: Annotated[str, Field(description="A move in UCI format, e.g. e2e4.")]) -> str:
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

    def start_turn(self) -> None:
        self.last_move = None

    def svg(self) -> str:
        move = self.last_move
        return chess.svg.board(
            self.board,
            arrows=[(move.from_square, move.to_square)] if move else [],
            fill={move.from_square: "gray"} if move else {},
            size=BOARD_SIZE,
        )


def create_player(game: Game, model: str, color: str, openai_api_key: str) -> Agent:
    client = OpenAIChatClient(
        model=model,
        api_key=openai_api_key,
        function_invocation_configuration=FunctionInvocationConfiguration(max_iterations=MAX_TOOL_ITERATIONS),
    )
    return Agent(
        client=client,
        name=f"Player {color.capitalize()}",
        instructions=INSTRUCTIONS.format(color=color),
        tools=[
            tool(game.get_legal_moves, name="get_legal_moves", description="Call this tool to get legal moves."),
            tool(game.make_move, name="make_move", description="Call this tool to make a move."),
        ],
    )


async def play(openai_api_key: str, llm_white: str, llm_black: str, num_moves: int) -> AsyncIterator[str]:
    """Play num_moves half-moves between two LLM players, yielding each move as Markdown with an SVG board."""
    game = Game()
    players = {
        chess.WHITE: create_player(game, llm_white, "white", openai_api_key),
        chess.BLACK: create_player(game, llm_black, "black", openai_api_key),
    }
    sessions = {color: player.create_session() for color, player in players.items()}
    opponent_analysis = ""

    for number in range(1, num_moves + 1):
        if game.board.is_game_over():
            yield f"**Game over:** {game.board.result()} ({game.board.outcome().termination.name.replace('_', ' ').lower()})\n\n"
            return

        color = game.board.turn
        player = players[color]
        game.start_turn()
        prompt = "Let's play chess! Make your move." if number == 1 else f"Your opponent played. {opponent_analysis}\nMake your move."

        for _ in range(MAX_ATTEMPTS):
            response = await player.run(prompt, session=sessions[color])
            if game.last_move is not None:
                break
            prompt = "You didn't make a move. Call get_legal_moves(), then make_move(move)."
        else:
            raise RuntimeError(f"{player.name} didn't make a move after {MAX_ATTEMPTS} attempts.")

        opponent_analysis = response.text
        yield f"**{player.name}, Move {number}**\n{response.text}\n{game.svg()}\n\n"
