import chess, chess.svg
import functools, operator

from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
        
board = None
board_svgs = None

move_num = 0
num_moves = 0

legal_moves = ""

@tool
def get_legal_moves() -> Annotated[str, "A list of legal moves in UCI format"]:
    """Returns a list of legal moves in UCI format. 
       The input should always be an empty string, 
       and this function will always return legal moves in UCI format."""
    try:
        print("## get_legal_moves")
        global legal_moves
        legal_moves = ",".join([str(move) for move in board.legal_moves])
        return legal_moves
    except Exception as e:
        print(f"An error occurred in get_legal_moves: {e}")
        return "Error: unable to get legal moves"

@tool
def make_move(move: Annotated[str, "A move in UCI format."]) -> Annotated[str, "Result of the move."]:
    """Makes a move. 
       The input should always be a move in UCI format, 
       and this function will always return the result of the move in UCI format."""
    try:
        print("## make_move")
        move = chess.Move.from_uci(move)
        board.push_uci(str(move))

        global move_num
        move_num += 1
        print("## move_num=" + str(move_num))

        board_svgs.append(chess.svg.board(
            board,
            arrows=[(move.from_square, move.to_square)],
            fill={move.from_square: "gray"},
            size=250
        ))

        piece = board.piece_at(move.to_square)
        piece_symbol = piece.unicode_symbol()
        piece_name = (
            chess.piece_name(piece.piece_type).capitalize()
            if piece_symbol.isupper()
            else chess.piece_name(piece.piece_type)
        )
        
        return f"Moved {piece_name} ({piece_symbol}) from "\
               f"{chess.SQUARE_NAMES[move.from_square]} to "\
               f"{chess.SQUARE_NAMES[move.to_square]}."
    except Exception as e:
        print(f"An error occurred in make_move: {e}")
        return f"Error: unable to make move {move}"
        
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, 
                         tools=tools,
                         handle_parsing_errors=True,
                         return_intermediate_steps=True,
                         verbose=True,
                         max_iterations=5) # get_legal_moves & make_move

def agent_node(state, agent, name):
    try:
        print("## agent_node=" + name)
        result = agent.invoke(state)
        print("## result=" + str(result))
        print("## result['output']=" + result["output"])
        return {
            "messages": [HumanMessage(content=result["output"], name=name)]
        }
    except Exception as e:
        print(f"An error occurred in agent_node: {e}")
        return {"messages": [HumanMessage(content=f"Error: {e}", name=name)]}
   
def create_graph():
    players = ["player_white", "player_black"]
    
    system_prompt = (
        "You are a Chess Board Proxy tasked with managing a game of chess "
        "between player_white and player_black. player_white makes the first move, "
        "then the players take turns."
    )

    options = players

    function_def = {
        "name": "route",
        "description": "Select the next player.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "If player_white made a move, player_black must make the next move. "
                "If player_black made a move, player_white must make the next move. "
                "Select one of: {options}.",
            ),
        ]
    ).partial(options=str(options), members=", ".join(players), verbose=True)
    
    llm_chess_board_proxy = ChatOpenAI(model="gpt-4o")
    llm_player_white = ChatOpenAI(model="gpt-4o")
    llm_player_black = ChatOpenAI(model="gpt-4o")
    
    supervisor_chain = (
        prompt
        | llm_chess_board_proxy.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    player_white_agent = create_agent(llm_player_white, [get_legal_moves, make_move], system_prompt=
                                     "You are a chess Grandmaster and you play as white. "
                                     "First call get_legal_moves(), to get a list of legal moves. "
                                     "Then study the moves and call make_move(move) to make the best move. "
                                     "Finally analyze the move in format: **Analysis:** move in UCI format, emoji of piece emoji, unordered list of 3 items.")
    player_white_node = functools.partial(agent_node, agent=player_white_agent, name="player_white")

    player_black_agent = create_agent(llm_player_black, [get_legal_moves, make_move], system_prompt=
                                     "You are a chess Grandmaster and you play as black. "
                                     "First call get_legal_moves(), to get a list of UCI legal moves. "
                                     "Then study the moves and call make_move(move) to make the best move. "
                                     "Finally analyze the move in format: **Analysis:** move in UCI format, emoji of piece emoji, unordered list of 3 items.")
    player_black_node = functools.partial(agent_node, agent=player_black_agent, name="player_black")
    
    graph = StateGraph(AgentState)
    
    graph.add_node("chess_board_proxy", supervisor_chain)
    graph.add_node("player_white", player_white_node)
    graph.add_node("player_black", player_black_node)

    #graph.add_edge("chess_board_proxy", "player_white")
    #graph.add_edge("chess_board_proxy", "player_black")
    
    graph.add_conditional_edges(
        "player_white", 
        should_continue, 
        {"chess_board_proxy": "chess_board_proxy", END: END}
    )

    graph.add_conditional_edges(
        "player_black", 
        should_continue, 
        {"chess_board_proxy": "chess_board_proxy", END: END}
    )
    
    conditional_map = {k: k for k in players}
    conditional_map["END"] = END
    
    graph.add_conditional_edges("chess_board_proxy", lambda x: x["next"], conditional_map)
    
    graph.set_entry_point("chess_board_proxy")
    
    return graph.compile()

def should_continue(state):
    global move_num, num_moves, legal_moves
    
    if move_num == num_moves:
        return END # max moves reached
    
    if not legal_moves:
        return END # checkmate or stalemate
    
    return "chess_board_proxy"

def initialize():
    global board, board_svgs, num_moves, move_num, legal_moves

    board = chess.Board()
    board_svgs = []

    num_moves = 0
    move_num = 0
    
    legal_moves = ""

def run_multi_agent(moves_num):
    initialize()

    global num_moves

    num_moves = moves_num
    
    graph = create_graph()

    result = ""
    
    try:
        config = {"recursion_limit": 100}
        
        result = graph.invoke({
            "messages": [
                HumanMessage(content="Let's play chess, player_white starts.")
            ]
        }, config=config)
    except Exception as e:
        print(f"An error occurred: {e}")

    result_md = ""
    num_move = 0

    if "messages" in result:
        for message in result["messages"]:
            player = ""
            
            if num_move % 2 == 0:
                player = "Player Black"
            else:
                player = "Player White"
    
            if num_move > 0:
                result_md += f"**{player}, Move {num_move}**\n{message.content}\n{board_svgs[num_move - 1]}\n\n"
            
            num_move += 1
    
            if num_moves % 2 == 0 and num_move == num_moves + 1:
                break
    
    print("===")
    print(str(result_md))
    print("===")
    
    return result_md
