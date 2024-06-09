import chess, chess.svg, math
import functools, operator

from datetime import date

from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
        
board = None
board_svgs = None

num_moves = 0
move_num = 0

legal_moves = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    print("## create_agent")
    global num_moves
    
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
                         max_iterations=num_moves)

def agent_node(state, agent, name):
    try:
        print("## agent_node=" + name)
        print("## state=" + str(state))
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name=name)]}
    except Exception as e:
        print(f"An error occurred in agent_node: {e}")
        return {"messages": [HumanMessage(content=f"Error: {e}", name=name)]}

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

        global move_num
        move_num += 1
        print("## move_num=" + str(move_num))
        
        return f"Moved {piece_name} ({piece_symbol}) from "\
               f"{chess.SQUARE_NAMES[move.from_square]} to "\
               f"{chess.SQUARE_NAMES[move.to_square]}."
    except Exception as e:
        print(f"An error occurred in make_move: {e}")
        return f"Error: unable to make move {move}"
    
def create_graph():
    print("## create_graph")
    
    players = ["player_white", "player_black"]
    
    system_prompt = (
        "You are a Chess Board Proxy tasked with managing a game of chess "
        "between player_white and player_black. player_white makes the first move, "
        "then the players take turns."
    )

    #options = ["FINISH"] + players
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
    
    llm_1 = ChatOpenAI(model="gpt-4o")
    llm_2 = ChatOpenAI(model="gpt-4o")
    llm_3 = ChatOpenAI(model="gpt-4o")
    
    supervisor_chain = (
        prompt
        | llm_1.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    player_white_agent = create_agent(llm_2, [get_legal_moves, make_move], system_prompt=
                                     "You are a chess Grandmaster and you play as white. "
                                     "1. First call get_legal_moves(), to get a list of legal moves. "
                                     "2. Then call make_move(move) to make a move. ONLY make a move in the list returned by step 1.")
                                     #"3. Finally analyze the move in format: **Analysis:** move in UCI format, emoji of piece emoji, unordered list.")
    player_white_node = functools.partial(agent_node, agent=player_white_agent, name="player_white")

    player_black_agent = create_agent(llm_3, [get_legal_moves, make_move], system_prompt=
                                     "You are a chess Grandmaster and you play as black. "
                                     "1. First call get_legal_moves(), to get a list of legal moves. "
                                     "2. Then call make_move(move) to make a move. ONLY make a move in the list returned by step 1.")
                                     #"3. Finally analyze the move in format: **Analysis:** move in UCI format, emoji of piece emoji, unordered list.")
    player_black_node = functools.partial(agent_node, agent=player_black_agent, name="player_black")
    
    graph = StateGraph(AgentState)
    graph.add_node("player_white", player_white_node)
    graph.add_node("player_black", player_black_node)
    graph.add_node("chess_board_proxy", supervisor_chain)

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
    print("#### should_continue")
    global move_num, num_moves, legal_moves
    if move_num == num_moves:
        print("False (move_num == num_moves)")
        return END # max moves reached
    if not legal_moves:
        print("False (not legal_moves)")
        return END # checkmate or stalemate
    print("True")
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
    
    print("## START")
    print("## num_moves=" + str(num_moves))
    
    graph = create_graph()

    result = ""
    
    try:
        config = {"recursion_limit": 500}
        
        result = graph.invoke({
            "messages": [
                HumanMessage(content="Let's play chess, player_white starts.")
            ]
        }, config=config)
    except Exception as e:
        print(f"An error occurred: {e}")

    ###
    
    result2 = ""
    num_move = 0

    """
    for message in result["messages"]:
            if message.name:
                print(f"{message.name}: {message.content}")
            else:
                print(message.content)
    """

    print("### "+ str(type(result)))
    print("### "+ str(len(result["messages"])))
    
    if "messages" in result:
        for message in result["messages"]:
            player = ""
            
            if num_move % 2 == 0:
                player = "Player Black"
            else:
                player = "Player White"
    
            if num_move > 0:
                result2 += f"**{player}, Move {num_move}**\n{message.content}\n{board_svgs[num_move - 1]}\n\n"
            
            num_move += 1
    
            if num_moves % 2 == 0 and num_move == num_moves + 1:
                break
    
    print("===")
    print(str(result))
    print("===")
    print(str(result2))
    print("===")
    
    return str(result2)
