import chess, chess.svg
from autogen import ConversableAgent, register_function
from typing_extensions import Annotated

made_move = False

board = chess.Board()
board_svgs = []

def get_legal_moves() -> Annotated[str, "A list of legal moves in UCI format"]:
    return "Possible moves are: " + ",".join(
        [str(move) for move in board.legal_moves]
    )

def make_move(move: Annotated[str, "A move in UCI format."]) -> Annotated[str, "Result of the move."]:
    move = chess.Move.from_uci(move)
    board.push_uci(str(move))
    global made_move
    made_move = True

    svg = chess.svg.board(
        board,
        arrows=[(move.from_square, move.to_square)],
        fill={move.from_square: "gray"},
        size=250
    )

    piece = board.piece_at(move.to_square)
    piece_symbol = piece.unicode_symbol()
    piece_name = (
        chess.piece_name(piece.piece_type).capitalize()
        if piece_symbol.isupper()
        else chess.piece_name(piece.piece_type)
    )

    board_svgs.append(svg)

    return f"Moved {piece_name} ({piece_symbol}) from "\
    f"{chess.SQUARE_NAMES[move.from_square]} to "\
    f"{chess.SQUARE_NAMES[move.to_square]}."

def check_made_move(msg):
    global made_move
    if made_move:
        made_move = False
        return True
    else:
        return False

def run_multi_agent(llm_white, llm_black, num_turns):   
    llm_config_white = {"model": llm_white}
    llm_config_black = {"model": llm_black}
    
    board_proxy = ConversableAgent(
        name="Board Proxy",
        llm_config=False,
        is_termination_msg=check_made_move,
        default_auto_reply="Please make a move.",
        human_input_mode="NEVER",
    )
    
    player_white = ConversableAgent(
        name="Player White",
        system_message="You are a chess player and you play as white. "
        "First call get_legal_moves(), to get a list of legal moves. "
        "Then call make_move(move) to make a move. "
        "After a move is made, analyze the move in 3 bullet points. Respond in format **Analysis:** move from/to, unordered list, your turn player black. "
        "Then continue playing.",
        llm_config=llm_config_white,
    )
    
    player_black = ConversableAgent(
        name="Player Black",
        system_message="You are a chess player and you play as black. "
        "First call get_legal_moves(), to get a list of legal moves. "
        "Then call make_move(move) to make a move. "
        "After a move is made, analyze the move in 3 bullet points. Respond in format **Analysis:** move from/to, unordered list, your turn player white. "
        "Then continue playing.",
        llm_config=llm_config_black,
    )
    
    for caller in [player_white, player_black]:
        register_function(
            get_legal_moves,
            caller=caller,
            executor=board_proxy,
            name="get_legal_moves",
            description="Get legal moves.",
        )
    
        register_function(
            make_move,
            caller=caller,
            executor=board_proxy,
            name="make_move",
            description="Call this tool to make a move.",
        )
    
    player_white.register_nested_chats(
        trigger=player_black,
        chat_queue=[
            {
                "sender": board_proxy,
                "recipient": player_white,
                "summary_method": "last_msg",
                "silent": True,
            }
        ],
    )
    
    player_black.register_nested_chats(
        trigger=player_white,
        chat_queue=[
            {
                "sender": board_proxy,
                "recipient": player_black,
                "summary_method": "last_msg",
                "silent": True,
            }
        ],
    )
       
    chat_result = player_black.initiate_chat(
        player_white,
        message="Let's play chess!",
        max_turns=num_turns,
        verbose=False
    )

    chat_history = chat_result.chat_history
    
    result = ""
    turn_num = 0

    for chat in chat_history:
        player = ""
        
        if turn_num % 2 == 0:
            player = "Player Black"
        else:
            player = "Player White"

        if turn_num > 0:
            result += f"**{player}, Move {turn_num}**\n{chat.get('content')}\n{board_svgs[turn_num - 1]}\n\n"
        
        turn_num += 1

    result = result.rstrip("\n\n")

    print("###")
    print(result)
    print("###")
    
    return result
