import chess, chess.svg, math
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

def check_made_move(msg):
    global made_move
    
    if made_move:
        made_move = False
        return True
    else:
        return False

def get_num_turns(num_moves):
    # Each turn includes two moves (one by each player)
    # The first move by player black kicks off the chat
    # The first move by player white starts the game 

    num_turns = math.ceil(num_moves / 2)
    
    if num_moves % 2 == 0:
        num_turns += 1
        
    return num_turns
    
def run_multi_agent(llm_white, llm_black, num_moves):   
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
        system_message="You are a chess Grandmaster and you play as white. "
        "First call get_legal_moves(), to get a list of legal moves. "
        "Then call make_move(move) to make a move. "
        "After a move is made, analyze the move in 3 bullet points. Respond in format **Analysis:** move from/to, unordered list.",
        llm_config=llm_config_white,
    )
    
    player_black = ConversableAgent(
        name="Player Black",
        system_message="You are a chess Grandmaster and you play as black. "
        "First call get_legal_moves(), to get a list of legal moves. "
        "Then call make_move(move) to make a move. "
        "After a move is made, analyze the move in 3 bullet points. Respond in format **Analysis:** move from/to, unordered list.",
        llm_config=llm_config_black,
    )
    
    for caller in [player_white, player_black]:
        register_function(
            get_legal_moves,
            caller=caller,
            executor=board_proxy,
            name="get_legal_moves",
            description="Call this tool to get legal moves.",
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
                "silent": False,
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
                "silent": False,
            }
        ],
    )

    chat_result = None
    chat_history = []
    
    try:
        chat_result = player_black.initiate_chat(
            player_white,
            message="Let's play chess!",
            max_turns=get_num_turns(num_moves),
            verbose=True
        )
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if chat_result != None:
            chat_history = chat_result.chat_history
    
    result = ""
    num_move = 0

    for chat in chat_history:
        player = ""
        
        if num_move % 2 == 0:
            player = "Player Black"
        else:
            player = "Player White"

        if num_move > 0:
            result += f"**{player}, Move {num_move}**<br>{chat.get('content')}<br>{board_svgs[num_move - 1]}<br><br>"
        
        num_move += 1

        if num_moves % 2 == 0 and num_move == num_moves + 1:
            break

    #print("===")
    #print(result)
    #print("===")
    
    return result
