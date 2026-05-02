from TicTacToe import TicTacToeEnv, QAgent
import numpy as np
from typing import Optional

def test_colors():
    env = TicTacToeEnv()
    agent = QAgent()
    
    q_table = {
        (env.get_state(), 0): -1.0,
        (env.get_state(), 1): -0.9,
        (env.get_state(), 2): -0.8,
        (env.get_state(), 3): -0.5,
        (env.get_state(), 4): 0.0,
        (env.get_state(), 5): 0.5,
        (env.get_state(), 6): 0.8,
        (env.get_state(), 7): 0.9,
        (env.get_state(), 8): 1.0
    }
    agent.q_table = q_table
    
    state = env.get_state()
    available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    color_codes = [None] * 9
    for a in available_actions:
        q_val = agent.get_q_value(state, a)
        color_codes[a] = agent.get_color_code(q_val)
        print(f"Action {a}, Q-value {q_val}, Color {color_codes[a]}")
    
    env.print_board(color_codes)

def get_max_size_of_q_table():
    seen_state_actions = set()

    def check_winner(board : np.ndarray) -> Optional[int]:
        # Check rows
        for i in range(3):
            if np.all(board[i*3:i*3+3] == 1): return 1
            if np.all(board[i*3:i*3+3] == -1): return -1
        # Check columns
        for i in range(3):
            if (board[i+0] == board[i+3] == board[i+6] == 1): return 1
            if (board[i+0] == board[i+3] == board[i+6] == -1): return -1
        # Check diagonals
        if (board[0] == board[4] == board[8] == 1): return 1
        if (board[2] == board[4] == board[6] == 1): return 1
        if (board[0] == board[4] == board[8] == -1): return -1
        if (board[2] == board[4] == board[6] == -1): return -1
        if not np.any(board == 0):
            return 0 # Draw (no zeros left)
        return None # Game is not yet over
    
    def recursion(board: np.ndarray, player: int):
        state_str = str(board.astype(int))
        available_actions = [i for i, x in enumerate(board) if x == 0]
        
        for action in available_actions:
            # Record this state-action pair
            seen_state_actions.add((state_str, action))
            
            # Try the move
            board[action] = player
            winner = check_winner(board)
            
            # If game continues, recurse
            if winner is None:
                recursion(board, player * -1)
            
            board[action] = 0 # BACKTRACK: Undo the move for the next iteration
    
    board = np.zeros(9)
    recursion(board, 1) # Start with player 1
    print(f"Total unique (state, action) pairs: {len(seen_state_actions)}")

if __name__ == "__main__":
    # test_colors()
    get_max_size_of_q_table()

