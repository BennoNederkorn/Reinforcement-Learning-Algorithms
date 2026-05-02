import numpy as np
import random
import pickle
import os
import time
from typing import List, Optional, Tuple, Dict

class TicTacToeEnv:
    """
    A simple Tic-Tac-Toe environment.
    The board is represented as a 1D numpy array of size 9.
    Players are represented as 1 (X) and -1 (O). 0 represents an empty cell.
    """

    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros(9)
        self.done: bool = False

    def reset(self) -> str:
        """Resets the environment to the initial state."""
        self.board = np.zeros(9)
        self.done = False
        return self.get_state()

    def get_state(self) -> str:
        """Returns the current board state as a string representation."""
        return str(self.board.astype(int))

    def step(self, action: int, player: int) -> Tuple[str, float, bool]:
        """
        Applies an action to the board for a given player.
        
        Args:
            action: The index (0-8) where the player wants to move.
            player: The player identifier (1 or -1).
            
        Returns:
            A tuple of (next_state, reward, done).
        """
        self.board[action] = player
        
        winner = self.check_winner()
        if winner is not None:
            self.done = True

        reward: float = 0.0
        if winner == player:
            reward = 1.0
        elif winner == 0:  # Draw
            reward = 0.0
        else:
            # Loss or game continues
            reward = 0.0
            
        return self.get_state(), reward, self.done

    def check_winner(self) -> Optional[int]:
        """
        Checks if there is a winner or a draw.
        
        Returns:
            1 if X wins, -1 if O wins, 0 if it's a draw, 
            or None if the game is still ongoing.
        """
        # Define winning combinations (rows, columns, diagonals)
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for combo in wins:
            if self.board[combo[0]] != 0 and \
               self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]]:
                return int(self.board[combo[0]])

        if not np.any(self.board == 0):
            return 0  # Draw
            
        return None

    def print_board(self, color_codes: Optional[List[Optional[List[int]]]] = None) -> None:
        """
        Prints the board to the terminal.
        
        Args:
            color_codes: Optional list of RGB colors for each cell, 
                         used to visualize Q-values.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

        def format_cell(i: int) -> str:
            if self.board[i] == 1:
                return " X "
            elif self.board[i] == -1:
                return " O "
            else:
                if color_codes and i < len(color_codes) and color_codes[i]:
                    r, g, b = color_codes[i]
                    # Black text (30) on colored background (48;2;r;g;b)
                    return f"\033[48;2;{r};{g};{b}m\033[30m {i+1} \033[0m"
                else:
                    return f" {i+1} "

        print(format_cell(6) + "|" + format_cell(7) + "|" + format_cell(8))
        print("-----------")
        print(format_cell(3) + "|" + format_cell(4) + "|" + format_cell(5))
        print("-----------")
        print(format_cell(0) + "|" + format_cell(1) + "|" + format_cell(2))

class QAgent:
    """
    A Q-learning agent that learns to play Tic-Tac-Toe.
    Uses a minimax-style update rule for zero-sum self-play.
    """
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2) -> None:
        self.q_table: Dict[Tuple[str, int], float] = {}
        self.alpha: float = alpha      # Learning rate
        self.gamma: float = gamma      # Discount factor
        self.epsilon: float = epsilon  # Exploration rate

    def get_q_value(self, state: str, action: int) -> float:
        """Returns the Q-value for a given state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: str, available_actions: List[int]) -> int:
        """
        Chooses an action using an epsilon-greedy strategy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        
        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state: str, action: int, reward: float, 
              next_state: str, next_available_actions: List[int]) -> None:
        """
        Updates the Q-value using the minimax Q-learning rule.
        V(s, a) = V(s, a) + alpha * (reward - gamma * max(V(s', a')) - V(s, a))
        The subtraction of the next max accounts for the opponent's optimal play.
        """
        old_v = self.get_q_value(state, action)
        
        next_max = 0.0
        if next_available_actions:
            next_max = max([self.get_q_value(next_state, a) for a in next_available_actions])
            
        new_v = old_v + self.alpha * (reward - self.gamma * next_max - old_v)
        self.q_table[(state, action)] = new_v

    def save_q_table(self, filename: Optional[str] = None) -> None:
        """Persists the Q-table to a file."""
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "q_table.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename: Optional[str] = None) -> bool:
        """Loads the Q-table from a file."""
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "q_table.pkl")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            return True
        return False
    
    def get_color_code(self, q_value: float) -> List[int]:
        """
        Generates a color code (RGB) based on a Q-value for visualization.
        Maps values roughly from -0.5 (Red) to 0.5 (Green).
        """
        normalized = max(-0.5, min(0.5, q_value)) + 0.5
        
        r = int(255 * (1 - normalized))
        g = int(255 * normalized)
        b = 0
        return [r, g, b]

def training_main(agent: QAgent, env: TicTacToeEnv, 
                  training_episodes: int, q_file: str) -> None:
    """
    Trains the agent through self-play.
    """
    if agent.load_q_table(q_file):
        print(f"Loaded Q-table from {q_file}. Skipping training.")
    else:
        print("Start Training...")
        for episode in range(training_episodes):
            state = env.reset()
            player = -1
            done = False
            while not done:
                available_actions = [i for i, x in enumerate(env.board) if x == 0]
                
                prev_state = state
                action = agent.choose_action(prev_state, available_actions)
                player = player * -1
                state, reward, done = env.step(action, player)
                
                next_available_actions = [i for i, x in enumerate(env.board) if x == 0]
                agent.learn(prev_state, action, reward, state, next_available_actions)

            if episode % 1000 == 0:
                print(f"{episode}/{training_episodes} episodes done. Q-table size: {len(agent.q_table)}/16167")

        print("Finished Training")
        agent.save_q_table(q_file)
        print(f"Saved Q-table to {q_file}")

def should_the_player_start() -> bool:
    """
    Prompts the user to decide who starts the game.
    
    Returns:
        True if the user starts, False if the agent starts.
    """
    print("Who should start? Agent=1, You=2")
    try:
        line = input("Your choice: ")
        if line == '1':
            print("Agent starts!")
            return False
        elif line == '2':
            print("You start!")
            return True
        else:
            print("Invalid choice, you start!")
            return True
    except (ValueError, EOFError):
        print("Invalid choice, you start!")
        return True
    finally:
        time.sleep(1)

def game_loop(player_starts: bool, agent: QAgent, env: TicTacToeEnv) -> None:
    """
    Main game loop for playing against the agent.
    """
    state = env.get_state()
    player = -1
    
    # If agent starts, it takes the first move
    if not player_starts:
        available_actions = [i for i, x in enumerate(env.board) if x == 0]
        action_agent = agent.choose_action(state, available_actions)
        player *= -1
        state, _, _ = env.step(action_agent, player=player)

    done = False
    while not done:
        # User turn
        available_actions = [i for i, x in enumerate(env.board) if x == 0]
        color_codes: List[Optional[List[int]]] = [None] * 9
        for a in available_actions:
            q_val = agent.get_q_value(state, a)
            color_codes[a] = agent.get_color_code(q_val)
        
        env.print_board(color_codes)
        try:
            line = input("Your move: ")
            action_player = int(line) - 1
            if action_player < 0 or action_player > 8 or env.board[action_player] != 0:
                print("Invalid move, spot already taken or out of range.")
                time.sleep(1)
                continue
        except (ValueError, IndexError, EOFError):
            print("Invalid input. Please enter a number from 1 to 9.")
            time.sleep(1)
            continue

        player *= -1
        state, _, done = env.step(action_player, player=player)
        if done:
            break

        # Agent turn visualization
        available_actions = [i for i, x in enumerate(env.board) if x == 0]
        color_codes: List[Optional[List[int]]] = [None] * 9
        for a in available_actions:
            q_val = agent.get_q_value(state, a)
            color_codes[a] = agent.get_color_code(q_val)
        env.print_board(color_codes)
        print("Agent's move:", end="", flush=True)
        time.sleep(1.5)
        action_agent = agent.choose_action(state, available_actions)
        print("\rAgent's move: " + str(action_agent + 1))
        time.sleep(1.5)
        player *= -1
        state, _, done = env.step(action_agent, player=player)
        
    # Game over
    env.print_board()
    winner = env.check_winner()
    if winner == 0:
        print("It's a draw!")
    elif winner is not None:
        user_symbol = 1 if player_starts else -1
        if winner == user_symbol:
            print("You win!")
        else:
            print("Agent wins!")

def main() -> None:
    """Entry point for the Tic-Tac-Toe game."""
    env = TicTacToeEnv()
    agent = QAgent(alpha=0.1, gamma=0.9, epsilon=0.2)
    training_episodes = 1000
    q_file = os.path.join(os.path.dirname(__file__), "q_table.pkl")

    training_main(agent, env, training_episodes, q_file)

    agent.epsilon = 0.0  # Optimal play
    env.reset()
    
    print("Start Playing. Use Num Pad (1-9)")
    player_starts = should_the_player_start()
    game_loop(player_starts, agent, env)

if __name__ == "__main__":
    main()
