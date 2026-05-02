# Reinforcement Learning Algorithms

This repository contains implementations of various Reinforcement Learning (RL) algorithms, starting with Q-Learning applied to classic games.

## 1. Q-Learning

### Tic-Tac-Toe
A zero-sum self-play implementation where an agent learns to play Tic-Tac-Toe optimally using the Q-Learning algorithm.

#### How to Start
To run the Tic-Tac-Toe game, execute the following command from the root directory:

```bash
python Q-Learning/TicTacToe.py
```

#### What's Happening in the Background
- **Self-Play Training:** When the program starts, it checks for an existing `q_table.pkl` in the `Q-Learning/` directory. If not found, the agent trains by playing against itself for 1,000 episodes.
- **Learning Rule:** The agent uses a minimax-style Q-learning update:

  `V(s, a) = V(s, a) + alpha * (reward - gamma * max(V(s', a')) - V(s, a))`
  By subtracting the maximum Q-value of the next state, the agent accounts for the opponent's optimal response.
- **Persistence:** After training, the agent's knowledge is saved to `Q-Learning/q_table.pkl`, allowing it to skip training in future sessions.
- **Decision Making:** During gameplay, the agent uses its learned Q-values to choose the action with the highest expected reward (epsilon is set to 0 for optimal play).

#### Interpreting Terminal Colors
The game board uses color-coding to visualize the agent's "thoughts" for each available move. This helps you understand how the agent evaluates the current board state:

- **Green Background:** High Q-value. The agent evaluates this as a strong move likely leading to a win or a draw.
- **Red Background:** Low Q-value. The agent evaluates this as a weak move likely leading to a loss.
- **Mud Green Tones:** Neutral or uncertain evaluation.

---

## Future Algorithms
This repository is designed to expand. Future additions will include:
<!-- - **Deep Q-Networks (DQN)** -->
<!-- - **SARSA** -->
<!-- - **Actor-Critic Models** -->
- **Policy Gradient Methods**
- **Trust Region Policy Optimization (TRPO)**
- **Proximal Policy Optimization (PPO)**
