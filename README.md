# Tic-Tac-Toe
![Java](https://img.shields.io/badge/Java-17-blue?logo=java)
![Eclipse](https://img.shields.io/badge/Eclipse-IDE-red)
![Q-Learning](https://img.shields.io/badge/Q--Learning-RL-blueviolet)
![Value Iteration](https://img.shields.io/badge/Value%20Iteration-RL-green)
![Policy Iteration](https://img.shields.io/badge/Policy%20Iteration-RL-yellow)


This project implements intelligent agents that learn to play a 3x3 Tic-Tac-Toe game using Markov Decision Processes (MDPs) and Reinforcement Learning techniques.

## Overview:

Three types of agents were implemented:

### Value Iteration Agent

- Solves the Tic-Tac-Toe game using the Value Iteration algorithm with a predefined MDP model.

### Policy Iteration Agent

- Uses Policy Iteration to compute the optimal policy based on the MDP model.

### Q-Learning Agent

- A Reinforcement Learning agent that learns through interaction with the environment without requiring a full model.

> Each agent was tested against other rule-based agents (such as a Random Agent), and supports human interaction through a command-line interface.

## Repository Structure:

### Implemented Agents

`ValueIterationAgent.java` -
Implements a value iteration algorithm to compute state utilities and derive an optimal policy.

`PolicyIterationAgent.java` -
Implements policy evaluation and improvement to converge to the optimal policy.

`QLearningAgent.java` -
A model-free agent that learns state-action values through trial-and-error.

### Provided Framework

These files were provided as part of the coursework and were not modified:

`Game.java` – Core Tic-Tac-Toe game logic.

`TTTMDP.java` – Defines the MDP model of the game.

`TTTEnvironment.java` – Environment for reinforcement learning.

`Agent.java` – Abstract base class for all agents.

`HumanAgent.java` – Allows a human to play via command line.

`RandomAgent.java` – Agent that plays moves randomly.

`Move.java`, `Outcome.java`, `TransitionProb.java` – Core classes for game transitions and outcomes.

`Policy.java`, `RandomPolicy.java` – Abstract and basic policy implementations.

## Features:

- Complete MDP-based planning and model-free learning implementations.

- Modular and extensible agent design.

- Playable via command-line interface.

- Supports evaluation against both rule-based and learned agents.

## Requirements:

Java Development Kit (JDK) 8 or higher.

## Running the Project:

To run a match between agents or play as a human:

> Compile all .java files.

> Run the main class or test files provided with the project check the game outputs.

<!-- ## Learning Algorithms

### Value Iteration

- Computes utilities of all states iteratively using the Bellman equation.
- Converges when the change in utility values falls below a threshold.

### Policy Iteration

- Alternates between evaluating the current policy and improving it.
- Guarantees convergence to the optimal policy for finite MDPs.

### Q-Learning

- Learns the optimal action-value function through exploration and exploitation.

- Does not require a model of the environment. -->

## Results and Implementation Notes:

### Policy Iteration Agent

| Against      | Wins | Losses | Draws |
| ------------ | ---- | ------ | ----- |
| `Random`     | 49   | 0      | 1     |
| `Defensive`  | 41   | 0      | 9     |
| `Aggressive` | 50   | 0      | 0     |

**Key Methods Implemented**

- `initRandomPolicy()`  
  Initializes a random policy by assigning random moves to all non-terminal states where it's X’s turn.

- `evaluatePolicy()`  
  Evaluates the current policy by computing state values using expected rewards and transitions, skipping terminal states.

- `improvePolicy()`  
  Improves the policy by selecting the best move for each state based on expected values. If a better move is found, the policy is updated.

- `train()`  
  Alternates between evaluation and improvement until convergence, resulting in the final policy.

### Value Iteration Agent

**Performance**

| Against      | Wins | Losses | Draws |
| ------------ | ---- | ------ | ----- |
| `Random`     | 50   | 0      | 0     |
| `Defensive`  | 41   | 0      | 9     |
| `Aggressive` | 49   | 0      | 1     |

**Key Methods Implemented**

- `iterate()`  
  Performs the value iteration loop over a fixed number of iterations. For each state, it calculates the best move’s expected value and updates the state value accordingly.

- `extractPolicy()`  
  Builds a policy from the value function by selecting the best action (with the highest expected value) for each non-terminal state.

**Helper Functions**

- `calculateExpectedValue()`  
  Computes the expected value of a move by aggregating over all possible transitions.

- `findBestMove()`  
  Identifies the move with the highest expected value from the current state.

### Q-Learning Agent

**Performance**

| Against      | Wins | Losses | Draws |
| ------------ | ---- | ------ | ----- |
| `Random`     | 50   | 0      | 0     |
| `Defensive`  | 45   | 0      | 5     |
| `Aggressive` | 50   | 0      | 0     |

**Key Methods Implemented**

- `train()`  
  Runs the Q-learning training loop over multiple episodes. Uses an epsilon-greedy strategy for exploration and decays epsilon over time. Updates Q-values using the Q-learning formula.

- `extractPolicy()`  
  Constructs the policy from the Q-table by selecting the highest Q-value move for each non-terminal state.

## Author

- **Vaishnavi Chintha** [`@Vaishnavi-chintha`](https://github.com/Vaishnavi-chintha)
