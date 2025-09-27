# Reinforcement Learning in Social Dilemma

This repository implements **2-Step SARSA** (State-Action-Reward-State-Action) reinforcement learning algorithm applied to the classic **Prisoner's Dilemma** game theory problem.

## Overview

The Prisoner's Dilemma is a fundamental game theory scenario where two rational individuals might not cooperate even when it would be in their mutual best interest to do so. This implementation uses 2-step SARSA to study how artificial agents learn optimal strategies through repeated interactions.

## Key Features

- **2-Step SARSA Algorithm**: Implements n-step temporal difference learning with n=2
- **Prisoner's Dilemma Environment**: Classic payoff matrix (R=3, T=5, S=0, P=1)
- **Epsilon-Greedy Exploration**: Dynamic exploration rate with decay
- **Self-Play Learning**: Two agents learning simultaneously through interaction
- **Comprehensive Visualization**: Q-value evolution, cooperation rates, and strategy analysis
- **Parameter Tuning**: Configurable learning rates, discount factors, and exploration parameters

## Algorithm Details

### 2-Step SARSA Update Rule

The Q-value update follows the 2-step SARSA formula:

```
G_{t:t+2} = R_{t+1} + γR_{t+2} + γ²Q(S_{t+2}, A_{t+2})
Q(S_t, A_t) ← Q(S_t, A_t) + α[G_{t:t+2} - Q(S_t, A_t)]
```

Where:
- `α`: Learning rate (step size)
- `γ`: Discount factor  
- `G_{t:t+2}`: Two-step return
- `R_{t+1}, R_{t+2}`: Immediate and next rewards
- `Q(S_{t+2}, A_{t+2})`: Q-value estimate at step t+2

### Action Selection

Actions are chosen using an ε-greedy policy:
- With probability `ε`: Random action (exploration)
- With probability `1-ε`: Best action according to Q-values (exploitation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/micah2021/Reinforcement-Learning-in-Social-Dilemma.git
cd Reinforcement-Learning-in-Social-Dilemma
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the main simulation:
```bash
python prisoners_dilemma_2step_sarsa.py
```

### Run tests to verify implementation:
```bash
python test_implementation.py
```

### Run quick demo:
```bash
python demo.py
```

## Results and Analysis

The implementation tracks and visualizes:

1. **Episode Rewards**: Total rewards per episode for both agents
2. **Cooperation Rates**: Probability of cooperation over time
3. **Q-Value Evolution**: How action-value estimates change during learning
4. **Strategy Convergence**: Final learned strategies and Q-tables
5. **Comparative Analysis**: Heatmaps showing Q(Cooperate) vs Q(Defect)

### Sample Results

- **Learning Dynamics**: Agents typically start with high exploration (random actions) and gradually develop consistent strategies
- **Cooperation Emergence**: Depending on parameters, agents may learn to cooperate, defect, or develop mixed strategies
- **Q-Value Convergence**: The algorithm shows stable convergence of action-value estimates

## Parameters

Key hyperparameters that can be tuned:

- `alpha` (0.1): Learning rate - how quickly Q-values are updated
- `gamma` (0.9): Discount factor - importance of future rewards
- `epsilon_start` (1.0): Initial exploration rate
- `epsilon_decay` (0.995): Rate at which exploration decreases
- `epsilon_min` (0.01): Minimum exploration rate
- `num_episodes` (1000): Number of training episodes
- `steps_per_episode` (100): Number of game rounds per episode

## File Structure

- `prisoners_dilemma_2step_sarsa.py`: Main implementation with classes and simulation
- `test_implementation.py`: Comprehensive test suite
- `demo.py`: Quick demonstration script
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Theoretical Background

This implementation is based on the theoretical framework outlined in the included LaTeX documents, which provide detailed mathematical foundations for:

- 2-Step SARSA algorithm formulation
- Prisoner's Dilemma game theory
- Reinforcement learning in multi-agent environments
- Convergence analysis and parameter selection

## Contributing

Feel free to contribute improvements, additional features, or bug fixes. Please ensure all tests pass before submitting changes.

## License

This project is for educational and research purposes.

## Author

**Micah Godwin Munang**  
Research in Reinforcement Learning and Game Theory
