"""
Reinforcement Learning in Prisoner's Dilemma Using 2-Step SARSA

This implementation demonstrates the application of 2-step SARSA algorithm
to the classic Prisoner's Dilemma game. The algorithm learns optimal strategies
through repeated interactions with the environment.

Author: Micah Godwin Munang
Date: February 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional


class PrisonersDilemmaEnvironment:
    """
    Prisoner's Dilemma game environment.
    """
    
    def __init__(self, reward_matrix: Optional[Dict] = None):
        """
        Initialize the Prisoner's Dilemma environment.
        
        Args:
            reward_matrix: Custom reward matrix. If None, uses standard values.
        """
        # Standard Prisoner's Dilemma payoff matrix
        # R=3, T=5, S=0, P=1 where T>R>P>S and 2R>T+S
        self.reward_matrix = reward_matrix or {
            ('C', 'C'): (3, 3),  # Mutual cooperation (R, R)
            ('C', 'D'): (0, 5),  # Cooperate vs Defect (S, T)
            ('D', 'C'): (5, 0),  # Defect vs Cooperate (T, S)
            ('D', 'D'): (1, 1),  # Mutual defection (P, P)
        }
        
        self.actions = ['C', 'D']  # Cooperate, Defect
        
    def get_reward(self, action1: str, action2: str) -> Tuple[float, float]:
        """
        Get rewards for both players given their actions.
        
        Args:
            action1: Action of player 1
            action2: Action of player 2
            
        Returns:
            Tuple of rewards (reward1, reward2)
        """
        return self.reward_matrix[(action1, action2)]


class TwoStepSARSAAgent:
    """
    2-Step SARSA agent for Prisoner's Dilemma.
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon_start: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize the 2-Step SARSA agent.
        
        Args:
            alpha: Learning rate (step size)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_decay: Exploration decay rate
            epsilon_min: Minimum exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: Q(state, action) -> value
        # State is represented as the last opponent action (or None for first move)
        self.q_table = {}
        
        # Action history for analysis
        self.action_history = []
        self.q_value_history = {'C': [], 'D': []}
        
    def get_state(self, opponent_last_action: Optional[str]) -> str:
        """
        Get current state based on opponent's last action.
        
        Args:
            opponent_last_action: Opponent's previous action or None
            
        Returns:
            State representation
        """
        return opponent_last_action or 'start'
    
    def get_q_value(self, state: str, action: str) -> float:
        """
        Get Q-value for state-action pair.
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            Q-value
        """
        return self.q_table.get((state, action), 0.0)
    
    def set_q_value(self, state: str, action: str, value: float):
        """
        Set Q-value for state-action pair.
        
        Args:
            state: Current state
            action: Action
            value: New Q-value
        """
        self.q_table[(state, action)] = value
    
    def choose_action(self, state: str) -> str:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(['C', 'D'])
        else:
            # Exploit: best action
            q_c = self.get_q_value(state, 'C')
            q_d = self.get_q_value(state, 'D')
            
            if q_c > q_d:
                return 'C'
            elif q_d > q_c:
                return 'D'
            else:
                # Equal Q-values, choose randomly
                return np.random.choice(['C', 'D'])
    
    def update_q_value_2step(self, 
                             state_t: str, 
                             action_t: str, 
                             reward_t1: float,
                             state_t1: str,
                             action_t1: str,
                             reward_t2: float,
                             state_t2: str,
                             action_t2: str):
        """
        Update Q-value using 2-step SARSA formula.
        
        Args:
            state_t: State at time t
            action_t: Action at time t
            reward_t1: Reward at t+1
            state_t1: State at t+1
            action_t1: Action at t+1
            reward_t2: Reward at t+2
            state_t2: State at t+2
            action_t2: Action at t+2
        """
        # Calculate 2-step return
        # G_{t:t+2} = R_{t+1} + γ*R_{t+2} + γ²*Q(S_{t+2}, A_{t+2})
        g_2step = (reward_t1 + 
                   self.gamma * reward_t2 + 
                   self.gamma**2 * self.get_q_value(state_t2, action_t2))
        
        # Update Q-value
        # Q(S_t, A_t) ← Q(S_t, A_t) + α[G_{t:t+2} - Q(S_t, A_t)]
        current_q = self.get_q_value(state_t, action_t)
        new_q = current_q + self.alpha * (g_2step - current_q)
        self.set_q_value(state_t, action_t, new_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def record_action(self, action: str):
        """Record action for analysis."""
        self.action_history.append(action)
    
    def record_q_values(self, state: str):
        """Record current Q-values for analysis."""
        self.q_value_history['C'].append(self.get_q_value(state, 'C'))
        self.q_value_history['D'].append(self.get_q_value(state, 'D'))


class PrisonersDilemmaSARSASimulation:
    """
    Simulation runner for 2-Step SARSA in Prisoner's Dilemma.
    """
    
    def __init__(self, 
                 num_episodes: int = 1000,
                 steps_per_episode: int = 100):
        """
        Initialize simulation.
        
        Args:
            num_episodes: Number of training episodes
            steps_per_episode: Number of steps per episode
        """
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.env = PrisonersDilemmaEnvironment()
        
        # Results storage
        self.episode_rewards = []
        self.cooperation_rates = []
        self.q_value_evolution = {'C': [], 'D': []}
        
    def run_simulation_self_play(self, 
                                agent1_params: Dict = None, 
                                agent2_params: Dict = None) -> Dict:
        """
        Run self-play simulation with two learning agents.
        
        Args:
            agent1_params: Parameters for agent 1
            agent2_params: Parameters for agent 2
            
        Returns:
            Simulation results dictionary
        """
        # Initialize agents
        params1 = agent1_params or {}
        params2 = agent2_params or {}
        
        agent1 = TwoStepSARSAAgent(**params1)
        agent2 = TwoStepSARSAAgent(**params2)
        
        episode_rewards_1 = []
        episode_rewards_2 = []
        cooperation_rates_1 = []
        cooperation_rates_2 = []
        
        print(f"Running {self.num_episodes} episodes of self-play simulation...")
        
        for episode in range(self.num_episodes):
            # Initialize episode
            total_reward_1 = 0
            total_reward_2 = 0
            episode_actions_1 = []
            episode_actions_2 = []
            
            # Episode history for 2-step updates
            states = []
            actions = []
            rewards = []
            
            # Initial state
            opponent_last_action_1 = None  # For agent 1
            opponent_last_action_2 = None  # For agent 2
            
            for step in range(self.steps_per_episode):
                # Get current states
                state_1 = agent1.get_state(opponent_last_action_1)
                state_2 = agent2.get_state(opponent_last_action_2)
                
                # Choose actions
                action_1 = agent1.choose_action(state_1)
                action_2 = agent2.choose_action(state_2)
                
                # Get rewards
                reward_1, reward_2 = self.env.get_reward(action_1, action_2)
                
                # Store step information
                states.append((state_1, state_2))
                actions.append((action_1, action_2))
                rewards.append((reward_1, reward_2))
                
                # Update totals
                total_reward_1 += reward_1
                total_reward_2 += reward_2
                episode_actions_1.append(action_1)
                episode_actions_2.append(action_2)
                
                # Record Q-values periodically
                if step % 20 == 0:
                    agent1.record_q_values(state_1)
                    agent2.record_q_values(state_2)
                
                # 2-step SARSA update (after we have at least 2 steps)
                if step >= 2:
                    # Update agent 1
                    agent1.update_q_value_2step(
                        state_t=states[step-2][0],
                        action_t=actions[step-2][0],
                        reward_t1=rewards[step-1][0],
                        state_t1=states[step-1][0],
                        action_t1=actions[step-1][0],
                        reward_t2=rewards[step][0],
                        state_t2=states[step][0],
                        action_t2=actions[step][0]
                    )
                    
                    # Update agent 2
                    agent2.update_q_value_2step(
                        state_t=states[step-2][1],
                        action_t=actions[step-2][1],
                        reward_t1=rewards[step-1][1],
                        state_t1=states[step-1][1],
                        action_t1=actions[step-1][1],
                        reward_t2=rewards[step][1],
                        state_t2=states[step][1],
                        action_t2=actions[step][1]
                    )
                
                # Update opponent's last action for next iteration
                opponent_last_action_1 = action_2
                opponent_last_action_2 = action_1
            
            # Decay exploration
            agent1.decay_epsilon()
            agent2.decay_epsilon()
            
            # Record episode results
            episode_rewards_1.append(total_reward_1)
            episode_rewards_2.append(total_reward_2)
            
            cooperation_rate_1 = episode_actions_1.count('C') / len(episode_actions_1)
            cooperation_rate_2 = episode_actions_2.count('C') / len(episode_actions_2)
            cooperation_rates_1.append(cooperation_rate_1)
            cooperation_rates_2.append(cooperation_rate_2)
            
            # Progress update
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.num_episodes} - "
                      f"Agent 1 Avg Reward: {np.mean(episode_rewards_1[-100:]):.2f}, "
                      f"Agent 2 Avg Reward: {np.mean(episode_rewards_2[-100:]):.2f}, "
                      f"Cooperation: {cooperation_rate_1:.2f}, {cooperation_rate_2:.2f}")
        
        return {
            'agent1': agent1,
            'agent2': agent2,
            'episode_rewards_1': episode_rewards_1,
            'episode_rewards_2': episode_rewards_2,
            'cooperation_rates_1': cooperation_rates_1,
            'cooperation_rates_2': cooperation_rates_2
        }
    
    def visualize_results(self, results: Dict):
        """
        Create comprehensive visualizations of the simulation results.
        
        Args:
            results: Results dictionary from simulation
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('2-Step SARSA in Prisoner\'s Dilemma - Simulation Results', fontsize=16)
        
        # 1. Episode Rewards
        axes[0, 0].plot(results['episode_rewards_1'], alpha=0.7, label='Agent 1')
        axes[0, 0].plot(results['episode_rewards_2'], alpha=0.7, label='Agent 2')
        
        # Moving averages
        window = 50
        if len(results['episode_rewards_1']) >= window:
            ma1 = np.convolve(results['episode_rewards_1'], np.ones(window)/window, mode='valid')
            ma2 = np.convolve(results['episode_rewards_2'], np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(ma1)+window-1), ma1, '--', linewidth=2, label='Agent 1 MA')
            axes[0, 0].plot(range(window-1, len(ma2)+window-1), ma2, '--', linewidth=2, label='Agent 2 MA')
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cooperation Rates
        axes[0, 1].plot(results['cooperation_rates_1'], alpha=0.7, label='Agent 1')
        axes[0, 1].plot(results['cooperation_rates_2'], alpha=0.7, label='Agent 2')
        axes[0, 1].set_title('Cooperation Rates')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cooperation Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Q-Value Evolution for Agent 1
        if results['agent1'].q_value_history['C']:
            axes[0, 2].plot(results['agent1'].q_value_history['C'], label='Q(s, C)', linewidth=2)
            axes[0, 2].plot(results['agent1'].q_value_history['D'], label='Q(s, D)', linewidth=2)
            axes[0, 2].set_title('Agent 1 Q-Value Evolution')
            axes[0, 2].set_xlabel('Time Steps (×20)')
            axes[0, 2].set_ylabel('Q-Value')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Value Evolution for Agent 2
        if results['agent2'].q_value_history['C']:
            axes[1, 0].plot(results['agent2'].q_value_history['C'], label='Q(s, C)', linewidth=2)
            axes[1, 0].plot(results['agent2'].q_value_history['D'], label='Q(s, D)', linewidth=2)
            axes[1, 0].set_title('Agent 2 Q-Value Evolution')
            axes[1, 0].set_xlabel('Time Steps (×20)')
            axes[1, 0].set_ylabel('Q-Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Q-Value Difference Heatmap (Agent 1)
        if results['agent1'].q_value_history['C']:
            q_diff_1 = np.array(results['agent1'].q_value_history['C']) - np.array(results['agent1'].q_value_history['D'])
            
            # Reshape for heatmap (if we have enough data)
            if len(q_diff_1) >= 10:
                # Create a 2D representation
                n_rows = min(10, len(q_diff_1) // 10)
                if n_rows > 0:
                    heatmap_data = q_diff_1[:n_rows*10].reshape(n_rows, -1)
                    sns.heatmap(heatmap_data, ax=axes[1, 1], cmap='RdBu_r', center=0, 
                               cbar_kws={'label': 'Q(C) - Q(D)'})
                    axes[1, 1].set_title('Agent 1: Q(C) - Q(D) Evolution')
                    axes[1, 1].set_xlabel('Time Steps')
                    axes[1, 1].set_ylabel('Episode Group')
        
        # 6. Final Q-Tables Comparison
        states = ['start', 'C', 'D']
        actions = ['C', 'D']
        
        q_table_1 = np.zeros((len(states), len(actions)))
        q_table_2 = np.zeros((len(states), len(actions)))
        
        for i, state in enumerate(states):
            for j, action in enumerate(actions):
                q_table_1[i, j] = results['agent1'].get_q_value(state, action)
                q_table_2[i, j] = results['agent2'].get_q_value(state, action)
        
        # Create comparison heatmap
        diff_table = q_table_1 - q_table_2
        sns.heatmap(diff_table, ax=axes[1, 2], annot=True, fmt='.2f', 
                   xticklabels=actions, yticklabels=states,
                   cmap='RdBu_r', center=0, cbar_kws={'label': 'Q-Value Difference'})
        axes[1, 2].set_title('Final Q-Table Difference (Agent1 - Agent2)')
        
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        print("\n" + "="*50)
        print("FINAL SIMULATION RESULTS")
        print("="*50)
        
        avg_reward_1 = np.mean(results['episode_rewards_1'][-100:])
        avg_reward_2 = np.mean(results['episode_rewards_2'][-100:])
        final_coop_1 = np.mean(results['cooperation_rates_1'][-100:])
        final_coop_2 = np.mean(results['cooperation_rates_2'][-100:])
        
        print(f"Agent 1 - Final Average Reward: {avg_reward_1:.2f}")
        print(f"Agent 2 - Final Average Reward: {avg_reward_2:.2f}")
        print(f"Agent 1 - Final Cooperation Rate: {final_coop_1:.2f}")
        print(f"Agent 2 - Final Cooperation Rate: {final_coop_2:.2f}")
        
        print("\nFinal Q-Tables:")
        print("Agent 1:")
        for state in states:
            q_c = results['agent1'].get_q_value(state, 'C')
            q_d = results['agent1'].get_q_value(state, 'D')
            print(f"  State '{state}': Q(C)={q_c:.3f}, Q(D)={q_d:.3f}")
        
        print("Agent 2:")
        for state in states:
            q_c = results['agent2'].get_q_value(state, 'C')
            q_d = results['agent2'].get_q_value(state, 'D')
            print(f"  State '{state}': Q(C)={q_c:.3f}, Q(D)={q_d:.3f}")


def main():
    """
    Main function to run the simulation and demonstrate the 2-step SARSA algorithm.
    """
    print("2-Step SARSA in Prisoner's Dilemma")
    print("="*40)
    
    # Initialize simulation
    simulation = PrisonersDilemmaSARSASimulation(num_episodes=1000, steps_per_episode=100)
    
    # Agent parameters
    agent1_params = {
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    }
    
    agent2_params = {
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    }
    
    # Run simulation
    results = simulation.run_simulation_self_play(agent1_params, agent2_params)
    
    # Visualize results
    simulation.visualize_results(results)


if __name__ == "__main__":
    main()