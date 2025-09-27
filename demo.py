"""
Demo script for 2-Step SARSA Prisoner's Dilemma
This runs a quick demonstration of the algorithm
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prisoners_dilemma_2step_sarsa import PrisonersDilemmaSARSASimulation

def run_demo():
    """Run a quick demo simulation."""
    print("2-Step SARSA in Prisoner's Dilemma - Demo")
    print("="*40)
    
    # Initialize simulation with shorter parameters for demo
    simulation = PrisonersDilemmaSARSASimulation(num_episodes=100, steps_per_episode=50)
    
    # Agent parameters
    agent1_params = {
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.98,
        'epsilon_min': 0.05
    }
    
    agent2_params = {
        'alpha': 0.15,  # Slightly different learning rate
        'gamma': 0.85,
        'epsilon_start': 1.0,
        'epsilon_decay': 0.97,
        'epsilon_min': 0.1
    }
    
    # Run simulation
    results = simulation.run_simulation_self_play(agent1_params, agent2_params)
    
    # Save plots instead of showing them
    print("\nGenerating visualizations...")
    try:
        simulation.visualize_results(results)
        print("✓ Demo completed successfully!")
    except Exception as e:
        print(f"Note: Visualization skipped due to: {e}")
        print("✓ Core simulation completed successfully!")

if __name__ == "__main__":
    run_demo()