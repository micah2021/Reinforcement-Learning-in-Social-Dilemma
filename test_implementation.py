"""
Test script for 2-Step SARSA Prisoner's Dilemma implementation
"""

import numpy as np
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prisoners_dilemma_2step_sarsa import (
    PrisonersDilemmaEnvironment,
    TwoStepSARSAAgent,
    PrisonersDilemmaSARSASimulation
)


def test_environment():
    """Test the Prisoner's Dilemma environment."""
    print("Testing Prisoner's Dilemma Environment...")
    
    env = PrisonersDilemmaEnvironment()
    
    # Test reward matrix
    assert env.get_reward('C', 'C') == (3, 3), "Mutual cooperation should return (3, 3)"
    assert env.get_reward('C', 'D') == (0, 5), "Cooperate vs Defect should return (0, 5)"
    assert env.get_reward('D', 'C') == (5, 0), "Defect vs Cooperate should return (5, 0)"
    assert env.get_reward('D', 'D') == (1, 1), "Mutual defection should return (1, 1)"
    
    print("✓ Environment tests passed!")


def test_agent():
    """Test the 2-Step SARSA agent."""
    print("Testing 2-Step SARSA Agent...")
    
    agent = TwoStepSARSAAgent(alpha=0.1, gamma=0.9, epsilon_start=1.0)
    
    # Test state representation
    assert agent.get_state(None) == 'start', "Initial state should be 'start'"
    assert agent.get_state('C') == 'C', "State after cooperation should be 'C'"
    assert agent.get_state('D') == 'D', "State after defection should be 'D'"
    
    # Test Q-value initialization
    assert agent.get_q_value('start', 'C') == 0.0, "Initial Q-values should be 0"
    assert agent.get_q_value('start', 'D') == 0.0, "Initial Q-values should be 0"
    
    # Test Q-value setting
    agent.set_q_value('start', 'C', 1.5)
    assert agent.get_q_value('start', 'C') == 1.5, "Q-value should be updated"
    
    # Test action selection (with epsilon=1, should be random)
    action = agent.choose_action('start')
    assert action in ['C', 'D'], "Action should be C or D"
    
    print("✓ Agent tests passed!")


def test_simulation_short():
    """Test a short simulation run."""
    print("Testing Short Simulation...")
    
    simulation = PrisonersDilemmaSARSASimulation(num_episodes=10, steps_per_episode=20)
    
    agent_params = {
        'alpha': 0.1,
        'gamma': 0.9,
        'epsilon_start': 0.5,
        'epsilon_decay': 0.99,
        'epsilon_min': 0.1
    }
    
    results = simulation.run_simulation_self_play(agent_params, agent_params)
    
    # Check that results contain expected keys
    expected_keys = ['agent1', 'agent2', 'episode_rewards_1', 'episode_rewards_2', 
                     'cooperation_rates_1', 'cooperation_rates_2']
    for key in expected_keys:
        assert key in results, f"Results should contain key '{key}'"
    
    # Check that we have the right number of episodes
    assert len(results['episode_rewards_1']) == 10, "Should have 10 episodes of rewards"
    assert len(results['cooperation_rates_1']) == 10, "Should have 10 episodes of cooperation rates"
    
    # Check that cooperation rates are between 0 and 1
    for rate in results['cooperation_rates_1']:
        assert 0 <= rate <= 1, "Cooperation rate should be between 0 and 1"
    
    print("✓ Short simulation test passed!")


def test_q_value_update():
    """Test the 2-step SARSA Q-value update."""
    print("Testing Q-value Update...")
    
    agent = TwoStepSARSAAgent(alpha=0.5, gamma=0.9)
    
    # Set some initial Q-values
    agent.set_q_value('start', 'C', 1.0)
    agent.set_q_value('C', 'D', 2.0)
    
    # Perform a 2-step update
    # G_{t:t+2} = R_{t+1} + γ*R_{t+2} + γ²*Q(S_{t+2}, A_{t+2})
    # Expected: G = 3 + 0.9*0 + 0.9²*2.0 = 3 + 0 + 1.62 = 4.62
    # Update: Q_new = Q_old + α*(G - Q_old) = 1.0 + 0.5*(4.62 - 1.0) = 1.0 + 1.81 = 2.81
    
    initial_q = agent.get_q_value('start', 'C')
    agent.update_q_value_2step(
        state_t='start', action_t='C', reward_t1=3,
        state_t1='C', action_t1='D', reward_t2=0,
        state_t2='C', action_t2='D'
    )
    updated_q = agent.get_q_value('start', 'C')
    
    assert updated_q != initial_q, "Q-value should be updated"
    assert updated_q > initial_q, "Q-value should increase due to positive reward"
    
    print("✓ Q-value update test passed!")


def run_all_tests():
    """Run all tests."""
    print("Running all tests for 2-Step SARSA implementation...")
    print("=" * 50)
    
    try:
        test_environment()
        test_agent()
        test_q_value_update()
        test_simulation_short()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The 2-Step SARSA implementation is working correctly.")
        print("=" * 50)
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)