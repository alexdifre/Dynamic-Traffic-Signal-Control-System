import unittest
from unittest.mock import Mock, patch
import os
import sys
import tempfile


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Reinf_Learn.utils import (
    run_training_session,
    run_evaluation_session,
    EPSILON
)

class Test_Training_Running(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.epsilon = EPSILON
        self.mock_model.q_data = {}
        self.mock_model.select_action = Mock(return_value=0)
        self.mock_model.learn = Mock()
        
        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.restart_environment = Mock(return_value=(0, 5, 3, False))
        self.mock_env.perform_step = Mock()
    
    def test_training_tracks_best_reward(self):
        """TEST: Should track best episode reward correctly"""
        rewards = [10.0, -5.0, 25.0, 15.0, 30.0]  # Best = 30.0
        
        reward_idx = [0]
        def mock_step(action):
            current_reward = rewards[min(reward_idx[0], len(rewards)-1)]
            reward_idx[0] += 1
            # End episode after each reward
            return (0, 3, 2, False), current_reward, True, False
        
        self.mock_env.perform_step = mock_step
        
        # Capture print output to check best reward
        with patch('builtins.print') as mock_print:
            run_training_session(
                self.mock_model,
                self.mock_env,
                "test_model.dat",
                total_episodes=len(rewards),
                display=False
            )
            
            # Check that best reward (30.0) was printed
            output = str(mock_print.call_args_list)
            self.assertIn("30.00", output, "Best reward should be printed")
            
            
    def test_evaluation_accumulates_rewards_correctly(self):
        """Should accumulate rewards within each episode correctly"""
        # Episode with multiple steps
        step_rewards = [1.0, 2.0, 3.0, 4.0]  # Total = 10.0
        
        step_idx = [0]
        def mock_step(action):
            reward = step_rewards[step_idx[0]]
            step_idx[0] += 1
            done = (step_idx[0] >= len(step_rewards))
            if done:
                step_idx[0] = 0
            return (0, 3, 2, False), reward, done, False
        
        self.mock_env.perform_step = mock_step
        
        with patch('builtins.print') as mock_print:
            run_evaluation_session(
                self.mock_model,
                self.mock_env,
                total_episodes=1,
                display=False
            )
            
            output = str(mock_print.call_args_list)
            # Episode total should be 10.0
            self.assertIn("10.00", output)


if __name__ == '__main__':
    # Run with maximum verbosity to see what's happening
    unittest.main(verbosity=2)