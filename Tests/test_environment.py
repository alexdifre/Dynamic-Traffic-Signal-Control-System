import unittest
from unittest.mock import Mock, patch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Reinf_Learn.environment import Environment

class TestEnvironment(unittest.TestCase):
    
    def test_1_reward_calculation_correctness(self):
        """Reward calculation must be mathematically correct
        
        If this fails, the agent learns completely wrong behavior.
        This is the CORE of your reinforcement learning system.
        """
        env = Environment()
        
        # Scenario 1Traffic improves (vehicles decrease from 10 to 5)
        env._last_state_vehicle_count = 10
        state_improved = (0, 3, 2, False)  # 3 + 2 = 5 vehicles
        reward_improved = env._compute_reward_signal(state_improved)
        self.assertAlmostEqual(reward_improved, 0.5, places=5,
                       msg="Reward must be positive and normalized when traffic improves")

        # Scenario 2Traffic worsens (vehicles increase from 5 to 12)
        env._last_state_vehicle_count = 5
        state_worsened = (1, 7, 5, True)  # 7 + 5 = 12 vehicles
        reward_worsened = env._compute_reward_signal(state_worsened)
        self.assertAlmostEqual(reward_worsened, -1.4, places=5,
                       msg="Reward must be negative and normalized when traffic worsens")

        # Scenario 3No change
        env._last_state_vehicle_count = 8
        state_same = (0, 4, 4, False)  # 4 + 4 = 8 vehicles
        reward_same = env._compute_reward_signal(state_same)
        self.assertAlmostEqual(reward_same, 0.0, places=5,
                       msg="Reward must be zero when no change")
    def test_2_step_executes_simulation_and_updates_cache(self):
        """TEST #2 - step() must run simulation AND update vehicle cache
        
        If simulation doesn't run, nothing happens,If cache doesn't update, all subsequent rewards are wrong..
        """
        env = Environment()
        
        mock_sim = Mock()
        mock_sim.completed = False
        mock_sim.gui_closed = False
        mock_sim.run = Mock()  # Track if run() is called
        
        mock_signal = Mock()
        mock_signal.current_cycle = [0]
        
        # Set up 8 vehicles (5 + 3)
        mock_road_1 = Mock()
        mock_road_1.vehicles = [Mock()] * 5
        mock_road_2 = Mock()
        mock_road_2.vehicles = [Mock()] * 3
        
        mock_signal.roads = [[mock_road_1], [mock_road_2]]
        mock_sim.traffic_signals = [mock_signal]
        mock_sim.n_vehicles_on_map = 8
        mock_sim.outbound_roads = []
        mock_sim.roads = []
        
        env.sim = mock_sim
        env._last_state_vehicle_count = 0
        
        # Execute step with action 1
        state, reward, done, truncated = env.step(1)
        
        #  CHECK 1Simulation was actually run with correct action
        mock_sim.run.assert_called_once_with(1)
        
        #  CHECK 2Cache updated to current vehicle count
        self.assertEqual(env._last_state_vehicle_count, 8,
                        "Cache must update for next reward calculation")
        
        #  CHECK 3Returns correct types
        self.assertIsInstance(state, tuple)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
    
    def test_3_state_structure_correctness(self):
        """TEST #3  State tuple must have correct structure
        
        Agent depends on state structure(signal, dir1_vehicles, dir2_vehicles, occupied)
        Wrong structure = agent cannot learn.
        """
        env = Environment()
        
        mock_sim = Mock()
        mock_signal = Mock()
        mock_signal.current_cycle = [1]  
        
        # Direction 1exactly 4 vehicles
        mock_road_1 = Mock()
        mock_road_1.vehicles = [Mock(), Mock(), Mock(), Mock()]
        
        # Direction 2exactly 6 vehicles
        mock_road_2 = Mock()
        mock_road_2.vehicles = [Mock()] * 6
        
        mock_signal.roads = [[mock_road_1], [mock_road_2]]
        mock_sim.traffic_signals = [mock_signal]
        mock_sim.n_vehicles_on_map = 10
        mock_sim.outbound_roads = []
        mock_sim.roads = []
        
        env.sim = mock_sim
        state = env._build_observation_tuple()
        
        self.assertEqual(len(state), 4, "State MUST have exactly 4 elements")
        self.assertEqual(state[0], 1, "Element 0 must be signal phase")
        self.assertEqual(state[1], 4, "Element 1 must be direction 1 vehicle count")
        self.assertEqual(state[2], 6, "Element 2 must be direction 2 vehicle count")
        self.assertIsInstance(state[3], bool, "Element 3 must be boolean occupancy")
  

if __name__ == '__main__':
    # Run with maximum verbosity to see what's happening
    unittest.main(verbosity=2)