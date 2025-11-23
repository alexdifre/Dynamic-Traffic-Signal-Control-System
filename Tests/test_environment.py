import unittest
from unittest.mock import Mock, patch
from Environment import Environment


class TestEnvironmentCritical(unittest.TestCase):
    """ONLY THE MOST CRITICAL TESTS - Minimum to ensure correctness"""
    
    def test_1_reward_calculation_correctness(self):
        """TEST #1 - MOST CRITICAL: Reward calculation must be mathematically correct
        
        If this fails, the agent learns completely wrong behavior.
        This is the CORE of your reinforcement learning system.
        """
        env = Environment()
        
        # Scenario 1: Traffic improves (vehicles decrease from 10 to 5)
        env._last_state_vehicle_count = 10
        state_improved = (0, 3, 2, False)  # 3 + 2 = 5 vehicles
        reward_improved = env._compute_reward_signal(state_improved)
        self.assertEqual(reward_improved, 5.0, 
                        "CRITICAL: Reward must be positive when traffic improves")
        
        # Scenario 2: Traffic worsens (vehicles increase from 5 to 12)
        env._last_state_vehicle_count = 5
        state_worsened = (1, 7, 5, True)  # 7 + 5 = 12 vehicles
        reward_worsened = env._compute_reward_signal(state_worsened)
        self.assertEqual(reward_worsened, -7.0,
                        "CRITICAL: Reward must be negative when traffic worsens")
        
        # Scenario 3: No change
        env._last_state_vehicle_count = 8
        state_same = (0, 4, 4, False)  # 4 + 4 = 8 vehicles
        reward_same = env._compute_reward_signal(state_same)
        self.assertEqual(reward_same, 0.0,
                        "CRITICAL: Reward must be zero when no change")
    
    def test_2_step_executes_simulation_and_updates_cache(self):
        """TEST #2 - CRITICAL: step() must run simulation AND update vehicle cache
        
        If simulation doesn't run, nothing happens.
        If cache doesn't update, all subsequent rewards are wrong.
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
        
        # CRITICAL CHECK 1: Simulation was actually run with correct action
        mock_sim.run.assert_called_once_with(1)
        
        # CRITICAL CHECK 2: Cache updated to current vehicle count
        self.assertEqual(env._last_state_vehicle_count, 8,
                        "CRITICAL: Cache must update for next reward calculation")
        
        # CRITICAL CHECK 3: Returns correct types
        self.assertIsInstance(state, tuple)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
    
    def test_3_state_structure_correctness(self):
        """TEST #3 - CRITICAL: State tuple must have correct structure
        
        Agent depends on state structure: (signal, dir1_vehicles, dir2_vehicles, occupied)
        Wrong structure = agent cannot learn.
        """
        env = Environment()
        
        mock_sim = Mock()
        mock_signal = Mock()
        mock_signal.current_cycle = [1]  # Signal phase = 1
        
        # Direction 1: exactly 4 vehicles
        mock_road_1 = Mock()
        mock_road_1.vehicles = [Mock(), Mock(), Mock(), Mock()]
        
        # Direction 2: exactly 6 vehicles
        mock_road_2 = Mock()
        mock_road_2.vehicles = [Mock()] * 6
        
        mock_signal.roads = [[mock_road_1], [mock_road_2]]
        mock_sim.traffic_signals = [mock_signal]
        mock_sim.n_vehicles_on_map = 10
        mock_sim.outbound_roads = []
        mock_sim.roads = []
        
        env.sim = mock_sim
        state = env._build_observation_tuple()
        
        # CRITICAL CHECKS
        self.assertEqual(len(state), 4, "State MUST have exactly 4 elements")
        self.assertEqual(state[0], 1, "Element 0 must be signal phase")
        self.assertEqual(state[1], 4, "Element 1 must be direction 1 vehicle count")
        self.assertEqual(state[2], 6, "Element 2 must be direction 2 vehicle count")
        self.assertIsInstance(state[3], bool, "Element 3 must be boolean occupancy")
    
    def test_4_reward_consistency_integration(self):
        """TEST #4 - CRITICAL INTEGRATION: Rewards must be consistent across multiple steps
        
        This tests the entire reward pipeline end-to-end.
        If this fails, there's a bug in how state/reward interact.
        """
        with patch('Environment.two_way_intersection_setup') as mock_setup:
            mock_sim = Mock()
            mock_sim.completed = False
            mock_sim.gui_closed = False
            
            mock_signal = Mock()
            mock_signal.current_cycle = [0]
            
            # Start: 10 vehicles (6 + 4)
            mock_road_1 = Mock()
            mock_road_1.vehicles = [Mock()] * 6
            mock_road_2 = Mock()
            mock_road_2.vehicles = [Mock()] * 4
            mock_signal.roads = [[mock_road_1], [mock_road_2]]
            
            mock_sim.traffic_signals = [mock_signal]
            mock_sim.n_vehicles_on_map = 10
            mock_sim.outbound_roads = []
            mock_sim.roads = []
            mock_setup.return_value = mock_sim
            
            env = Environment()
            env.reset()
            
            # Step 1: Cache starts at 0, current is 10
            _, r1, _, _ = env.step(0)
            self.assertEqual(r1, -10.0, "Step 1: 0 - 10 = -10")
            
            # Step 2: Cache is 10, reduce to 7 (4 + 3)
            mock_road_1.vehicles = [Mock()] * 4
            mock_road_2.vehicles = [Mock()] * 3
            _, r2, _, _ = env.step(1)
            self.assertEqual(r2, 3.0, "Step 2: 10 - 7 = +3")
            
            # Step 3: Cache is 7, increase to 11 (6 + 5)
            mock_road_1.vehicles = [Mock()] * 6
            mock_road_2.vehicles = [Mock()] * 5
            _, r3, _, _ = env.step(0)
            self.assertEqual(r3, -4.0, "Step 3: 7 - 11 = -4")
            
            # CRITICAL: Sum of rewards = net flow
            total_reward = r1 + r2 + r3
            net_flow = 0 - 11  # Started at 0, ended at 11
            self.assertEqual(total_reward, net_flow,
                           "CRITICAL INVARIANT: Reward sum must equal net vehicle flow")
    
    def test_5_termination_flags(self):
        """TEST #5 - CRITICAL: Termination flags must work correctly
        
        If terminated/truncated don't work, episodes never end or end incorrectly.
        """
        env = Environment()
        
        mock_sim = Mock()
        mock_signal = Mock()
        mock_signal.current_cycle = [0]
        mock_signal.roads = [[], []]
        mock_sim.traffic_signals = [mock_signal]
        mock_sim.n_vehicles_on_map = 0
        mock_sim.outbound_roads = []
        mock_sim.roads = []
        
        env.sim = mock_sim
        
        # Test 1: Normal operation (not terminated, not truncated)
        mock_sim.completed = False
        mock_sim.gui_closed = False
        _, _, terminated, truncated = env.step(0)
        self.assertFalse(terminated, "Should not terminate during episode")
        self.assertFalse(truncated, "Should not truncate during episode")
        
        # Test 2: Episode completes normally
        mock_sim.completed = True
        mock_sim.gui_closed = False
        _, _, terminated, truncated = env.step(0)
        self.assertTrue(terminated, "CRITICAL: Must terminate when episode completes")
        self.assertFalse(truncated, "Should not truncate on normal completion")
        
        # Test 3: GUI closed (truncation)
        mock_sim.completed = False
        mock_sim.gui_closed = True
        _, _, terminated, truncated = env.step(0)
        self.assertTrue(truncated, "CRITICAL: Must truncate when GUI closes")
    
    def test_6_reset_initialization(self):
        """TEST #6 - CRITICAL: Reset must properly initialize for new episode
        
        If reset doesn't work, training will fail after first episode.
        """
        with patch('Environment.two_way_intersection_setup') as mock_setup:
            mock_sim = Mock()
            mock_signal = Mock()
            mock_signal.current_cycle = [0]
            mock_signal.roads = [[], []]
            mock_sim.traffic_signals = [mock_signal]
            mock_sim.n_vehicles_on_map = 0
            mock_sim.outbound_roads = []
            mock_sim.roads = []
            mock_setup.return_value = mock_sim
            
            env = Environment()
            
            # Mess up the environment state
            env._last_state_vehicle_count = 999
            env.sim = Mock()
            
            # Reset should clean everything
            initial_state = env.reset()
            
            # CRITICAL CHECK 1: New simulation created
            mock_setup.assert_called_once_with(50)
            
            # CRITICAL CHECK 2: Cache reset to 0
            self.assertEqual(env._last_state_vehicle_count, 0,
                           "CRITICAL: Cache must reset to 0")
            
            # CRITICAL CHECK 3: Returns valid initial state
            self.assertIsInstance(initial_state, tuple)
            self.assertEqual(len(initial_state), 4)


if __name__ == '__main__':
    # Run with maximum verbosity to see what's happening
    unittest.main(verbosity=2)