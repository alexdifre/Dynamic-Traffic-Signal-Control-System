from typing import List, Tuple
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir) 

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from TrafficSimulator.two_way_intersection import two_way_intersection_setup 


class Environment:
    def __init__(self):
        self.action_space: List = [0, 1]
        self.sim = None
        self.max_gen: int = 50
        self._last_state_vehicle_count: int = 0 

    def perform_step(self, control_signal) -> Tuple[Tuple, float, bool, bool]:
        """Processes one control interval in the simulation."""
        self.sim.run(control_signal)

        current_state: Tuple = self._capture_environment_state()
        performance_score: float = self._determine_performance(current_state)

        # Update vehicle count cache for next reward calculation
        n_west_east_vehicles, n_south_north_vehicles = current_state[1], current_state[2]
        self._last_state_vehicle_count = n_west_east_vehicles + n_south_north_vehicles

        # Whether a terminal state is reached
        simulation_ended: bool = self.sim.completed

        # Whether a truncation condition is satisfied 
        visualization_terminated: bool = self.sim.gui_closed

        return current_state, performance_score, simulation_ended, visualization_terminated

    
    def _capture_environment_state(self) -> Tuple:
        """
        Captures current intersection status:
        (signal_state, lane_a_count, lane_b_count, internal_traffic_present)
        """
        state = []
        for traffic_signal in self.sim.traffic_signals:
            junction = []
            traffic_signal_state = traffic_signal.current_cycle[0]
            junction.append(traffic_signal_state)

            for direction in traffic_signal.roads:
                junction.append(sum(len(road.vehicles) for road in direction))

            n_direction_1_vehicles, n_direction_2_vehicles = junction[1], junction[2]
            out_bound_vehicles = sum(len(self.sim.roads[i].vehicles) for i in self.sim.outbound_roads)
            non_empty_junction = bool(self.sim.n_vehicles_on_map - out_bound_vehicles -
                                      n_direction_1_vehicles - n_direction_2_vehicles)
            junction.append(non_empty_junction)
            state.append(junction)
        state = state[0]  # Optimization for a single junction simulation setup
        return tuple(state)


    def _determine_performance(self, state_observation: Tuple) -> float:
        """
        reward: penalizes congestion at each step.
        """
        _, n_direction_1_vehicles, n_direction_2_vehicles, _= state_observation

        current_vehicle_count = n_direction_1_vehicles + n_direction_2_vehicles

        prev_count = max(1, self._last_state_vehicle_count)
        reward = (prev_count - current_vehicle_count) / prev_count

        return float(reward)



    def restart_environment(self, enable_display: bool = False) -> Tuple:
        """Resets traffic simulation and returns initial conditions."""
        self.sim = two_way_intersection_setup(self.max_gen)
        if enable_display:
            self.sim.init_gui()
        starting_state = self._capture_environment_state()
        self._last_state_vehicle_count = 0  # Reset the counter
        return starting_state

    def retrieve_current_conditions(self) -> Tuple:
        """Provides current environmental observation."""
        return self._capture_environment_state()

    def assess_state_performance(self, environmental_state: Tuple) -> float:
        """External method for performance evaluation."""
        return self._determine_performance(environmental_state)

    def step(self, action):
        """Alias for perform_step to match standard gym/test interface."""
        return self.perform_step(action)
    
    def _compute_reward_signal(self, state):
        """Alias for _determine_performance to match test interface."""
        return self._determine_performance(state)
    
    def _build_observation_tuple(self):
        """Alias for _capture_environment_state to match test interface."""
        return self._capture_environment_state()
    

    @property
    def action_set(self):
        return self.action_space
    
    @property
    def traffic_model(self):
        return self.sim