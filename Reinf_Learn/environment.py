from typing import Optional, List, Tuple
import sys
import os

# Aggiungi il path della cartella parent per importare TrafficSimulator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TrafficSimulator import two_way_intersection_setup 


class Environment:
    def __init__(self):
        self.action_space: List = [0, 1]
        self.sim: Optional[Simulation] = None
        self.max_gen: int = 50
        self._vehicles_on_inbound_roads: int = 0

    def perform_step(self, control_signal) -> Tuple[Tuple, float, bool, bool]:
        """Processes one control interval in the simulation."""
        self.sim.run(control_signal)

        current_state: Tuple = self._capture_environment_state()
        performance_score: float = self._determine_performance(current_state)

        # Set the number of vehicles on inbound roads in the new state
        n_west_east_vehicles, n_south_north_vehicles = current_state[1], current_state[2]
        self._vehicles_on_inbound_roads = n_west_east_vehicles + n_south_north_vehicles

        # Whether a terminal state is reached
        simulation_ended: bool = self.sim.completed

        # Whether a truncation condition is satisfied (GUI closed)
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
        Performance metric: penalize queue length (negative reward for waiting vehicles).
        This incentivizes the agent to minimize total waiting vehicles.
        """
        traffic_signal_state, n_direction_1_vehicles, n_direction_2_vehicles, non_empty_junction = state_observation
        
        # Negative reward for each vehicle waiting in queue
        total_queue = n_direction_1_vehicles + n_direction_2_vehicles
        reward = -total_queue
        
        return float(reward)

    def restart_environment(self, enable_display: bool = False) -> Tuple:
        """Resets traffic simulation and returns initial conditions."""
        self.sim = two_way_intersection_setup(self.max_gen)
        if enable_display:
            self.sim.init_gui()
        starting_state = self._capture_environment_state()
        self._vehicles_on_inbound_roads = 0  # Reset the counter
        return starting_state

    def retrieve_current_conditions(self) -> Tuple:
        """Provides current environmental observation."""
        return self._capture_environment_state()

    def assess_state_performance(self, environmental_state: Tuple) -> float:
        """External method for performance evaluation."""
        return self._determine_performance(environmental_state)
    
    # Mantieni compatibilità con i nomi originali se necessario
    @property
    def action_set(self):
        return self.action_space
    
    @property
    def traffic_model(self):
        return self.sim