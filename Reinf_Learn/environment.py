from typing import Tuple

from TrafficSimulator import two_way_intersection_setup 

class Environment:
    """RL environment for managing traffic flow at intersections."""
    
    def __init__(self):
        self.action_set = [0, 1]
        self.traffic_model = None
        self.max_vehicles = 50
        self.previous_queue_size = 0

    def perform_step(self, control_signal) -> Tuple[Tuple, float, bool, bool]:
        """Processes one control interval in the simulation."""
        self.traffic_model.advance_simulation(control_signal)

        current_state = self._capture_environment_state()
        performance_score = self._determine_performance(current_state)

        # Maintain history for performance comparison
        _, lane_a_queue, lane_b_queue, _ = current_state
        self.previous_queue_size = lane_a_queue + lane_b_queue

        simulation_ended = self.traffic_model.simulation_complete
        visualization_terminated = self.traffic_model.display_terminated

        return current_state, performance_score, simulation_ended, visualization_terminated

    def _capture_environment_state(self) -> Tuple:
        """
        Captures current intersection status:
        (signal_state, lane_a_count, lane_b_count, internal_traffic_present)
        """
        state_components = []

        for controller in self.traffic_model.signal_units:
            intersection_state = []

            # Traffic signal state
            signal_state = controller.current_phase_pattern[0]
            intersection_state.append(signal_state)

            # Vehicle accumulation on entry lanes
            for lane_group in controller.entry_lanes:
                accumulated_vehicles = sum(len(lane.vehicles) for lane in lane_group)
                intersection_state.append(accumulated_vehicles)

            # Estimate vehicles within intersection
            queue_a, queue_b = intersection_state[1], intersection_state[2]

            departing_traffic = sum(
                len(self.traffic_model.road_segments[segment_id].vehicles)
                for segment_id in self.traffic_model.departure_segments
            )

            total_active_vehicles = self.traffic_model.active_vehicles
            approaching_traffic = queue_a + queue_b
            intersection_traffic = total_active_vehicles - departing_traffic - approaching_traffic

            intersection_state.append(intersection_traffic > 0)
            state_components.append(intersection_state)

        return tuple(state_components[0])  # Single intersection setup

    def _determine_performance(self, state_observation: Tuple) -> float:
        """
        Performance metric: reduction in queued vehicles.
        """
        _, current_a, current_b, _ = state_observation
        present_total = current_a + current_b
        improvement = self.previous_queue_size - present_total
        return float(improvement)

    def restart_environment(self, enable_display: bool = False) -> Tuple:
        """Resets traffic simulation and returns initial conditions."""
        self.traffic_model = two_way_intersection_setup(self.max_vehicles)

        if enable_display:
            self.traffic_model.launch_visual_interface()

        starting_state = self._capture_environment_state()
        self.previous_queue_size = 0
        return starting_state

    def retrieve_current_conditions(self) -> Tuple:
        """Provides current environmental observation."""
        return self._capture_environment_state()

    def assess_state_performance(self, environmental_state: Tuple) -> float:
        """External method for performance evaluation."""
        return self._determine_performance(environmental_state)