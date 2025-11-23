import random

class Q_Learn:
    """Implementation of Q-learning reinforcement algorithm"""
    
    def __init__(self, learning_parameter, exploration_parameter, discount_parameter, action_space):
        self.alpha = float(learning_parameter)
        self.epsilon = float(exploration_parameter)
        self.gamma = float(discount_parameter)
        self.actions = action_space
        self.q_data = {}

    def get_action_value(self, state, action):
        """Retrieves Q-value for state-action pair"""
        if (state, action) not in self.q_data:
            return 0.0
        return self.q_data.get((state, action), 0.0)

    def compute_state_value(self, state):
        """Calculates maximum value across possible actions in state"""
        if not self.actions:
            return 0.0
            
        action_values = [self.get_action_value(state, act) for act in self.actions]
        random.shuffle(action_values)  # Prevent bias for equal values
        return max(action_values) if action_values else 0.0

    def determine_optimal_action(self, state):
        """Identifies best action according to current policy"""
        if not self.actions:
            return None
            
        # Map actions to their values
        action_value_pairs = [(act, self.get_action_value(state, act)) for act in self.actions]
        max_value = max([self.get_action_value(state, act) for act in self.actions])
        
        # Select randomly among equally optimal actions
        best_actions = [act for act, val in action_value_pairs if val == max_value]
        return random.choice(best_actions)

    def select_action(self, state):
        """Chooses action using epsilon-greedy strategy"""
        if not self.actions:
            return None
            
        # Exploration case
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Exploitation case
        else:
            return self.determine_optimal_action(state)

    def learn(self, state, action, next_state, reward):
        """Updates Q-values using temporal difference learning"""
        current_q = self.get_action_value(state, action)
        best_future_value = self.compute_state_value(next_state)
        
        # Q-learning update rule: Q(s,a) = (1-α)*Q(s,a) + α*(r + γ*max_Q(s',a'))
        new_q = (1 - self.alpha) * current_q + self.alpha * (
            reward + self.gamma * best_future_value
        )
        
        self.q_data[(state, action)] = new_q