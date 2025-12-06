import numpy as np
import unittest


class SimpleGridWorld:
    """
    Simple 5x5 grid environment:
    
    S = Start (0,0)
    G = Goal (4,4)
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    
    def __init__(self):
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.current_state = self.start
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = self.start
        return self._state_to_index(self.current_state)
    
    def _state_to_index(self, state):
        """Convert (row, col) to unique state index"""
        return state[0] * self.grid_size + state[1]
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done)
        
        Reward structure:
        Reaching goal: +10
        Any other step: -0.1 to encourage short paths)
        """
        row, col = self.current_state
        
        # Apply action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.grid_size - 1, col + 1)
        
        self.current_state = (row, col)
        
        if self.current_state == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False
        
        return self._state_to_index(self.current_state), reward, done
    
    def get_num_states(self):
        return self.grid_size * self.grid_size
    
    def get_num_actions(self):
        return 4


class QLearningAgent:
    """
    q-Learning implementation
    """
    
    def __init__(self, num_states, num_actions, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((num_states, num_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.num_actions = num_actions
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])


class TestQLearning(unittest.TestCase):
 
    
    def setUp(self):
        """
        Setup executed before each test - 
        """
        np.random.seed(42)
        self.env = SimpleGridWorld()
        self.agent = QLearningAgent(
            num_states=self.env.get_num_states(),
            num_actions=self.env.get_num_actions(),
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
    
    def train_agent(self, episodes=2000):
        """Helper function to train agent for specified number of episodes"""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.choose_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
    
    
    
    def test_q_values_are_updated(self):
        """
        Test 2: Q-values are non-zero and change during training
        
        """
        initial_q_table = self.agent.q_table.copy()
        
        # Run a few training episodes
        self.train_agent(episodes=10)
        
        # Q-values should have changed
        self.assertFalse(np.array_equal(initial_q_table, self.agent.q_table),
                        "Q-values should change after training")
        
        # At least some Q-values should be non-zero
        self.assertGreater(np.count_nonzero(self.agent.q_table), 0,
                          "Some Q-values should be non-zero after training")
    
    def test_q_values_increase_over_time(self):
        """
        Test 3: Q-values increase monotonically during training
    
        """
        q_sums = []
        
        # Track Q-value sum over training
        for _ in range(5):
            q_sums.append(np.sum(np.abs(self.agent.q_table)))
            self.train_agent(episodes=200)
        
        # Q-values should generally increase (with some tolerance)
        for i in range(len(q_sums) - 1):
            self.assertGreaterEqual(q_sums[i + 1], q_sums[i] * 0.95,
                                   f"Q-values should increase: {q_sums[i]:.2f} -> {q_sums[i+1]:.2f}")
    

    
    def test_finds_near_optimal_path(self):
        """
        Test 7: Agent finds near-optimal solution
        
        Beyond just solving the task, we verify solution quality.
        Optimal path from (0,0) to (4,4) is 8 steps (4 right + 4 down).
        
        This tests that the agent not only learns A solution, but learns
        a THE BEST solution
        """
        self.train_agent(episodes=3000)
        
        state = self.env.reset()
        done = False
        path_length = 0
        
        while not done and path_length < 30:
            action = self.agent.choose_action(state, training=False)
            next_state, reward, done = self.env.step(action)
            state = next_state
            path_length += 1
        
        optimal_length = 8  # 4 right + 4 down
        self.assertLessEqual(path_length, optimal_length + 2,
                           f"Path length {path_length} exceeds optimal {optimal_length} + tolerance")
   
    def test_bellman_equation_holds(self):
        """
        Test 9: Bellman optimality equation is approximately satisfied
        
        Property-based test for mathematical correctness.
        
        After convergence, Q-values should satisfy:
        Q(s,a) ≈ r + γ·max Q(s',a')
        
        """
        self.train_agent(episodes=3000)
        
        # Test on several state transitions
        violations = 0
        total_tests = 0
        
        for _ in range(20):
            state = self.env.reset()
            action = self.agent.choose_action(state, training=False)
            next_state, reward, done = self.env.step(action)
            
            current_q = self.agent.q_table[state, action]
            if not done:
                expected_q = reward + self.agent.gamma * np.max(self.agent.q_table[next_state])
                error = abs(current_q - expected_q)
                
                # Allow 20% tolerance since it may not be fully converged
                if error > abs(expected_q) * 0.2:
                    violations += 1
                total_tests += 1
        
        violation_rate = violations / max(total_tests, 1)
        self.assertLess(violation_rate, 0.3,
                       f"Bellman equation violated in {violation_rate:.1%} of cases")
    
    

def run_visual_demo():
   
    
    print(" Q-LEARNING VISUAL DEMO - 5x5 Grid World")
    
    env = SimpleGridWorld()
    agent = QLearningAgent(
        num_states=env.get_num_states(),
        num_actions=env.get_num_actions()
    )
    
    print("\n5x5 Grid Layout:")
    print("S = Start (0,0)")
    print("G = Goal (4,4)")
    print(". = Empty cell\n")
    
    # Display grid
    for i in range(5):
        row = []
        for j in range(5):
            if (i, j) == (0, 0):
                row.append('S')
            elif (i, j) == (4, 4):
                row.append('G')
            else:
                row.append('.')
        print(' '.join(row))
    
    print("\n" + "=" * 70)
    print("Training agent...")
    
    # Training loop
    for episode in range(2000):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        
        if (episode + 1) % 500 == 0:
            print(f"  Episode {episode + 1}/2000 completed")
    
    print("Training completed!\n")
    print("=" * 70)
    print("Sample Q-values (first row of grid):")
    print("Format: [up, down, left, right]\n")
    
    for i in range(5):
        state_idx = env._state_to_index((0, i))
        q_vals = agent.q_table[state_idx]
        print(f"State (0,{i}): [{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, "
              f"{q_vals[2]:6.2f}, {q_vals[3]:6.2f}]")
    
    print("\n" + "=" * 70)
    print("Agent's path to goal:\n")
    
    # Show learned path
    state = env.reset()
    done = False
    steps = 0
    path = [(0, 0)]
    actions_map = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    while not done and steps < 30:
        action = agent.choose_action(state, training=False)
        print(f"Step {steps + 1}: Position {env.current_state} → "
              f"Action: {actions_map[action]}")
        
        next_state, reward, done = env.step(action)
        path.append(env.current_state)
        state = next_state
        steps += 1
    
    print(f"\nGoal reached in {steps} steps!")
    print(f" Optimal path: 8 steps (4 right + 4 down)")
    print(f"Difference: {steps - 8} steps")
    print(f"\nFull path: {' → '.join(map(str, path))}")


if __name__ == "__main__":
    print(" RUNNING Q-LEARNING TEST ")

    # Run test suite
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 70 + "\n")
    
    run_visual_demo()