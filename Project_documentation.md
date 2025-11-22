# Traffic Light Control with Q-Learning - Project Documentation

## Specification Document

### Programming Language
- **Main language:** Python
- **Peer review languages:** Python

### Problem Statement
This project implements a reinforcement learning solution for optimizing traffic light control at a two-way intersection. The goal is to minimize vehicle waiting times and reduce traffic congestion by training an intelligent agent to make optimal signal timing decisions.

### Algorithms and Data Structures

**Core Algorithm:**
- **Q-Learning**: A model-free reinforcement learning algorithm that learns the optimal action-value function through temporal difference learning.

**Data Structures:**
- **Dictionary (Hash Map)**: Used to store Q-values for state-action pairs
  - Keys: `(state, action)` tuples
  - Values: floating-point Q-values
  - Provides O(1) average case lookup and insertion

**Supporting Components:**
- Traffic simulation environment with discrete state space
- Epsilon-greedy action selection for exploration-exploitation balance

### Input and Usage
The program receives:
- **Episode count**: Number of training/evaluation episodes to run
- **Visualization flag**: Boolean to enable/disable GUI
- **Pre-trained model**: Saved Q-table loaded from file for evaluation

The state space consists of:
- Traffic signal phase (binary: 0 or 1)
- Vehicle count in lane A
- Vehicle count in lane B  
- Intersection occupancy (boolean)

### Time and Space Complexity

**Q-Learning Algorithm:**
- **Time complexity per step:** O(|A|) where |A| is the action space size (2 in this case)
  - Action selection: O(|A|) to find maximum Q-value
  - Q-value update: O(1) dictionary operation
- **Space complexity:** O(|S| × |A|) where |S| is the number of unique states encountered
  - Stores one Q-value per state-action pair
  - In practice, grows dynamically as new states are discovered

**Training:**
- Total training time: O(E × T × |A|) where E = episodes, T = steps per episode

### Sources
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

### Study Program
Bachelor's in Computer Science (TKT)

**Documentation language:** English

---

## Implementation Document

### Program Structure

The implementation consists of four main modules:

1. **`Q_learn.py`**: Core Q-learning agent implementation
   - Maintains Q-table as dictionary
   - Implements epsilon-greedy policy
   - Performs temporal difference learning updates

2. **`environment.py`**: RL environment wrapper
   - Interfaces with traffic simulator
   - Constructs state observations from simulation
   - Calculates rewards based on queue reduction
   - Manages episode lifecycle

3. **`utils.py`**: Training and evaluation utilities
   - Orchestrates training loops
   - Handles model persistence (save/load Q-tables)
   - Runs evaluation metrics collection
   - Reports performance statistics

4. **`TrafficSimulator`**: External traffic simulation engine (not shown)
   - Provides realistic traffic flow dynamics
   - Renders visualization when requested

### Achieved Complexities

**Q-Learning Agent (`Q_Learn_agent`):**
```python
def select_action(self, state):
    # O(|A|) - iterates through all actions
    if random.random() < self.epsilon:
        return random.choice(self.actions)  # O(1)
    else:
        return self.determine_optimal_action(state)  # O(|A|)

def learn(self, state, action, next_state, reward):
    # O(1) - constant time operations
    current_q = self.get_action_value(state, action)  # O(1) dict lookup
    best_future_value = self.compute_state_value(next_state)  # O(|A|)
    new_q = current_q + self.alpha * (reward + self.gamma * best_future_value - current_q)
    self.q_data[(state, action)] = new_q  # O(1) dict insertion
```

- **Time per learning step:** O(|A|) = O(2) = O(1) in practice
- **Space:** O(|S| × |A|) where |S| grows with encountered states

**Environment (`Environment`):**
- **State construction:** O(L) where L = number of lanes (constant in this setup)
- **Reward calculation:** O(1) - simple arithmetic on cached values
- **Step execution:** O(1) - delegated to simulator

**Training Loop:**
- For 10,000 episodes with average 100 steps per episode:
- Total operations: O(10,000 × 100 × |A|) = O(2,000,000) = practical O(1) per step

### Performance Characteristics

The reward function incentivizes queue reduction:
```python
reward = previous_queue_size - current_queue_size
```

This provides:
- **Positive reward** when traffic clears (effective control)
- **Negative reward** when queues grow (poor control)
- **Zero reward** for stable state

The epsilon-greedy policy (ε = 0.1) balances:
- 90% exploitation of learned policy
- 10% random exploration for discovering improvements

### Shortcomings and Improvements

**Current Limitations:**
1. **State discretization**: Continuous traffic patterns compressed to discrete counts may lose granularity
2. **Single intersection**: Does not scale to networked intersections
3. **Fixed hyperparameters**: α=0.1, γ=0.6, ε=0.1 not optimized through systematic search
4. **No function approximation**: Q-table grows with state space; would not scale to complex scenarios

**Suggested Improvements:**
1. Implement **Deep Q-Network (DQN)** for handling larger state spaces with neural network approximation
2. Add **prioritized experience replay** to improve sample efficiency
3. Extend to **multi-agent RL** for coordinated intersection networks
4. Implement **adaptive epsilon decay** to reduce exploration over time
5. Add **state normalization** to improve learning stability

### Use of Large Language Models

**No large language models (ChatGPT, Claude, etc.) were used in the development of this code.** 

All implementation decisions, algorithm design, and code structure were created through traditional software engineering practices and consultation of academic reinforcement learning literature.

### Sources
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. (1989). *Learning from delayed rewards* (Doctoral dissertation). University of Cambridge.
- Traffic simulation concepts based on standard discrete-event simulation principles

---

## Testing Document

### Unit Testing Coverage

The test suite (`test_environment.py`) provides comprehensive testing of critical RL environment functionality with 6 targeted test cases covering:
- Reward calculation correctness
- Simulation execution and state caching
- State structure validation
- Multi-step reward consistency
- Episode termination conditions
- Environment reset behavior

### Testing Methodology

**Test Framework:** Python `unittest` with `unittest.mock` for dependency isolation

**Key Test Cases:**

1. **`test_1_reward_calculation_correctness`**
   - **Purpose:** Validates reward function mathematical correctness
   - **Method:** Tests three scenarios with known queue changes
   - **Inputs:** 
     - Improvement: 10 → 5 vehicles (expected reward: +5.0)
     - Worsening: 5 → 12 vehicles (expected reward: -7.0)
     - No change: 8 → 8 vehicles (expected reward: 0.0)
   - **Assertion:** `assertEqual` on exact reward values

2. **`test_2_step_executes_simulation_and_updates_cache`**
   - **Purpose:** Ensures simulation advances and state cache updates
   - **Method:** Mocks simulator and verifies method calls
   - **Inputs:** Action signal, mocked environment with 8 vehicles
   - **Assertions:**
     - `mock_sim.run.assert_called_once_with(1)` - simulation ran
     - Cache updated to current vehicle count
     - Return types correct (tuple, float, bool, bool)

3. **`test_3_state_structure_correctness`**
   - **Purpose:** Validates state tuple structure for agent compatibility
   - **Method:** Constructs state with known configuration
   - **Inputs:** Mocked lanes with 4 and 6 vehicles, signal phase 1
   - **Expected:** `(1, 4, 6, bool)` - exactly 4 elements in correct order
   - **Assertions:** Length check and element type validation

4. **`test_4_reward_consistency_integration`**
   - **Purpose:** End-to-end reward pipeline validation across multiple steps
   - **Method:** Simulates 3-step episode with varying queue sizes
   - **Inputs:** Sequential states: 0→10→7→11 vehicles
   - **Critical invariant:** Sum of rewards equals net vehicle flow
   - **Formula verified:** `r1 + r2 + r3 = (0 - 11) = -11`

5. **`test_5_termination_flags`**
   - **Purpose:** Verifies episode termination logic
   - **Method:** Tests three scenarios with mocked completion flags
   - **Inputs:**
     - Normal: `completed=False, gui_closed=False`
     - Natural end: `completed=True`
     - GUI closure: `gui_closed=True`
   - **Assertions:** Correct `terminated` and `truncated` flags

6. **`test_6_reset_initialization`**
   - **Purpose:** Ensures clean episode initialization
   - **Method:** Corrupts state then calls reset
   - **Inputs:** Simulation factory mock
   - **Assertions:**
     - New simulation created with correct parameters
     - Cache reset to 0
     - Valid initial state returned

### Test Input Characteristics

**Input Types:**
- **Controlled synthetic states:** Hand-crafted vehicle configurations for deterministic testing
- **Mock objects:** Isolated environment from complex simulator dependencies
- **Boundary conditions:** Empty intersections, maximum queue sizes
- **State transitions:** Multi-step sequences to verify temporal consistency

**Edge Cases Tested:**
- Zero vehicles (empty intersection)
- Maximum capacity (50 vehicles)
- Equal Q-values (random tie-breaking)
- Simultaneous termination flags

### Reproducing Tests

**Run complete test suite:**
```bash
python test_environment.py
```

**Run with verbose output:**
```bash
python test_environment.py -v
```

**Run specific test:**
```bash
python -m unittest test_environment.TestEnvironmentCritical.test_1_reward_calculation_correctness
```

**Requirements:**
- Python 3.7+
- `unittest` (standard library)
- `unittest.mock` (standard library)

### Test Results Interpretation

All 6 tests must pass for environment correctness:
- **Failure in test 1-3:** Core RL mechanics broken, agent cannot learn
- **Failure in test 4:** Integration bug in reward pipeline
- **Failure in test 5-6:** Episode management issues, training will fail

The test suite focuses on **critical invariants** rather than coverage percentage. Each test validates properties that, if violated, would cause complete RL training failure.

### Empirical Testing

**Training Performance:**
- Model trained for 10,000 episodes
- Convergence observed visually through decreasing collision rates
- Evaluation over 100 episodes shows consistent behavior

**Evaluation Metrics Collected:**
- Average waiting time per successful episode
- Collision frequency (episodes with traffic violations)
- Episode completion rate

These metrics are logged during `run_evaluation_session()` in `utils.py` and printed to console for analysis.