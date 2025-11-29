from .environment import Environment
from .Q_Learn import Q_Learn

# Hyperparameter configuration
ALPHA = 0.125
GAMMA = 0.5
EPSILON = 0.1
EPSILON_MIN = 0.01  # Minimum epsilon
EPSILON_DECAY = 0.995  # Decay rate per episode

def store_q_data(destination_path, q_data):
    """Persists Q-learning model data to storage"""
    with open(destination_path, 'w') as storage_file:
        storage_file.write(repr(q_data))

def retrieve_q_data(source_path):
    """Retrieves Q-learning model data from storage"""
    with open(source_path, 'r') as storage_file:
        return eval(storage_file.read().strip())

def run_training_session(model, simulation_env, save_location, total_episodes: int, display: bool = False):
    """Orchestrates the model training process"""
    print(f"\nStarting {total_episodes} training episodes...")
    
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode_num in range(1, total_episodes + 1):
        current_observation = simulation_env.restart_environment(enable_display=display)        
        total_reward = 0
        terminated = False
        step_count = 0
        
        # DEBUG: Print first episode details
        if episode_num == 1:
            print(f"\n=== DEBUG EPISODE 1 ===")
            print(f"Initial state: {current_observation}")

        while not terminated:
            action_taken = model.select_action(current_observation)
            new_observation, reward, terminated, interrupted = simulation_env.perform_step(action_taken)
            
            if interrupted:
                raise SystemExit("Simulation interrupted")
            
            model.learn(current_observation, action_taken, new_observation, reward)
            
            # DEBUG: Print first 5 steps of episode 1
            if episode_num == 1 and step_count < 5:
                print(f"\nStep {step_count}:")
                print(f"  Action: {action_taken}")
                print(f"  Old state: {current_observation}")
                print(f"  New state: {new_observation}")
                print(f"  Reward: {reward:.2f}")
            
            current_observation = new_observation
            total_reward += reward
            step_count += 1
        
        episode_rewards.append(total_reward)
        
        # Track best performance
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Epsilon decay
        if model.epsilon > EPSILON_MIN:
            model.epsilon *= EPSILON_DECAY
        
        if episode_num == 1:
            print(f"\nTotal steps in episode: {step_count}")
            print(f"Total reward: {total_reward:.2f}")
            print("======================\n")
        
        if episode_num % 100 == 0:
            avg_reward_last_100 = sum(episode_rewards[-100:]) / 100
            print(f"Episode {episode_num}/{total_episodes} - Reward: {total_reward:.2f} - "
                  f"Avg(100): {avg_reward_last_100:.2f} - Best: {best_reward:.2f} - "
                  f"Epsilon: {model.epsilon:.4f} - Steps: {step_count}")

    store_q_data(save_location, model.q_data)
    print(f"\nTraining completed!")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Average last 100 episodes: {sum(episode_rewards[-100:]) / 100:.2f}")
    print(f"Q-table size: {len(model.q_data)} state-action pairs")

    store_q_data(save_location, model.q_data)
    print("Training session completed")

def run_evaluation_session(model, simulation_env, total_episodes: int, display: bool = False):
    """Assesses trained model performance"""
    print(f"\nEvaluating model over {total_episodes} episodes...")
    
    total_reward_sum = 0
    episode_rewards = []

    for episode_num in range(1, total_episodes + 1):
        current_observation = simulation_env.restart_environment(enable_display=display)
        episode_reward = 0
        terminal_state = False

        while not terminal_state:
            action_taken = model.select_action(current_observation)
            current_observation, reward, terminal_state, interrupted = simulation_env.perform_step(action_taken)
            
            if interrupted:
                raise SystemExit("Simulation interrupted")
            
            episode_reward += reward

        episode_rewards.append(episode_reward)
        total_reward_sum += episode_reward
        print(f"Episode {episode_num}: Total reward: {episode_reward:.2f}")
    
    print(f"\nEvaluation Results ({total_episodes} episodes):")
    print(f"Average reward per episode: {total_reward_sum/total_episodes:.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Worst episode reward: {min(episode_rewards):.2f}")

def launch_q_learning_simulation(num_episodes: int, render: bool , mode: bool ):
    """Primary controller for Q-learning simulation"""
    sim_env = Environment()
    action_options = sim_env.action_set
    
    q_model = Q_Learn(
        learning_parameter=ALPHA,
        exploration_parameter=EPSILON,
        discount_parameter=GAMMA,
        action_space=action_options
    )
    
    training_cycles = 10000
    model_storage_path = f"model_{training_cycles}.dat"
    
    if mode == True:
        run_training_session(q_model, sim_env, model_storage_path, training_cycles, False)
        
    saved_q_data = retrieve_q_data(model_storage_path)
    q_model.q_data = saved_q_data
    
    run_evaluation_session(q_model, sim_env, num_episodes, render)