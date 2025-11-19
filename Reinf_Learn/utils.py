from Reinf_Learn import Environment, Q_Learn_agent

# Hyperparameter configuration
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.1

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
    
    for episode_num in range(1, total_episodes + 1):
        current_observation = simulation_env.initialize(display)
        total_reward = 0
        terminated = False

        while not terminated:
            action_taken = model.select_action(current_observation)
            new_observation, reward, terminated, interrupted = simulation_env.execute(action_taken)
            
            if interrupted:
                raise SystemExit("Simulation interrupted")
            
            model.learn(current_observation, action_taken, new_observation, reward)
            current_observation = new_observation
            total_reward += reward

    store_q_data(save_location, model.q_data)
    print("Training session completed")

def run_evaluation_session(model, simulation_env, total_episodes: int, display: bool = False):
    """Assesses trained model performance"""
    print(f"\nEvaluating model over {total_episodes} episodes...")
    
    total_wait_time = 0
    collision_episodes = 0
    completed_episodes = 0

    for episode_num in range(1, total_episodes + 1):
        current_observation = simulation_env.initialize(display)
        episode_reward = 0
        episode_collisions = 0
        terminal_state = False

        while not terminal_state:
            action_taken = model.select_action(current_observation)
            current_observation, reward, terminal_state, interrupted = simulation_env.execute(action_taken)
            
            if interrupted:
                raise SystemExit("Simulation interrupted")
            
            episode_reward += reward
            episode_collisions += simulation_env.collision_occurred

        if episode_collisions > 0:
            print(f"Episode {episode_num}: Collisions detected - {episode_collisions}")
            collision_episodes += 1
        else:
            avg_wait = simulation_env.average_waiting_time
            total_wait_time += avg_wait
            print(f"Episode {episode_num}: Average wait - {avg_wait:.1f}s")

    completed_episodes = total_episodes - collision_episodes
    
    print(f"\nEvaluation Results ({total_episodes} episodes):")
    if completed_episodes > 0:
        print(f"Average waiting time: {total_wait_time/completed_episodes:.1f}s")
    print(f"Collision frequency: {collision_episodes/total_episodes:.1%}")

def launch_q_learning_simulation(episode_count: int, visualize: bool):
    """Primary controller for Q-learning simulation"""
    sim_env = Environment()
    action_options = sim_env.available_actions
    
    q_model = Q_Learn_agent(
        learning_parameter=ALPHA,
        exploration_parameter=EPSILON,
        discount_parameter=GAMMA,
        action_space=action_options
    )
    
    training_cycles = 10000
    model_storage_path = f"ReinforcementLearning/traffic_model_{training_cycles}.dat"
    
    # run_training_session(q_model, sim_env, model_storage_path, training_cycles, False)
    
    saved_q_data = retrieve_q_data(model_storage_path)
    q_model.q_data = saved_q_data
    
    run_evaluation_session(q_model, sim_env, episode_count, visualize)