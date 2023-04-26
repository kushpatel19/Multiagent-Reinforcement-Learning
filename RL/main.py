# Import required libraries
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import agentenv_module
import msvcrt  # Required for reading key presses on Windows

# Define the ID of the custom gym environment
env_id = 'AgentEnv-v1'
# Create a gym environment using the specified ID
env = gym.make(env_id)
# Create a vectorized gym environment
env = make_vec_env(env_id, n_envs=1)

# Define and train the agent using PPO algorithm and MlpPolicy
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=5e-5,     # Learning rate for the optimizer
            n_epochs=100,           # Number of times to iterate over the entire dataset
            clip_range=0.1,         # Clipping parameter for the surrogate loss
            gamma=0.99,             # Discount factor for future rewards
            gae_lambda=0.95,        # Lambda parameter for Generalized Advantage Estimation (GAE)
            batch_size=128,         # Size of the minibatch for each optimization step
            n_steps=2048,           # Number of steps to run in each environment per rollout
            ent_coef=0.01           # Added entropy coefficient
            )

# model.learn(total_timesteps=1000000, progress_bar=True)
# model.save('ppo_agentenv_weights_1M_4Agent_PPT')
model.load('ppo_agentenv_weights_10M_4Agent_PPT.zip')

# Evaluate the trained model
total_reward =0
obs = env.reset()

while True:
    action, _ = model.predict(obs)            # Get the predicted action from the model
    obs, reward, done, _ = env.step(action)   # Take a step in the environment with the predicted action
    env.render()                              # Render the environment
    total_reward += reward                    # Add the reward to the total reward
    if done or msvcrt.kbhit():                # Check if the episode is done or a key has been pressed to exit the loop
        print("Total reward:", total_reward)  # Print the total reward
        break

# Close the environment
env.close()


# nume = 5
# for i in range(nume):
#     obs = env.reset()
#     while True:
#         action, _ = model.predict(obs)r̥
#         obs, reward, done, info = env.step(action)
#         print("Reward:", info[3])
#         env.render()  # display the screen
#         if done:
#             print("Reward:", info[0]['episode']['r'])
#             obs = env.reset()
#             #print("Num episodes:", info[0]['episode']['l'])

#             # Save the model weights
#         # model.save('ppo_agentenv_weights')

#             # Close the environment
#             env.close()
#             break
# # Close the environment
# env.close()
