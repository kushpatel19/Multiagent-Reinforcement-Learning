# Import required libraries
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import agentenv_module

# Define the ID of the custom gym environment
env_id = 'AgentEnv-v1'
# Create a gym environment using the specified ID
env = gym.make(env_id)
# Create a vectorized gym environment
env = make_vec_env(env_id, n_envs=1)

# Define hyperparameters to test
learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]       # Learning rate for the optimizer
n_epochs_list  = [50, 100, 200, 300, 400]             # Number of times to iterate over the entire dataset
clip_ranges    = [0.1, 0.2, 0.3, 0.4, 0.5]            # Clipping parameter for the surrogate loss
gammas         = [0.1, 0.5, 0.9, 0.95, 0.99]          # Discount factor for future rewards
gae_lambdas    = [0.8, 0.9, 0.95, 0.98, 0.99]         # Lambda parameter for Generalized Advantage Estimation (GAE)
batch_sizes    = [32, 64, 128, 256, 512]              # Size of the minibatch for each optimization step
n_steps_list   = [2048, 4096, 8192, 16384, 32768]     # Number of steps to run in each environment per rollout
ent_coefs      = [0.01, 0.1, 1.0, 10.0, 100.0]        # Added entropy coefficient
vf_coefs       = [0.5, 1.0, 2.0, 5.0, 10.0]

results = []  # Store the results in a list

# Loop through all combinations of hyperparameters
for lr in learning_rates:
    for n_epochs in n_epochs_list:
        for clip_range in clip_ranges:
            for gamma in gammas:
                for gae_lambda in gae_lambdas:
                    for batch_size in batch_sizes:
                        for n_steps in n_steps_list:
                            for ent_coef in ent_coefs:
                                for vf_coef in vf_coefs:
                                    # Define and train the agent using PPO algorithm and MlpPolicy
                                    model = PPO("MlpPolicy", env, verbose=1,
                                                learning_rate=lr,
                                                n_epochs=n_epochs,
                                                clip_range=clip_range,
                                                gamma=gamma,
                                                gae_lambda=gae_lambda,
                                                batch_size=batch_size,
                                                n_steps=n_steps,
                                                ent_coef=ent_coef,
                                                vf_coef=vf_coef)

                                    model.learn(total_timesteps=100000, progress_bar=True)
                                    # model.save(f'ppo_agentenv_weights_100k_4Agent_PPT_{lr}{n_epochs}{clip_range}{gamma}{gae_lambda}{batch_size}{n_steps}{ent_coef}{vf_coef}')
                                    # model.load(f'ppo_agentenv_weights_100k_4Agent_PPT_{lr}{n_epochs}{clip_range}{gamma}{gae_lambda}{batch_size}{n_steps}{ent_coef}{vf_coef}.zip')

                                    # Evaluate the trained model
                                    total_reward = 0
                                    obs = env.reset()
                                    while True:
                                        # Choose an action using the trained model
                                        action, _ = model.predict(obs)
                                        # Execute the action and get the reward and new state
                                        obs, reward, done, _ = env.step(action)
                                        # env.render()
                                        total_reward += reward
                                        # Stop the episode if the agent reaches a terminal state
                                        if done:
                                            # print(f'Total reward for lr={lr}, n_epochs={n_epochs}, clip_range={clip_range}, gamma={gamma}, gae_lambda={gae_lambda}, batch_size={batch_size}, n_steps={n_steps}, ent_coef={ent_coef}, vf_coef={vf_coef}:', total_reward)
                                            results.append({'learning_rate': lr,
                                                    'n_epochs': n_epochs,
                                                    'clip_range': clip_range,
                                                    'gamma': gamma,
                                                    'gae_lambda': gae_lambda,
                                                    'batch_size': batch_size,
                                                    'n_steps': n_steps,
                                                    'ent_coef': ent_coef,
                                                    'vf_coef': vf_coef,
                                                    'total_reward': total_reward[0]})
                                            break
                                    # Close the environment
                                    env.close()

# # Print the results
# for result in results:
#     print(result)

# print(results)
# # Find the hyperparameters with the highest total reward
# max_reward = max(results, key=lambda x: x['total_reward'])
# print(f"\nHyperparameters with highest total reward: {max_reward}")

# Convert the results to a Pandas DataFrame
results_df = pd.DataFrame(results)
# print(results_df)

# Use Seaborn to plot the effect of each hyperparameter on the total reward
sns.catplot(x='learning_rate', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='n_epochs', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='clip_range', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='gamma', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='gae_lambda', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='batch_size', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='n_steps', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='ent_coef', y='total_reward', data=results_df, kind='bar')
sns.catplot(x='vf_coef', y='total_reward', data=results_df, kind='bar')
plt.show()


# Sort the results by total_reward in descending order and select the top 10
top_10 = results_df.sort_values('total_reward', ascending=False).head(10)

# Print the table of top 10 hyperparameter combinations
table = tabulate(top_10, headers='keys', tablefmt='psql')
print(table)
