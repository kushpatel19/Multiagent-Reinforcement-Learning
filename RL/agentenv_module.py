# # Desired Motion for 1 agent

# import pygame

# # Initialize Pygame
# pygame.init()

# # Define some constants
# SCREEN_WIDTH = 640
# SCREEN_HEIGHT = 480
# BAR_WIDTH = 400
# BAR_HEIGHT = 10
# AGENT_SIZE = 50
# AGENT_ACCELERATION = 0.5  # pixels per millisecond squared
# AGENT_MAX_VELOCITY = 15  # pixels per millisecond

# # Set up the display
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("1D Motion Simulation")

# # Define the bar
# bar_rect = pygame.Rect((SCREEN_WIDTH - BAR_WIDTH) // 2, SCREEN_HEIGHT // 2,
#                         BAR_WIDTH, BAR_HEIGHT)

# # Define the agent
# agent_rect = pygame.Rect((SCREEN_WIDTH - AGENT_SIZE) // 2, bar_rect.top - AGENT_SIZE,
#                         AGENT_SIZE, AGENT_SIZE)
# agent_velocity = 0

# # Set up the clock
# clock = pygame.time.Clock()

# # Main game loop
# while True:
#     # Handle events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_1:
#                 agent_velocity += AGENT_ACCELERATION
#                 #agent_velocity = min(agent_velocity, AGENT_MAX_VELOCITY)
#             elif event.key == pygame.K_2:
#                 agent_velocity -= AGENT_ACCELERATION
#                 #agent_velocity = max(agent_velocity, -AGENT_MAX_VELOCITY)
#             elif event.key == pygame.K_q:
#                 pygame.quit()
#                 quit()
#         #elif event.type == pygame.KEYUP:
#             #if event.key in (pygame.K_1, pygame.K_2):
#                 #agent_velocity = 0

#     # Update the agent's position and velocity
#     agent_rect.move_ip(int(agent_velocity), 0)
#     agent_velocity = max(-AGENT_MAX_VELOCITY, min(agent_velocity, AGENT_MAX_VELOCITY))

#     # Keep the agent within the bounds of the bar
#     if agent_rect.left < bar_rect.left:
#         agent_rect.left = bar_rect.left
#         agent_velocity = 0
#     elif agent_rect.right > bar_rect.right:
#         agent_rect.right = bar_rect.right
#         agent_velocity = 0

#     # Clear the screen and draw the objects
#     screen.fill((255, 255, 255))
#     pygame.draw.rect(screen, (0, 0, 0), bar_rect)
#     pygame.draw.rect(screen, (255, 0, 0), agent_rect)

#     # Update the display
#     pygame.display.flip()

#     # Limit the frame rate
#     clock.tick(60)

########################################################################################################################

# #  Desired Motion for 4 agent without RL
# import pygame

# # Initialize Pygame
# pygame.init()

# # Define some constants
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600
# BAR_WIDTH = 600
# BAR_HEIGHT = 10
# AGENT_SIZE = 50
# AGENT_ACCELERATION = 0.5  # pixels per millisecond squared
# AGENT_MAX_VELOCITY = 15  # pixels per millisecond

# # Set up the display
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("1D Motion Simulation of 4 Agents")

# # Define the bars
# bar_rects = [
#     pygame.Rect((SCREEN_WIDTH - BAR_WIDTH) // 2, SCREEN_HEIGHT // 5 * (i+1), BAR_WIDTH, BAR_HEIGHT)
#     for i in range(4)
# ]

# # Define the agents
# agent_rects = [
#     pygame.Rect((SCREEN_WIDTH - AGENT_SIZE) // 2, bar_rects[i].top - AGENT_SIZE, AGENT_SIZE, AGENT_SIZE)
#     for i in range(4)
# ]
# agent_velocities = [0] * 4

# # Set up the clock
# clock = pygame.time.Clock()

# # Main game loop
# while True:
#     # Handle events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_1:
#                 agent_velocities[0] += AGENT_ACCELERATION
#             elif event.key == pygame.K_2:
#                 agent_velocities[0] -= AGENT_ACCELERATION
#             elif event.key == pygame.K_3:
#                 agent_velocities[1] += AGENT_ACCELERATION
#             elif event.key == pygame.K_4:
#                 agent_velocities[1] -= AGENT_ACCELERATION
#             elif event.key == pygame.K_5:
#                 agent_velocities[2] += AGENT_ACCELERATION
#             elif event.key == pygame.K_6:
#                 agent_velocities[2] -= AGENT_ACCELERATION
#             elif event.key == pygame.K_7:
#                 agent_velocities[3] += AGENT_ACCELERATION
#             elif event.key == pygame.K_8:
#                 agent_velocities[3] -= AGENT_ACCELERATION
#             elif event.key == pygame.K_q:
#                 pygame.quit()
#                 quit()

#     # Update the agents' positions and velocities
#     for i in range(4):
#         agent_rects[i].move_ip(int(agent_velocities[i]), 0)
#         agent_velocities[i] = max(-AGENT_MAX_VELOCITY, min(agent_velocities[i], AGENT_MAX_VELOCITY))

#         # Keep the agents within the bounds of their respective bars
#         if agent_rects[i].left < bar_rects[i].left:
#             agent_rects[i].left = bar_rects[i].left
#             agent_velocities[i] = 0
#         elif agent_rects[i].right > bar_rects[i].right:
#             agent_rects[i].right = bar_rects[i].right
#             agent_velocities[i] = 0

#     # Clear the screen and draw the objects
#     screen.fill((255, 255, 255))
#     for i in range(4):
#         pygame.draw.rect(screen, (0, 0, 0), bar_rects[i])
#         pygame.draw.rect(screen, (255, 0, 0), agent_rects[i])

#     # Update the display
#     pygame.display.flip()

#     # Limit the frame rate
#     clock.tick(60)



########################################################################################################################
# #   Desired Motion of 4 agent with RL Environment without obstacle
# import gym
# from gym import spaces
# # from gym.utils import seeding
# import numpy as np
# import pygame
# import time
# import sys

# # Constants for the window size, bar size, agent size, and FPS
# SCREEN_WIDTH = 800
# SCREEN_HEIGHT = 600
# BAR_WIDTH = 600
# BAR_HEIGHT = 10
# FPS = 30
# AGENT_SIZE = 50

# # Constants for the agent's acceleration, maximum velocity, and episode length
# AGENT_ACCELERATION = 0.5  # pixels per millisecond squared
# AGENT_MAX_VELOCITY = 15  # pixels per millisecond
# EPISODE_LENGTH = 1000  # milliseconds

# # Constants for the starting and desired positions of the agents
# SHAPE_COORDS = [(150, 300), (250, 230), (350, 160), (450, 90)]
# DESIRED_COORDS = [(250, 300), (250, 230), (250, 160), (250, 90)]

# class AgentEnv(gym.Env):
#     def _init_(self):
#         super()._init_()

#         # Set the action and observation spaces
#         self.action_space = spaces.Box(low=-AGENT_ACCELERATION, high=AGENT_ACCELERATION, shape=(4,), dtype=np.float32)
#         self.observation_space = spaces.Box(low=0, high=max(SCREEN_WIDTH,SCREEN_HEIGHT), shape=(8,), dtype=np.float32)
#         # Initialize the agent and bar rectangles and velocities
#         self.agent_rects = [
#             pygame.Rect((SCREEN_WIDTH - AGENT_SIZE) // 2, SHAPE_COORDS[i][1] - AGENT_SIZE, AGENT_SIZE, AGENT_SIZE)
#             for i in range(4)
#         ]
#         self.agent_velocities = [0] * 4
#         self.bar_rects = [
#             pygame.Rect((SCREEN_WIDTH - BAR_WIDTH) // 2, SHAPE_COORDS[i][1] - BAR_HEIGHT // 2, BAR_WIDTH, BAR_HEIGHT)
#             for i in range(4)
#         ]
#         # Initialize the current episode step and desired positions
#         self.current_episode_step = 0
#         self.desired_positions = np.array([coord[0] for coord in DESIRED_COORDS])
        
#     def reset(self):
#         # Reset the agent and bar rectangles and velocities
#         self.agent_rects = [
#             pygame.Rect((SCREEN_WIDTH - AGENT_SIZE) // 2, SHAPE_COORDS[i][1] - AGENT_SIZE, AGENT_SIZE, AGENT_SIZE)
#             for i in range(4)
#         ]
#         self.agent_velocities = [0] * 4
#         self.current_episode_step = 0
#         # Return the initial observation
#         return self._get_observation()
        
#     def step(self, action):
#         # Update the agents' positions and velocities based on the action
#         for i in range(4):
#             self.agent_velocities[i] += action[i]
#             self.agent_velocities[i] = max(-AGENT_MAX_VELOCITY, min(self.agent_velocities[i], AGENT_MAX_VELOCITY))
#             self.agent_rects[i].move_ip(int(self.agent_velocities[i]), 0)
            
#             # Keep the agents within the bounds of their respective bars
#             if self.agent_rects[i].left < self.bar_rects[i].left:
#                 self.agent_rects[i].left = self.bar_rects[i].left
#                 self.agent_velocities[i] = 0
#             elif self.agent_rects[i].right > self.bar_rects[i].right:
#                 self.agent_rects[i].right = self.bar_rects[i].right
#                 self.agent_velocities[i] = 0
        
#         # Calculate the reward based on the distance between current positions and desired positions
#         current_positions = np.array([rect.centerx for rect in self.agent_rects])
#         reward = -np.sum(np.abs(current_positions - self.desired_positions))          # Cooperative Task
        
#         # Check if the episode has ended
#         done = self.current_episode_step >= EPISODE_LENGTH
        
#         # Increment the current episode step
#         self.current_episode_step += 1
        
#         # Return the observation, reward, done flag, and info dict
#         observation = self._get_observation()
#         info = {}
        
#         return observation, reward, done, info

#     def render(self):
#         screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#         screen.fill((255, 255, 255))
        
#         # Draw the bars
#         for bar_rect in self.bar_rects:
#             pygame.draw.rect(screen, (0, 0, 0), bar_rect)
        
#         # Draw the agents
#         for agent_rect in self.agent_rects:
#             pygame.draw.rect(screen, (255, 0, 0), agent_rect)
        
#         pygame.display.flip()

#     def _get_observation(self):
#         # return np.array([rect.centerx for rect in self.agent_rects] + self.agent_velocities)
#         return np.array([rect.centerx for rect in self.agent_rects])

# # Register the environment with OpenAI Gym
# gym.register(
#     id='AgentEnv-v0',
#     entry_point='agentenv_module:AgentEnv',
# )


# if _name_ == '_main_':
#     env = AgentEnv()
#     env.reset()

#     clock = pygame.time.Clock()

#     # initialize the display
#     pygame.display.init()

#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()

#         # Get a random action and take a step in the environment
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)

#         # Render the environment
#         env.render()             # To see the window

#         # Sleep to control the frame rate
#         # clock.tick(90)
#         time.sleep(1.1/FPS)

#         # Reset the environment if the episode is done
#         if done:
#             env.reset()



# # if _name_ == '_main_':
# #     NUM_EPISODES = 1  # Number of episodes to run the loop for
# #     env = AgentEnv()
# #     total_reward = 0
# #     num_episodes = 0
    
# #     for i in range(NUM_EPISODES):
# #         env.reset()

# #         clock = pygame.time.Clock()

# #         # initialize the display
# #         pygame.display.init()

# #         while True:
# #             for event in pygame.event.get():
# #                 if event.type == pygame.QUIT:
# #                     pygame.quit()
# #                     sys.exit()

# #             # Get a random action and take a step in the environment
# #             action = env.action_space.sample()
# #             observation, reward, done, info = env.step(action)
# #             total_reward += reward

# #             # Render the environment
# #             env.render()             # To see the window

# #             # Sleep to control the frame rate
# #             # clock.tick(10/FPS)
# #             time.sleep(1.1/FPS)

# #             # Reset the environment if the episode is done
# #             if done:
# #                 num_episodes += 1
# #                 print(f"Episode {num_episodes} Reward: {total_reward}")
# #                 total_reward = 0  # Reset the total reward for next episode
# #                 break
# #     print(f"Total number of episodes: {num_episodes}")


#################################################################################################################

#   Desired Motion of 4 agent with RL Environment with obstacle
import gym
from gym import spaces
import numpy as np
import pygame
import time

# Constants for the window size, bar size, agent size, and FPS
SCREEN_WIDTH = 800 #800
SCREEN_HEIGHT = 600 #600
BAR_WIDTH = 600
BAR_HEIGHT = 10
FPS = 25
AGENT_SIZE = 20
OBSTACLE_RADIUS = 10

# Get a random angle in radians between 0 and pi
theta = np.random.uniform(low=0, high=np.pi)
OBSTACLE_SPEED_X = 8*np.sin(theta)
OBSTACLE_SPEED_Y = 8*np.cos(theta)

# Create global variables to keep track of the episode state
global flag
global done_e
flag = False
done_e = False

# Constants for the agent's acceleration, maximum velocity, and episode length
AGENT_ACCELERATION = 25  # pixels per millisecond squared
AGENT_MAX_VELOCITY = 15  # pixels per millisecond
EPISODE_LENGTH = 1000

# Constants for the starting and desired positions of the agents
SHAPE_COORDS = [(150, 300), (250, 230), (350, 160), (450, 90)]
DESIRED_COORDS = [(250, 300), (250, 230), (250, 160), (250, 90)]

class AgentEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Set the action and observation spaces
        self.action_space = spaces.Box(low=-AGENT_ACCELERATION, 
                                       high=AGENT_ACCELERATION, 
                                       shape=(4,), 
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, 
                                            high=max(SCREEN_WIDTH,SCREEN_HEIGHT), 
                                            shape=(18,), dtype=np.float32)

        # Initialize the agent and bar rectangles and velocities
        self.agent_rects = [
            pygame.Rect((SCREEN_WIDTH - AGENT_SIZE) // 2, 
                        SHAPE_COORDS[i][1] - AGENT_SIZE, 
                        AGENT_SIZE, 
                        AGENT_SIZE)
            for i in range(4)
        ]
        self.agent_velocities = [0] * 4
        # Initialize the bar and obstacle rectangles and velocities
        self.bar_rects = [
            pygame.Rect((SCREEN_WIDTH - BAR_WIDTH) // 2, 
                        SHAPE_COORDS[i][1] - BAR_HEIGHT // 2, 
                        BAR_WIDTH, 
                        BAR_HEIGHT)
            for i in range(4)
        ]

        self.obstacle_rect = pygame.Rect((SCREEN_WIDTH - OBSTACLE_RADIUS) // 3, 
                                         SCREEN_HEIGHT // 3, 
                                         OBSTACLE_RADIUS * 2, 
                                         OBSTACLE_RADIUS * 2)
        self.obstacle_velocity_x = OBSTACLE_SPEED_X
        self.obstacle_velocity_y = OBSTACLE_SPEED_Y

        # Initialize the current episode step and desired positions
        self.current_episode_step = 0
        self.desired_positions = np.array([coord[0] for coord in DESIRED_COORDS])
        self.reward = 0
        
    def reset(self):
        # Reset the agent and bar rectangles and velocities
        self.agent_rects = [
            pygame.Rect((SCREEN_WIDTH - AGENT_SIZE) // 2, 
                        SHAPE_COORDS[i][1] - AGENT_SIZE, 
                        AGENT_SIZE, 
                        AGENT_SIZE)
            for i in range(4)
        ]
        flag == False
        done_e == False
        self.agent_velocities = [0] * 4
        self.current_episode_step = 0
        # Reset the obstacle position and velocity
        self.obstacle_rect = pygame.Rect((SCREEN_WIDTH - OBSTACLE_RADIUS) // 3, 
                                         SCREEN_HEIGHT // 3, 
                                         OBSTACLE_RADIUS * 2, 
                                         OBSTACLE_RADIUS * 2)
        self.obstacle_velocity_x = OBSTACLE_SPEED_X
        self.obstacle_velocity_y = OBSTACLE_SPEED_Y
        # Return the initial observation
        return self._get_observation()
        
    def step(self, action):
        self.reward = 0
        global flag
        global done_e
        flag = False
        done_e = False
        # Update the agents' positions and velocities based on the action
        self.agent_velocities[0] += action[0]
        self.agent_velocities[1] += action[1]
        self.agent_velocities[2] += action[2]
        self.agent_velocities[3] += action[3]

        self.agent_velocities = np.clip(self.agent_velocities, -AGENT_MAX_VELOCITY, AGENT_MAX_VELOCITY)
        for i in range(4):
            self.agent_rects[i].x += int(self.agent_velocities[i])
            if self.agent_rects[i].left < self.bar_rects[i].left:
                self.agent_rects[i].left = self.bar_rects[i].left
                self.agent_velocities[i] = 0
            elif self.agent_rects[i].right > self.bar_rects[i].right:
                self.agent_rects[i].right = self.bar_rects[i].right
                self.agent_velocities[i] = 0

        # Update the obstacle position
        self.obstacle_rect.x -= self.obstacle_velocity_x
        self.obstacle_rect.y -= self.obstacle_velocity_y
        if self.obstacle_rect.right < 0:
            self.obstacle_rect.left = SCREEN_WIDTH
            self.obstacle_rect.centery = np.random.randint(0, SCREEN_HEIGHT)
            self.obstacle_velocity_x += 0.1

        # Increment the episode step counter
        self.current_episode_step += 1

        # Calculate the distances and penalties based on collision with the obstacle
        distances = self._calculate_obst_distances()        # Diff. b/w x-coordinate of agent & obstacle
        position_distances = self._calculate_distances()    # Diff. b/w x-coordinate of agent & desired position

        # Calculate the cooperative reward based on the distances and penalties
        rewards = np.zeros(4)
        reward_sum = 0

        penalties = np.zeros(4)
        for i in range(4):
            if self.obstacle_rect.colliderect(self.agent_rects[i]):
            # if self.agent_rects[i].center == self.obstacle_rect.center:
                penalties[i] = 1
                flag = True
                reward_sum -= 10000000

        for i in range(4):
            rel_velocity_agent_y = - OBSTACLE_SPEED_Y
            rel_velocity_agent_x = self.agent_velocities[i] - OBSTACLE_SPEED_X

            if distances[i]<=100:
                obstacle_reward = -1*np.exp(1.5 + (1 - (0.01*distances[i])**0.4))
                rewards[i]+= obstacle_reward
            else:
                distance_reward = 1*np.exp(1.5 + (1 - (0.01*position_distances[i])**0.4))
                rewards[i] += distance_reward       # add distance reward only if no obstacle is detected

            if position_distances[i]<=10:
                stay_reward = 100
                rewards[i] += stay_reward           # add a lower stay reward only if the distance is less than or equal to 10

            reward_sum = sum(rewards)

        # Add additional reward for staying in the desired position
        if not flag and position_distances.all() == 0:
            staying_reward = 10000
            reward_sum += staying_reward

        # Large penalty for collision with the obstacle
        if flag:
            self.reward += -0
        else:
            self.reward += reward_sum

        # Check if the episode is done
        done_e = self.current_episode_step >= EPISODE_LENGTH
        if (done_e==True):
            done = True
        else:
            done = False
        # Create the info dictionary
        info = {
            'distances': distances,
            'penalties': penalties,
            'rewards': self.reward,
            'reward_sum': reward_sum,
        }
        # Return the observation, reward, done flag, and empty info dictionary
        return self._get_observation(), self.reward, done, info


    def _calculate_distances(self):
            # Calculate the distances between the agents' current positions and desired positions [X - coordinate]
            positions = np.array([rect.centerx for rect in self.agent_rects])
            return np.abs(positions - self.desired_positions)

    def _calculate_obst_distances(self):
            # Calculate the absolute distances between the agents' center current positions and obstacle center positions
            positionsx = np.array([rect.centerx for rect in self.agent_rects])
            positionsy = np.array([rect.centery for rect in self.agent_rects])
            output = np.zeros(4)
            for i in range(4):
                output[i] = np.sqrt((positionsx[i] - self.obstacle_rect.centerx)**2 + (positionsy[i] - self.obstacle_rect.centery)**2)
            return np.abs(output)
    
    def render(self, _):
        # Initialize the pygame window and font
        pygame.init()
        font = pygame.font.Font(None, 36)
        # Set up the window and bar surface
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # Draw the agents, bar, and obstacle onto the bar surface
        for i in range(4):
            pygame.draw.rect(screen, (0, 0, 255), self.agent_rects[i])    # draw blue rectangle for each agent
        for i in range(4):
            pygame.draw.rect(screen, (0, 255, 0), self.bar_rects[i])      # draw green rectangle for each bar
        pygame.draw.rect(screen, (255, 0, 0), self.obstacle_rect)         # draw red rectangle for obstacle
        # Draw yellow circle for desired coordinates
        pygame.draw.circle(screen, color=(255,255,0), center=DESIRED_COORDS[0], radius=10)
        pygame.draw.circle(screen, color=(255,255,0), center=DESIRED_COORDS[1], radius=10)
        pygame.draw.circle(screen, color=(255,255,0), center=DESIRED_COORDS[2], radius=10)
        pygame.draw.circle(screen, color=(255,255,0), center=DESIRED_COORDS[3], radius=10)

        pygame.display.flip()  # update the screen
        time.sleep(1/FPS)      # delay to set the frame rate

    def _get_observation(self):
        ax = [self.agent_rects[i].x for i in range(4)]      # get x-coordinates of each agent
        ay = [self.agent_rects[i].y for i in range(4)]      # get y-coordinates of each agent
        obs = [self.obstacle_rect.x,self.obstacle_rect.y]   # get the x and y coordinates of the obstacle
        dx = [coord[0] for coord in DESIRED_COORDS[:4]]     # get x-coordinates of the desired coordinates
        dy = [coord[1] for coord in DESIRED_COORDS[:4]]     # get y-coordinates of the desired coordinates
        return np.concatenate((ax,ay,obs,dx,dy))            # concatenate all the information and return as a numpy array
    
# Register the environment with OpenAI Gym
gym.register(
    id='AgentEnv-v1',
    entry_point='agentenv_module:AgentEnv',
)



# if __name__ == '__main__':
#     NUM_EPISODES = 5  # Number of episodes to run the loop for
#     env = AgentEnv()
#     total_reward = 0
#     num_episodes = 0
#     for i in range(NUM_EPISODES):
#         env.reset()
#         steps = 0
#         clock = pygame.time.Clock()
#         done = False
#         # initialize the display
#         pygame.display.init()
#         while not done:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     sys.exit()
#             # Get a random action and take a step in the environment
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             total_reward += reward
#             # Render the environment
#             env.render( )             # To see the window
#             steps += 1
#             # Sleep to control the frame rate
#             # clock.tick(10/FPS)
#             time.sleep(1/FPS)
#             # Reset the environment if the episode is done
#             #if done:
#             #    num_episodes += 1
#             #    print(f"Episode {num_episodes} Reward: {total_reward}")
#             #    total_reward = 0  # Reset the total reward for next episode
#             #    break
#         num_episodes += 1
#         print(f"Episode {num_episodes} Reward: {total_reward}")
#         total_reward = 0  # Reset the total reward for next episode
#         print("Steps:" ,steps)
#     print(f"Total number of episodes: {num_episodes}")
