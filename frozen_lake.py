import gym
import numpy as np


env = gym.make('FrozenLake-v0')
env.reset()


#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Set learning parameters
LEARNING_RATE = .8
DISCOUNT_FACTOR = .95

NUM_EPISODES = 2000

score_episode = 0
scores = []

for i_episode in range(NUM_EPISODES):
    state = env.reset()
    if score_episode > 0:
    	scores.append(score_episode)
    	score_episode = 0
    
    for t in range(1000):
        env.render()
        print('state', state)
        #action = env.action_space.sample()
        action= np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)))
        #action = np.argmax(Q[state,:]) 
        state_updated, reward, done, info = env.step(action)
        score_episode += reward
        Q[state,action] = Q[state, action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q[state_updated,:]) - Q[state, action])
        state = state_updated
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
print('scores', scores)