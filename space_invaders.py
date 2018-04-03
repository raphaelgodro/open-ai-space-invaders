import gym


# Set learning parameters
LEARNING_RATE = .8
DISCOUNT_FACTOR = .95
NUM_EPISODES = 2000



def state_to_scalar(state):
    state_scalar = state.reshape(state.shape[0] * state.shape[1] * state.shape[2], 1)
    return state_scalar


env = gym.make('SpaceInvaders-v0')
env.reset()
print('env.observation_space.n', env.observation_space.n)

Q = np.zeros([env.observation_space.n,env.action_space.n])

for i_episode in range(NUM_EPISODES):
    state = observation = env.reset()
    state = state_to_scalar(state)
    for t in range(1000):
        print('state', state.shape)
        env.render()
        #print('env.action_space', env.action_space)
        #action = env.action_space.sample()
        action= np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)))
        state_updated, reward, done, info = env.step(action)
        state_updated = state_to_scalar(state_updated)
        print('reward', reward)

        state = state_updated
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
