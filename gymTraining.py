from tqdm import tqdm


# A collection of helper functions to train gymnasium RL models
def trainGymRlModel(agent, env, numEpisodes, progress=True):
    ran = range(numEpisodes)
    if progress:
        ran = tqdm(ran)

    for episode in ran:
        obs, info = env.reset()
        done = False

        # print("-------")

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
