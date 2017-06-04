# import random
# from collections import deque
# import gym
# import numpy as np
# import tensorflow as tf
# from gym import wrappers
# import ppaquette_gym_doom
# from ppaquette_gym_doom.wrappers import SetResolution, ToDiscrete
# from gym.wrappers import SkipWrapper
from doom_subfun import *
import dqn_doom as dqn
import matplotlib.pyplot as plt

env = create_env() # call doom environment
env = wrappers.Monitor(env, 'gym-test-1', force=True)

# Constants defining our neural network
input_size = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
output_size = env.action_space.n

REPLAY_MEMORY = 50000

if __name__ == "__main__":
    max_episodes = 2500
    # store the previous observations in replay memory
    replay_buffer = deque()

    last_100_game_reward = deque()

    # some variables
    e_param = 0
    reward_sum_hist = []

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            reward_sum = 0
            e_param += 1
            e = 1. / ((e_param / 50) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                reward_sum = reward_sum + reward

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

            print("Episode: {}  steps: {}  reward_sum: {}  e: {}".format(episode, step_count, reward_sum, e))

            if episode % 10 == 1:  # train every 10 episode
                # Get a random batch of experiences.
                for sample in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)
                    print('train : episode={}  sample={}  exp rate={}'.format(episode, sample, e))

                print("Loss: ", loss)
                # copy q_net -> target_net
                sess.run(copy_ops)

            last_100_game_reward.append(reward_sum)
            avg_reward = np.mean(last_100_game_reward)
            reward_sum_hist.append(avg_reward)
            plt.plot(reward_sum_hist)
            plt.draw()
            plt.pause(0.01)

            if avg_reward > 2000:
                print("Game Cleared in {episode} episodes with avg reward {avg_reward}")
                break

            if len(last_100_game_reward) > 100:
                a = last_100_game_reward.popleft()


        env.close()
        # gym.upload('/home/lwc/PycharmProjects/git/wc_DL/Doom_1dimension_action/gym-test-1',
        #            api_key='sk_lhnVA8tnREi9rHs1IbOOA')
    plt.show()

