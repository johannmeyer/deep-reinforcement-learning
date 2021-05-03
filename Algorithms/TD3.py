#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gym


from Logger import Logger

from ReplayMemory import ReplayMemory

from Agent import Actor, Critic

import imageio

d = 2
sigma_tps_reg = 0.05
tps_reg_clip = 0.1

def TD3(args):
    
    # Instantiate environment
    env = gym.make(args.env_name)
    
    num_actions = env.action_space.shape[0]
    num_states = env.observation_space.shape[0]
    action_bound = (env.action_space.low, env.action_space.high)
    
    
    # For exploration
    var_min = 0.3
    var = 2
    
    # Instantiate algorithm specific parts 
    replay_memory = ReplayMemory(args.replay_mem_size)

    actor = Actor(num_states, num_actions, learning_rate=args.lr, env_action_limit=action_bound, hidden_layers=args.hidden_layers)
    critic = Critic(num_states, num_actions, learning_rate=args.lr_c, hidden_layers=args.hidden_layers)
    critic2 = Critic(num_states, num_actions, learning_rate=args.lr_c, hidden_layers=args.hidden_layers)
    
    # For saving the models
    args.models = [actor.model, critic.model]
    

    episode_num = 0
    while episode_num <= args.num_episodes: # for loop does not handle changing of num_episodes
        episode_num += 1
        
        frame_list = [] # for saving the gif
        
        # Reset environment for next episode
        done = False
        state = env.reset()

        ep_reward = 0 # Total reward for an episode
        
        var = max([var*.997, var_min]) 
        
        t = 0 # stores number of steps       
        while not done:
            t += 1

            if args.save_render:
                frame = env.render(mode='rgb_array')
                frame_list.append(frame[::2,::2,:]) #subsample the image
                
            if args.render: env.render()

            action = np.squeeze(actor.predict(np.expand_dims(state, axis=1)))

            if not args.det:
                action = np.clip(np.random.normal(action, var), *action_bound)
                
            next_state, reward, done, _ = env.step(action)

            if reward < -1:
                clipped_reward = -1
            else:
                clipped_reward = reward
     
            if not args.det:
                replay_memory.append((state, action, clipped_reward, next_state, done))
                
                if len(replay_memory.replay_memory) >= 1e4:
                    states, actions, rewards, next_states, dones = replay_memory.get_batch(args.batch_size)
            
                    # Update critic
                    next_state_predicted_actions = actor.predict(next_states, target=True)
                    action_noise = np.clip(np.random.normal(0, sigma_tps_reg, size=next_state_predicted_actions.shape), -tps_reg_clip, tps_reg_clip) # clipped Gaussian distribution
                    next_state_predicted_actions = np.clip(next_state_predicted_actions + action_noise, np.expand_dims(action_bound[0], axis=1), np.expand_dims(action_bound[1],axis=1))
                    
                    target_q1 = critic.predict(next_states, next_state_predicted_actions, target=True).reshape(-1)
                    target_q2 = critic2.predict(next_states, next_state_predicted_actions, target=True).reshape(-1)
                    target_q = np.minimum(target_q1, target_q2)
                    
                    targets = rewards + args.gamma*target_q*(1-dones)
                    targets = np.expand_dims(targets, axis=0)
                    
                    critic.train(states, actions, targets)
                    critic2.train(states, actions, targets)
                    
                    if t % d == 0:
                        # Update actor in direction of more Q
                        actions = actor.predict(states)
                        action_gradient = critic.get_action_gradient(states, actions)
                        actor.train(states, action_gradient)
            
                        # Update target models
                        actor.blend_weights(tau=args.tau)
                        critic.blend_weights(tau=args.tau)
                        critic2.blend_weights(tau=args.tau)
                        
            ep_reward += reward
            state = next_state
    
        # Moving average of the reward
        args.mean_reward_deque.append(ep_reward)
        mean_reward = np.mean(args.mean_reward_deque)
        
        print("({}) mean reward {:5.2f} reward: {:.2f}".format(episode_num, mean_reward, ep_reward))
        
        # Save the gif file
        if args.save_render:
            print('Writing gif to file')
            imageio.mimwrite("{}-{}-{:.2f}.gif".format(args.algorithm, episode_num, ep_reward), frame_list)
            print('Completed')
        
        # Save the log data to file
        if args.log_dir is not None:
            args.reward_logger.append([ep_reward])
            if episode_num % 50 == 0:
                args.reward_logger.save()
    env.close()
