import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import visdom


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_index) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    # actor_critic.load_state_dict(torch.load('trained_models/ppo/HalfCheetah-v2.pt')[0].state_dict())
    # utils.get_vec_normalize(envs).obs_rms = torch.load('trained_models/ppo/HalfCheetah-v2.pt')[1]

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=args.num_expert_data, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    # # print('+++++++++++++++++++++++++++++++++++++')
    # # env = gym.make("HalfCheetah-v2")
    # # obs = torch.from_numpy(np.array(env.reset())).unsqueeze(0).type(torch.float32)
    # # print(obs.shape)
    # # print('-----------------------------------')
    # # done= False
    # # reward_total = 0
    # # for i in range(2048):
    # #     with torch.no_grad():
    # #         value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
    # #             obs, None, None)
    # #         env.render()
    # #         action=np.array(action.squeeze())
    # #         print(action)
    # #         obs, reward, done, infos = env.step(action)
    # #         reward_total += reward
    # #         obs = torch.from_numpy(np.array(obs)).unsqueeze(0).type(torch.float32)
    # # print(reward_total)
    #
    # # print('+++++++++++++++++++++++++++++++++++++')
    # # # obs = torch.from_numpy(np.array(env.reset())).unsqueeze(0).type(torch.float32)
    # # print(obs.shape)
    # # print('-----------------------------------')
    # # done = False
    # # reward_total = 0
    # # for j in range(5):
    # #     for i in range(2048):
    # #         with torch.no_grad():
    # #             value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
    # #                 obs, None, None)
    # #
    # #         obs, reward, done, infos = envs.step(action)
    # #         # print('-----------------------------------')
    # #         for info in infos:
    # #             if 'episode' in info.keys():
    # #                 episode_rewards.append(info['episode']['r'])
    # #
    # #     print(len(episode_rewards))
    # #     print(np.mean(episode_rewards))
    # obs_rms = utils.get_vec_normalize(envs).obs_rms
    # evaluate(actor_critic, obs_rms, args.env_name, args.seed,
    #          args.num_processes, eval_log_dir, device)

    # return
    vis = visdom.Visdom(env=args.vis_name, port=6029)
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        # if args.gail:
        if j >= 10:
            envs.venv.eval()

        gail_epoch = args.gail_epoch
        if j < 10:
            gail_epoch = args.warm_start_epoch  # Warm up

        origin_reward = torch.zeros(args.num_steps, args.num_processes, 1)
        for step in range(args.num_steps):
            rollouts.rewards[step], origin_reward[step] = discr.predict_reward(
                rollouts.obs[step], rollouts.actions[step], args.gamma,
                rollouts.masks[step], update_rms=args.no_rms, reward=args.gail_reward)

        for _ in range(gail_epoch):
            loss, gail_loss, grad_pen_loss, expert_loss, policy_loss, expert_acc, policy_acc = discr.update(gail_train_loader, rollouts,
                         utils.get_vec_normalize(envs)._obfilt, extra_loss=args.gail_loss)



        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        # for i in range(args.ppo_epoch):
        #     loss, gail_loss, grad_pen_loss, expert_loss, policy_loss, expert_acc, policy_acc, value_loss, action_loss, dist_entropy = update(discr, agent, gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt, extra_loss=args.gail_loss)
        vis.line(X=[total_num_steps], Y=[loss], win='loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'loss'})
        vis.line(X=[total_num_steps], Y=[gail_loss], win='gail_loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'gail_loss'})
        vis.line(X=[total_num_steps], Y=[grad_pen_loss], win='grad_pen_loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'grad_pen_loss'})
        vis.line(X=[total_num_steps], Y=[expert_loss], win='expert_loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'expert_loss'})
        vis.line(X=[total_num_steps], Y=[policy_loss], win='policy_loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'policy_loss'})
        vis.line(X=[total_num_steps], Y=[expert_acc], win='expert_acc', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'expert_acc'})
        vis.line(X=[total_num_steps], Y=[policy_acc], win='policy_acc', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'policy_acc'})

        mean_reward = rollouts.rewards.mean().item()
        mean_origin_reward = origin_reward.mean().item()
        vis.line(X=[total_num_steps], Y=[mean_reward], win='mean_reward', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'mean_reward'})
        vis.line(X=[total_num_steps], Y=[mean_origin_reward], win='mean_origin_reward', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'mean_origin_reward'})

        vis.line(X=[total_num_steps], Y=[value_loss], win='value_loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'value_loss'})
        vis.line(X=[total_num_steps], Y=[action_loss], win='action_loss', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'action_loss'})
        vis.line(X=[total_num_steps], Y=[dist_entropy], win='dist_entropy', update='append',
                 opts={"xlabel": 'steps', 'ylabel': 'value', 'title': 'dist_entropy'})

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            vis.line(X=[total_num_steps], Y=[np.mean(episode_rewards).item()], win='episode reward',  update='append', opts={'xlabel': 'steps', 'ylabel': 'episode reward', 'title': 'episode reward'})
            # print(
            #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            #     .format(j, total_num_steps,
            #             int(total_num_steps / (end - start)),
            #             len(episode_rewards), np.mean(episode_rewards),
            #             np.median(episode_rewards), np.min(episode_rewards),
            #             np.max(episode_rewards), dist_entropy, value_loss,
            #             action_loss))



        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

def update(discr, agent, expert_loader, rollouts, obsfilt, extra_loss):
    discr.train()

    loss = 0
    gail_loss_value = 0
    grad_pen_value = 0
    expert_loss_value = 0
    policy_loss_value = 0
    expert_acc_value = 0
    policy_acc_value = 0

    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0
    policy_data_generator = rollouts.feed_forward_generator(
        advantages, mini_batch_size=expert_loader.batch_size)

    n = 0
    for expert_batch, policy_batch in zip(expert_loader,
                                          policy_data_generator):
        obs_batch, recurrent_hidden_states_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
        adv_targ = policy_batch

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, _ = agent.actor_critic.evaluate_actions(
            obs_batch, recurrent_hidden_states_batch, masks_batch,
            actions_batch)

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - agent.clip_param,
                            1.0 + agent.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if agent.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                                 (values - value_preds_batch).clamp(-agent.clip_param, agent.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()


        policy_state, policy_action = obs_batch, actions_batch
        policy_d = discr.trunk(
            torch.cat([policy_state, policy_action], dim=1))

        expert_state, expert_action = expert_batch
        expert_state = obsfilt(expert_state.numpy(), update=False)
        expert_state = torch.FloatTensor(expert_state).to(discr.device)
        expert_action = expert_action.to(discr.device)
        expert_d = discr.trunk(
            torch.cat([expert_state, expert_action], dim=1))

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            torch.ones(expert_d.size()).to(discr.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            torch.zeros(policy_d.size()).to(discr.device))

        gail_loss = expert_loss + policy_loss
        grad_pen = discr.compute_grad_pen(expert_state, expert_action,
                                         policy_state, policy_action)
        zero = torch.zeros_like(policy_d)
        one = torch.ones_like(policy_d)
        expert_pred_prob = torch.sigmoid(expert_d)
        policy_pred_prob = torch.sigmoid(policy_d)
        expert_result = torch.where(expert_pred_prob > 0.5, one, zero)
        policy_result = torch.where(policy_pred_prob < 0.5, zero, one)
        expert_correct = (expert_result == one).sum().float()
        policy_correct = (policy_result == zero).sum().float()
        expert_acc = expert_correct / len(one)
        policy_acc = policy_correct / len(zero)
        expert_acc_value += expert_acc.item()
        policy_acc_value += policy_acc.item()

        if extra_loss == 'extra_loss':
            loss += (gail_loss + grad_pen).item()
            gail_loss_value += gail_loss.item()
            grad_pen_value += grad_pen.item()
            expert_loss_value += expert_loss.item()
            policy_loss_value += policy_loss.item()

        else:
            loss = gail_loss.item()
            gail_loss_value = 0
            grad_pen_value = 0
        n += 1

        discr.optimizer.zero_grad()
        agent.optimizer.zero_grad()
        if extra_loss == 'extra_loss':
            (gail_loss + grad_pen + value_loss * agent.value_loss_coef + action_loss -
         dist_entropy * agent.entropy_coef).backward()
        else:
            (gail_loss + value_loss * agent.value_loss_coef + action_loss -
         dist_entropy * agent.entropy_coef).backward()

        nn.utils.clip_grad_norm_(agent.actor_critic.parameters(),
                                 agent.max_grad_norm)

        discr.optimizer.step()
        agent.optimizer.step()

        value_loss_epoch += value_loss.item()
        action_loss_epoch += action_loss.item()
        dist_entropy_epoch += dist_entropy.item()


    value_loss_epoch /= n
    action_loss_epoch /= n
    dist_entropy_epoch /= n

    return loss / n, gail_loss_value / n, grad_pen_value / n, expert_loss_value / n, policy_loss_value / n, expert_acc_value / n, policy_acc_value / n, \
           value_loss_epoch, action_loss_epoch, dist_entropy_epoch

if __name__ == "__main__":
    main()
