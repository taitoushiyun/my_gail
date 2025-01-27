import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from stable_baselines3.common.running_mean_std import RunningMeanStd

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None, extra_loss='extra_loss'):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        gail_loss_value = 0
        grad_pen_value = 0
        expert_loss_value = 0
        policy_loss_value = 0
        expert_acc_value = 0
        policy_acc_value = 0

        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
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

            self.optimizer.zero_grad()
            if extra_loss == 'extra_loss':
                (gail_loss + grad_pen).backward()
            else:
                gail_loss.backward()
            self.optimizer.step()
        return loss / n, gail_loss_value / n, grad_pen_value / n, expert_loss_value / n, policy_loss_value / n, expert_acc_value / n, policy_acc_value / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True, reward='both'):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            if reward == 'both':
                reward = s.log() - (1 - s).log()
            elif reward == 'first':
                reward = s.log()
            elif reward == 'second':
                reward = - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), reward
            else:
                return reward, reward


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i]
