import math
from typing import Union
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(
        self, s_dim, a_dim, col, row, device, 
        reward_normalize, continuous=False
    ):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.col = col
        self.row = row
        self.reward_normalize = reward_normalize
        self.device = device
        self.continuous = continuous

        if continuous:
            self.a1 = nn.Linear(s_dim, 64)
            self.mu = nn.Linear(64, a_dim)
            self.sigma = nn.Linear(64, a_dim)
            self.c1 = nn.Linear(s_dim, 100)
            self.v = nn.Linear(100, 1)
            self.distribution = torch.distributions.Normal
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            conv_out_size = 4 * (self.row // 4) * (self.col // 4)
            self.fc_map = nn.Sequential(
                nn.Linear(conv_out_size, 20),
                nn.ReLU()
            )
            self.fc = nn.Sequential(
                nn.Linear(s_dim + 20, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
            self.policy = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, a_dim)
            )
            self.value = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            self.distribution = torch.distributions.Categorical

        self.layer_shape = {k: v.shape for k, v in self.state_dict().items()}

    def forward(self, x, map_info):
        """
        logits: (batch_size, a_dim);
        values: (batch_size, 1);
        """
        if self.continuous:
            a1 = F.relu(self.a1(x))
            mu = 2 * F.relu(self.mu(a1))
            sigma = F.softplus(self.sigma(a1)) + 0.001
            
            c1 = F.relu(self.c1(x))
            values = self.v(c1)

            return mu, sigma, values
        else:
            if len(map_info.shape) == 3:
                map_info = map_info.unsqueeze(1)
            elif len(map_info.shape) == 2:
                map_info = map_info.unsqueeze(0).unsqueeze(1)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            map_feature = self.conv_layers(map_info)
            map_feature = map_feature.view(map_feature.size(0), -1)
            map_feature = self.fc_map(map_feature)
            x = torch.cat((x, map_feature), dim=1)
            x = self.fc(x)
            logits = self.policy(x)
            values = self.value(x)
            return logits, values
    
    def choose_action(self, s, map_info, mask):
        if self.continuous:
            self.training == False
            mu, sigma, _ = self.forward(s)
            m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
            return m.sample().numpy()
        else:
            self.eval()
            logits, _ = self.forward(s, map_info)
            logits[:,mask == 0] = -float('inf')
            prob = F.softmax(logits, dim=1).data
            m = self.distribution(prob)
            return int(m.sample().cpu().numpy())
    
    def loss_func(self, s, map_info, a, v_t, mask):
        self.train()
        if self.continuous:
            mu, sigma, values = self.forward(s)
            td = v_t - values
            c_loss = td.pow(2)

            m = self.distribution(mu, sigma)
            log_prob = m.log_prob(a)
            entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
            exp_v = log_prob * td.detach() + 0.005 * entropy
            a_loss = -exp_v
            total_loss = (a_loss + c_loss).mean()
            return total_loss
        else:
            logits, values = self.forward(s, map_info)
            td = v_t - values
            td_mean = td.mean(dim=0)
            td_std = td.std(dim=0)
            td = (td - td_mean) / (td_std + 1e-10)

            c_loss = td.pow(2)
            
            logits[mask == 0] = -float('inf')
            # probs = F.softmax(logits, dim=1)
            # m = self.distribution(probs)
            # exp_v = m.log_prob(a) * td.detach().squeeze()
            # a_loss = -exp_v.unsqueeze(-1)
            log_probs = F.log_softmax(logits, dim=1)
            log_probs_a = log_probs.gather(1, a.long().unsqueeze(-1))
            a_loss = -log_probs_a * td.detach()
            total_loss = (c_loss + a_loss).sum()
            return total_loss, c_loss.sum(), a_loss.sum()
    
    def train_and_get_grad(self, bs, bmap, ba, br, bmask, done, s_, map_, gamma, opt, steps):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.forward(
                torch.tensor(s_, dtype=torch.float, device=self.device),
                torch.tensor(map_, dtype=torch.float, device=self.device)
            )[-1].data.item()
        # normalize br
        if self.reward_normalize:
            br = np.array(br)
            br = (br - br.mean()) / (br.std() + 1e-10)
        buffer_v_target = []
        for r in br[::-1]:
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        opt.zero_grad()
        loss, c_loss, a_loss = self.loss_func(
            torch.tensor(bs, dtype=torch.float, device=self.device),
            torch.tensor(bmap, dtype=torch.float, device=self.device),
            torch.tensor(ba, dtype=torch.float, device=self.device),
            torch.tensor(buffer_v_target, dtype=torch.float, device=self.device).unsqueeze(-1),
            torch.vstack(bmask).to(self.device),
        )
        loss.backward()
        # if self.norm_decay_steps == 0:
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        # else:
        #     norm_ = self.grad_norm if steps <= self.norm_decay_steps else self.grad_norm / (steps - self.norm_decay_steps)
        #     torch.nn.utils.clip_grad_norm_(self.parameters(), max(norm_, self.grad_norm_min))
        grad = self.get_serializable_state_list(to_numpy=True, option='grad')
        opt.zero_grad()
        return grad, loss.item(), c_loss.item(), a_loss.item()
    
    def load_serializable_state_list(
        self,
        serializable_state_list: Union[list, np.ndarray]
    ):
        if self.layer_shape == None:
            raise ValueError("layer_shape is not initialized")
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            param_data = serializable_state_list[pointer:pointer + num_param]
            param_data = torch.tensor(param_data, dtype=torch.float32, device=self.device)
            param.data = param_data.reshape(param.shape)
            pointer += num_param


    def get_serializable_state_list(self, to_numpy=True, option='param'):
        assert option in ['param', 'grad']
        if option == 'param':
            params_list = [param.view(-1) for param in self.parameters()]
        elif option == 'grad':
            params_list = [param.grad.view(-1) for param in self.parameters()]
        if to_numpy:
            return torch.cat(params_list).detach().cpu().numpy()
        else:
            return torch.cat(params_list)

