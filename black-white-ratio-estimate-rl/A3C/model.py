import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback


class Model(nn.Module):
    def __init__(
        self, s_dim, a_dim, col, row, device,
        grad_norm_init, norm_decay_steps, grad_norm_min, 
        reward_normalize, continuous=False
    ):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.col = col
        self.row = row
        self.grad_norm = grad_norm_init
        self.norm_decay_steps = norm_decay_steps
        self.grad_norm_min = grad_norm_min
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
                nn.Linear(128, 64),
                nn.ReLU(),
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
    
    def choose_action(self, s, map_info):
        if self.continuous:
            self.training == False
            mu, sigma, _ = self.forward(s)
            m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
            return m.sample().numpy()
        else:
            self.eval()
            logits, _ = self.forward(s, map_info)
            prob = F.softmax(logits, dim=1).data
            m = self.distribution(prob)
            return int(m.sample().cpu().numpy())
    
    def loss_func(self, s, map_info, a, v_t):
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
            c_loss = td.pow(2)
            
            probs = F.softmax(logits, dim=1)
            m = self.distribution(probs)
            exp_v = m.log_prob(a) * td.detach().squeeze()
            a_loss = -exp_v.unsqueeze(-1)
            total_loss = (c_loss + a_loss).mean()
            return total_loss
    
    def train_and_get_grad(self, bs, bmap, ba, br, done, s_, map_, gamma, opt, steps):
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
        loss = self.loss_func(
            torch.tensor(bs, dtype=torch.float, device=self.device),
            torch.tensor(bmap, dtype=torch.float, device=self.device),
            torch.tensor(ba, dtype=torch.float, device=self.device),
            torch.tensor(buffer_v_target, dtype=torch.float, device=self.device).unsqueeze(-1)
        )
        loss.backward()
        norm_ = self.grad_norm if steps <= self.norm_decay_steps else self.grad_norm / (steps - self.norm_decay_steps)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max(norm_, self.grad_norm_min))
        # torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        grad = self.get_serializable_state_list(to_list=True, option='grad')
        # print(f"max:{max(grad):.6f}, min:{min(grad):.6f}")
        opt.zero_grad()
        return grad, loss.item()
    
    def load_serializable_state_list(
        self,
        serializable_state_list: list
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


    def get_serializable_state_list(self, to_list=True, option='param'):
        assert option in ['param', 'grad']
        if option == 'param':
            params_list = [param.view(-1) for param in self.parameters()]
        elif option == 'grad':
            params_list = [param.grad.view(-1) for param in self.parameters()]
        if to_list:
            return torch.cat(params_list).tolist()
        else:
            return torch.cat(params_list)
