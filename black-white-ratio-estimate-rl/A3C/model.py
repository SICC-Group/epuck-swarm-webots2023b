import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback


class Model(nn.Module):
    def __init__(self, s_dim, a_dim, continuous=False):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.continuous = continuous

        if continuous:
            self.a1 = nn.Linear(s_dim, 64)
            self.mu = nn.Linear(64, a_dim)
            self.sigma = nn.Linear(64, a_dim)
            self.c1 = nn.Linear(s_dim, 100)
            self.v = nn.Linear(100, 1)
            self.distribution = torch.distributions.Normal
        else:
            self.pi1 = nn.Linear(s_dim, 128)
            self.pi2 = nn.Linear(128, a_dim)
            self.v1 = nn.Linear(s_dim, 128)
            self.v2 = nn.Linear(128, 1)
            self.distribution = torch.distributions.Categorical

        self.layer_shape = {k: v.shape for k, v in self.state_dict().items()}

    def forward(self, x):
        if self.continuous:
            a1 = F.relu(self.a1(x))
            mu = 2 * F.relu(self.mu(a1))
            sigma = F.softplus(self.sigma(a1)) + 0.001
            
            c1 = F.relu(self.c1(x))
            values = self.v(c1)

            return mu, sigma, values
        else:
            pi1 = torch.tanh(self.pi1(x))
            logits = self.pi2(pi1)
            v1 = torch.tanh(self.v1(x))
            values = self.v2(v1)
            return logits, values
    
    def choose_action(self, s):
        if self.continuous:
            self.training == False
            mu, sigma, _ = self.forward(s)
            m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
            return m.sample().numpy()
        else:
            self.eval()
            logits, _ = self.forward(s)
            prob = F.softmax(logits, dim=0).data
            m = self.distribution(prob)
            return int(m.sample().numpy())
    
    def loss_func(self, s, a, v_t):
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
            logits, values = self.forward(s)
            td = v_t - values
            c_loss = td.pow(2)
            
            probs = F.softmax(logits, dim=1)
            m = self.distribution(probs)
            try:
                exp_v = m.log_prob(a) * td.detach().squeeze()
            except ValueError:
                import pdb; pdb.set_trace()
                print(traceback.format_exc())
            a_loss = -exp_v
            total_loss = (c_loss + a_loss).mean()
            return total_loss
    
    @staticmethod
    def v_wrap(np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)
    
    def train_and_get_grad(self, bs, ba, br, done, s_, gamma, opt):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.forward(self.v_wrap(np.array(s_)))[-1].data.numpy()[0]
        buffer_v_target = []
        for r in br[::-1]:
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        opt.zero_grad()
            # import pdb; pdb.set_trace()
        loss = self.loss_func(
            self.v_wrap(np.vstack(bs)),
            self.v_wrap(np.array(ba), dtype=np.int64) if type(ba[0]) == int else self.v_wrap(np.vstack(ba)),
            self.v_wrap(np.array(buffer_v_target)[:, None])
        )
        loss.backward()
        grad = self.get_serializable_state_list(to_list=True, option='grad')
        opt.zero_grad()
        return grad
    
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
            param_data = torch.tensor(param_data, dtype=torch.float32)
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
        
    
if __name__ == '__main__':
    model = Model(4, 2)
    print(model)