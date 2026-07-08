# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

try:
    from diffrl.models import model_utils
except ImportError:  # pragma: no cover - legacy direct-script execution
    from models import model_utils


class ActorDeterministicMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorDeterministicMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
                        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.actor = nn.Sequential(*modules).to(device)
        bias_init = cfg_network.get("actor_output_bias_init")
        weight_scale = cfg_network.get("actor_output_weight_init_scale")
        if bias_init is not None or weight_scale is not None:
            output_layer = None
            for module in reversed(self.actor):
                if isinstance(module, nn.Linear):
                    output_layer = module
                    break
            if output_layer is None:
                raise RuntimeError("ActorDeterministicMLP has no Linear output layer to initialize.")
            if weight_scale is not None:
                with torch.no_grad():
                    output_layer.weight.mul_(float(weight_scale))
            if bias_init is not None:
                if isinstance(bias_init, (float, int)):
                    bias_tensor = torch.full((self.action_dim,), float(bias_init), dtype=torch.float32, device=self.device)
                else:
                    bias_tensor = torch.as_tensor(bias_init, dtype=torch.float32, device=self.device)
                    if tuple(bias_tensor.shape) != (self.action_dim,):
                        raise ValueError(
                            f"actor_output_bias_init shape mismatch: expected {(self.action_dim,)}, "
                            f"got {tuple(bias_tensor.shape)}."
                        )
                with torch.no_grad():
                    output_layer.bias.copy_(bias_tensor)

        print(self.actor)

    def get_logstd(self):
        # return self.logstd
        return None

    def forward(self, observations, deterministic = False):
        return self.actor(observations)


class ActorStochasticMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_network, device='cuda:0'):
        super(ActorStochasticMLP, self).__init__()

        self.device = device

        self.layer_dims = [obs_dim] + cfg_network['actor_mlp']['units'] + [action_dim]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
        
        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))
            else:
                modules.append(model_utils.get_activation_func('identity'))
            
        self.mu_net = nn.Sequential(*modules).to(device)

        logstd = cfg_network.get('actor_logstd_init', -1.0)

        self.logstd = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        print(self.mu_net)
        print(self.logstd)
    
    def get_logstd(self):
        return self.logstd

    def forward(self, obs, deterministic = False):
        mu = self.mu_net(obs)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp() # (num_actions)
            # eps = torch.randn((*obs.shape[:-1], std.shape[-1])).to(self.device)
            # sample = mu + eps * std
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample
    
    def forward_with_dist(self, obs, deterministic = False):
        mu = self.mu_net(obs)
        std = self.logstd.exp() # (num_actions)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std
        
    def evaluate_actions_log_probs(self, obs, actions):
        mu = self.mu_net(obs)

        std = self.logstd.exp()
        dist = Normal(mu, std)

        return dist.log_prob(actions)
