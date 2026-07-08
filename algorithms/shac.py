# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import importlib
import time
import math
import numpy as np
import copy
import torch
from contextlib import nullcontext
from torch.nn.utils.clip_grad import clip_grad_norm_
import yaml


def _debug_tensor_finiteness(name, tensor, step=None):
    if not torch.is_tensor(tensor):
        return
    finite = torch.isfinite(tensor)
    prefix = f"[DEBUG] {name}"
    if step is not None:
        prefix += f"[{step}]"
    if bool(finite.all()):
        detached = tensor.detach()
        print(
            f"{prefix}: finite=True "
            f"range=({float(detached.min().cpu()):.6g}, {float(detached.max().cpu()):.6g}) "
            f"norm={float(torch.linalg.vector_norm(detached.reshape(-1)).cpu()):.6g}"
        )
        return
    finite_values = tensor.detach()[finite]
    finite_range = None
    if finite_values.numel() > 0:
        finite_range = (float(finite_values.min().cpu()), float(finite_values.max().cpu()))
    print(
        f"{prefix}: finite=False "
        f"finite={int(finite.sum().cpu())}/{tensor.numel()} finite_range={finite_range}"
    )


def _grad_profile(named_parameters, *, top_k=5):
    total_count = 0
    finite_count = 0
    nonfinite_count = 0
    finite_sq_sum = 0.0
    abs_max = 0.0
    per_param = []

    for name, param in named_parameters:
        grad = param.grad
        if grad is None:
            per_param.append(
                {
                    "name": name,
                    "numel": 0,
                    "finite_count": 0,
                    "nonfinite_count": 0,
                    "abs_max": 0.0,
                    "norm": 0.0,
                    "has_grad": False,
                }
            )
            continue
        detached = grad.detach()
        flat = detached.reshape(-1)
        total_count += flat.numel()
        finite = torch.isfinite(flat)
        current_finite = int(finite.sum().cpu())
        current_nonfinite = flat.numel() - current_finite
        finite_count += current_finite
        nonfinite_count += current_nonfinite
        current_abs_max = 0.0
        current_norm = float("inf") if current_nonfinite > 0 else 0.0
        if current_finite > 0:
            finite_values = flat[finite].to(dtype=torch.float64)
            current_abs_max = float(finite_values.abs().max().cpu())
            current_sq_sum = float(torch.sum(finite_values * finite_values).cpu())
            if current_nonfinite == 0:
                current_norm = math.sqrt(current_sq_sum)
            finite_sq_sum += current_sq_sum
            abs_max = max(abs_max, current_abs_max)
        per_param.append(
            {
                "name": name,
                "numel": flat.numel(),
                "finite_count": current_finite,
                "nonfinite_count": current_nonfinite,
                "abs_max": current_abs_max,
                "norm": current_norm,
                "has_grad": True,
            }
        )

    norm = float("inf") if nonfinite_count > 0 else math.sqrt(finite_sq_sum)
    per_param.sort(key=lambda item: (not math.isfinite(item["norm"]), item["norm"], item["abs_max"]), reverse=True)
    return {
        "total_count": total_count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "abs_max": abs_max,
        "norm": norm,
        "top": per_param[:top_k],
    }


def _debug_grad_profile(prefix, profile):
    print(
        f"[DEBUG] {prefix}: norm={profile['norm']:.6g} abs_max={profile['abs_max']:.6g} "
        f"finite={profile['finite_count']}/{profile['total_count']} "
        f"nonfinite={profile['nonfinite_count']}"
    )
    for item in profile["top"]:
        print(
            f"[DEBUG] {prefix} top {item['name']}: norm={item['norm']:.6g} "
            f"abs_max={item['abs_max']:.6g} finite={item['finite_count']}/{item['numel']} "
            f"nonfinite={item['nonfinite_count']}"
        )


def _raise_on_nonfinite_grad_profile(prefix, profile):
    if profile["nonfinite_count"] <= 0 and math.isfinite(profile["norm"]):
        return
    _debug_grad_profile(prefix, profile)
    raise RuntimeError(
        f"{prefix} contains non-finite gradients: "
        f"finite={profile['finite_count']}/{profile['total_count']} "
        f"nonfinite={profile['nonfinite_count']} norm={profile['norm']}"
    )


try:
    from tensorboardX import SummaryWriter
except ImportError:  # pragma: no cover - depends on optional runtime deps
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def flush(self):
                pass

            def close(self):
                pass

try:
    from diffrl.interfaces import DifferentiableVecEnv
    from diffrl.models import actor as actor_module
    from diffrl.models import critic as critic_module
    from diffrl.utils.common import *
    from diffrl.utils import torch_utils as tu
    from diffrl.utils.average_meter import AverageMeter
    from diffrl.utils.dataset import CriticDataset
    from diffrl.utils.running_mean_std import RunningMeanStd
    from diffrl.utils.time_report import TimeReport
except ImportError:  # pragma: no cover - legacy direct-script execution
    import models.actor as actor_module
    import models.critic as critic_module
    from utils.common import *
    import utils.torch_utils as tu
    from utils.average_meter import AverageMeter
    from utils.dataset import CriticDataset
    from utils.running_mean_std import RunningMeanStd
    from utils.time_report import TimeReport

    DifferentiableVecEnv = object
legacy_envs = None

class SHAC:
    def __init__(self, cfg, env: DifferentiableVecEnv | None = None, env_fn=None):
        seeding(cfg["params"]["general"]["seed"])
        if env is None:
            if env_fn is None:
                env_name = cfg["params"]["diff_env"]["name"]
                try:
                    global legacy_envs
                    if legacy_envs is None:
                        legacy_envs = importlib.import_module("diffrl.envs")
                    env_fn = getattr(legacy_envs, env_name)
                except Exception as exc:
                    raise ImportError(
                        "DiffRL legacy environments require dflex and related optional dependencies. "
                        "Pass a DifferentiableVecEnv instance with SHAC(cfg, env=...) to use a custom backend."
                    ) from exc
            env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                                device = cfg["params"]["general"]["device"], \
                                render = cfg["params"]["general"].get("render", False), \
                                seed = cfg["params"]["general"]["seed"], \
                                episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                                stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                                MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                                no_grad = False)
        self.env = env

        print('num_envs = ', self.env.num_envs)
        print('num_actions = ', self.env.num_actions)
        print('num_obs = ', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg["params"]["general"]["device"]

        self.gamma = cfg['params']['config'].get('gamma', 0.99)
        
        self.critic_method = cfg['params']['config'].get('critic_method', 'one-step') # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = cfg['params']['config'].get('lambda', 0.95)

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
        
        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)
        self.actor_loss_critic_bootstrap = cfg['params']['config'].get('actor_loss_critic_bootstrap', True)
        self.obs_clip = cfg['params']['config'].get('obs_clip', None)

        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
            
        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', "Ant")
        self.run_final_eval = bool(cfg['params']['config'].get('run_final_eval', True))

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]
        self.grad_value_clip = cfg["params"]["config"].get("grad_value_clip", None)
        self.actor_raw_grad_profile = _grad_profile([])
        self.actor_raw_grad_norm_before_sanitize = torch.tensor(0.0, device=self.device)
        
        if cfg['params']['general']['train']:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (cfg['params']['config']['player'].get('determenistic', False) or cfg['params']['config']['player'].get('deterministic', False))
            self.steps_num = self.env.episode_length

        # create actor critic network
        self.actor_name = cfg["params"]["network"].get("actor", 'ActorStochasticMLP') # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.critic_name = cfg["params"]["network"].get("critic", 'CriticMLP')
        actor_fn = getattr(actor_module, self.actor_name)
        self.actor = actor_fn(self.num_obs, self.num_actions, cfg['params']['network'], device = self.device)
        critic_fn = getattr(critic_module, self.critic_name)
        self.critic = critic_fn(self.num_obs, cfg['params']['network'], device = self.device)
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)
    
        if cfg['params']['general']['train']:
            self.save('init_policy')
    
        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas = cfg['params']['config']['betas'], lr = self.critic_lr)

        # replay buffer
        self.obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_obs), dtype = torch.float32, device = self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.ret = torch.zeros((self.num_envs), dtype = torch.float32, device = self.device)

        # for kl divergence computing
        self.old_mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.old_sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = torch.int64, device = self.device)
        self.termination_reason_counts = {}
        self.termination_done_count = 0
        self.termination_unmatched_done_count = 0
        self.rollout_invalid_counts = {}
        self.actor_action_profile = {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        
        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()

    def _preprocess_obs(self, obs):
        if torch.is_tensor(obs):
            finite = torch.isfinite(obs)
            if not bool(finite.all()):
                first_bad = torch.nonzero(~finite.reshape(-1), as_tuple=False)
                first_bad_index = int(first_bad[0].detach().cpu()) if first_bad.numel() > 0 else -1
                raise RuntimeError(
                    "SHAC observation contains non-finite values: "
                    f"finite={int(finite.sum().detach().cpu())}/{obs.numel()} "
                    f"first_bad_flat_index={first_bad_index}"
                )
        if self.obs_clip is not None:
            obs = torch.clamp(obs, -float(self.obs_clip), float(self.obs_clip))
        return obs
        
    def compute_actor_loss(self, deterministic = False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        
        actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)
                
            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        obs = self.env.initialize_trajectory()
        obs = self._preprocess_obs(obs)
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
            # normalize the current obs
            obs = self._preprocess_obs(obs_rms.normalize(obs))
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            actions = self.actor(obs, deterministic = deterministic)
            if os.environ.get("ACELAB_SHAC_DEBUG_FORWARD_FINITE") == "1":
                _debug_tensor_finiteness("actor_obs", obs, step=i)
                _debug_tensor_finiteness("actor_actions_pre_tanh", actions, step=i)

            actor_actions = torch.tanh(actions)
            with torch.no_grad():
                detached_actions = actor_actions.detach()
                current_count = detached_actions.numel()
                current_sum = float(detached_actions.sum().cpu())
                previous_count = int(self.actor_action_profile["count"])
                total_count = previous_count + current_count
                self.actor_action_profile = {
                    "count": total_count,
                    "min": (
                        float(detached_actions.min().cpu())
                        if previous_count == 0
                        else min(float(self.actor_action_profile["min"]), float(detached_actions.min().cpu()))
                    ),
                    "max": (
                        float(detached_actions.max().cpu())
                        if previous_count == 0
                        else max(float(self.actor_action_profile["max"]), float(detached_actions.max().cpu()))
                    ),
                    "mean": (
                        (float(self.actor_action_profile["mean"]) * previous_count + current_sum) / total_count
                        if total_count > 0
                        else 0.0
                    ),
                }
            obs, rew, done, extra_info = self.env.step(actor_actions)
            if isinstance(extra_info, dict):
                invalid_envs = extra_info.get("rollout_invalid_envs")
                if torch.is_tensor(invalid_envs):
                    count = int(invalid_envs.to(device=self.device, dtype=torch.bool).sum().detach().cpu())
                    if count > 0:
                        self.rollout_invalid_counts["invalid"] = self.rollout_invalid_counts.get("invalid", 0) + count
                new_invalid_envs = extra_info.get("rollout_new_invalid_envs")
                if torch.is_tensor(new_invalid_envs):
                    count = int(new_invalid_envs.to(device=self.device, dtype=torch.bool).sum().detach().cpu())
                    if count > 0:
                        self.rollout_invalid_counts["new_invalid"] = (
                            self.rollout_invalid_counts.get("new_invalid", 0) + count
                        )
                invalid_sources = extra_info.get("rollout_invalid_sources")
                if isinstance(invalid_sources, dict):
                    for source_name, source_mask in invalid_sources.items():
                        if not torch.is_tensor(source_mask):
                            continue
                        count = int(source_mask.to(device=self.device, dtype=torch.bool).sum().detach().cpu())
                        if count > 0:
                            self.rollout_invalid_counts[source_name] = (
                                self.rollout_invalid_counts.get(source_name, 0) + count
                            )
            if os.environ.get("ACELAB_SHAC_DEBUG_FORWARD_FINITE") == "1":
                _debug_tensor_finiteness("env_obs", obs, step=i)
                _debug_tensor_finiteness("env_reward", rew, step=i)
            obs = self._preprocess_obs(obs)
            
            with torch.no_grad():
                raw_rew = rew.clone()
            
            # scale the reward
            rew = rew * self.rew_scale
            
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = self._preprocess_obs(obs_rms.normalize(obs))

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                    
                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1
        
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)
            termination_terms = extra_info.get("termination_terms") if isinstance(extra_info, dict) else None
            matched_done_count = 0
            if isinstance(termination_terms, dict) and len(done_env_ids) > 0:
                term_names = list(termination_terms.get("names") or [])
                term_dones = termination_terms.get("dones")
                if torch.is_tensor(term_dones) and term_dones.ndim == 2 and term_dones.shape[1] >= len(term_names):
                    done_term_dones = term_dones.to(device=self.device, dtype=torch.bool)[done_env_ids]
                    matched_done_count = int(done_term_dones.any(dim=1).sum().detach().cpu())
                    for term_index, term_name in enumerate(term_names):
                        count = int(done_term_dones[:, term_index].sum().detach().cpu())
                        if count > 0:
                            self.termination_reason_counts[term_name] = (
                                self.termination_reason_counts.get(term_name, 0) + count
                            )
            if len(done_env_ids) > 0:
                done_count = int(done_env_ids.numel())
                self.termination_done_count += done_count
                self.termination_unmatched_done_count += max(done_count - matched_done_count, 0)

            next_values[i + 1] = self.target_critic(obs).squeeze(-1)

            obs_before_reset = extra_info.get('obs_before_reset', obs)
            for id in done_env_ids:
                if torch.isnan(obs_before_reset[id]).sum() > 0 \
                    or torch.isinf(obs_before_reset[id]).sum() > 0 \
                    or (torch.abs(obs_before_reset[id]) > 1e6).sum() > 0: # ugly fix for nan values
                    next_values[i + 1, id] = 0.
                elif self.episode_length[id] < self.max_episode_length: # early termination
                    next_values[i + 1, id] = 0.
                else: # otherwise, use terminal value critic to estimate the long-term performance
                    if self.obs_rms is not None:
                        real_obs = self._preprocess_obs(obs_rms.normalize(self._preprocess_obs(obs_before_reset[id])))
                    else:
                        real_obs = self._preprocess_obs(obs_before_reset[id])
                    next_values[i + 1, id] = self.target_critic(real_obs).squeeze(-1)
            
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError
            
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                done_loss = -rew_acc[i + 1, done_env_ids]
                if self.actor_loss_critic_bootstrap:
                    done_loss = done_loss - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]
                actor_loss = actor_loss + done_loss.sum()
            else:
                # terminate all envs at the end of optimization iteration
                terminal_loss = -rew_acc[i + 1, :]
                if self.actor_loss_critic_bootstrap:
                    terminal_loss = terminal_loss - self.gamma * gamma * next_values[i + 1, :]
                actor_loss = actor_loss + terminal_loss.sum()
        
            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                            print('ep loss error')
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.
                        self.episode_discounted_loss[done_env_id] = 0.
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.

        actor_loss /= self.steps_num * self.num_envs

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
        if os.environ.get("ACELAB_SHAC_DEBUG_FORWARD_FINITE") == "1":
            _debug_tensor_finiteness("actor_loss", actor_loss)
            
        self.actor_loss = actor_loss.detach().cpu().item()
            
        self.step_count += self.steps_num * self.num_envs

        return actor_loss
    
    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic = False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(self.num_envs, dtype = torch.int64, device = self.device)
        episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)

        obs = self.env.reset()
        obs = self._preprocess_obs(obs)

        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                obs = self._preprocess_obs(self.obs_rms.normalize(obs))
            else:
                obs = self._preprocess_obs(obs)

            actions = self.actor(obs, deterministic = deterministic)

            obs, rew, done, _ = self.env.step(torch.tanh(actions))
            obs = self._preprocess_obs(obs)

            episode_length += 1

            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.
                    episode_discounted_loss[done_env_id] = 0.
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.
                    games_cnt += 1
        
        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))
 
        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            lam = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError
            
    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(num_games = num_games, deterministic = not self.stochastic_evaluation)
        print_info('mean episode loss = {}, mean discounted loss = {}, mean episode length = {}'.format(mean_policy_loss, mean_policy_discounted_loss, mean_episode_length))
        
    def train(self):
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

        self.time_report.start_timer("algorithm")

        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = torch.int64, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        
        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            detect_anomaly = os.environ.get("ACELAB_SHAC_DETECT_ANOMALY") == "1"
            anomaly_context = torch.autograd.set_detect_anomaly(True) if detect_anomaly else nullcontext()
            with anomaly_context:
                actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.actor_raw_grad_profile = _grad_profile(self.actor.named_parameters())
                raw_norm = self.actor_raw_grad_profile["norm"]
                self.actor_raw_grad_norm_before_sanitize = torch.tensor(
                    raw_norm,
                    dtype=torch.float64,
                    device=self.device,
                )
                debug_grads = os.environ.get("ACELAB_SHAC_DEBUG_GRADS") == "1"
                debug_grad_threshold = float(os.environ.get("ACELAB_SHAC_DEBUG_GRAD_THRESHOLD", "1000000.0"))
                if debug_grads and raw_norm > debug_grad_threshold:
                    _debug_grad_profile("actor raw grad before value clip", self.actor_raw_grad_profile)
                _raise_on_nonfinite_grad_profile("actor raw grad before value clip", self.actor_raw_grad_profile)
                for params in self.actor.parameters():
                    if params.grad is not None:
                        if self.grad_value_clip is not None:
                            params.grad.clamp_(min=-float(self.grad_value_clip), max=float(self.grad_value_clip))
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 
                
                # sanity check
                if not torch.isfinite(self.grad_norm_before_clip):
                    if os.environ.get("ACELAB_SHAC_DEBUG_GRADS") == "1":
                        for name, param in self.actor.named_parameters():
                            grad = param.grad
                            if grad is None:
                                print(f"[DEBUG] actor grad {name}: None")
                                continue
                            finite = torch.isfinite(grad)
                            if not bool(finite.all()):
                                finite_values = grad[finite]
                                if finite_values.numel() > 0:
                                    grad_range = (
                                        float(finite_values.min().detach().cpu()),
                                        float(finite_values.max().detach().cpu()),
                                    )
                                else:
                                    grad_range = None
                                print(
                                    "[DEBUG] non-finite actor grad "
                                    f"{name}: finite={int(finite.sum().detach().cpu())}/{grad.numel()}, "
                                    f"finite_range={grad_range}"
                                )
                                break
                    print('Non-finite gradient')
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure).detach().item()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last = False)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()
                    
                    critic_grad_profile = _grad_profile(self.critic.named_parameters())
                    _raise_on_nonfinite_grad_profile("critic raw grad", critic_grad_profile)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                
                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.time_report.end_timer("critic training")

            self.iter_count += 1
            
            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar('actor_loss/step', self.actor_loss, self.step_count)
            self.writer.add_scalar('actor_loss/iter', self.actor_loss, self.iter_count)
            self.writer.add_scalar('value_loss/step', self.value_loss, self.step_count)
            self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)
            has_episode_stats = len(self.episode_loss_his) > 0
            if has_episode_stats:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
                
                self.writer.add_scalar('policy_loss/step', mean_policy_loss, self.step_count)
                self.writer.add_scalar('policy_loss/time', mean_policy_loss, time_elapse)
                self.writer.add_scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                self.writer.add_scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                self.writer.add_scalar('best_policy_loss/step', self.best_policy_loss, self.step_count)
                self.writer.add_scalar('best_policy_loss/iter', self.best_policy_loss, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_loss = 0.0
                mean_policy_discounted_loss = 0.0
                mean_episode_length = 0.0
            
            print('iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.2f}, grad norm after clip {:.2f}'.format(\
                    self.iter_count, mean_policy_loss, mean_policy_discounted_loss, mean_episode_length, self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch), self.value_loss, self.grad_norm_before_clip, self.grad_norm_after_clip))
            if os.environ.get("ACELAB_SHAC_DEBUG_DONE_REASONS") == "1" and self.termination_reason_counts:
                reasons = ", ".join(
                    f"{name}={count}" for name, count in sorted(self.termination_reason_counts.items())
                )
                unmatched = self.termination_unmatched_done_count
                print(f"[DEBUG] termination reasons: done={self.termination_done_count}, {reasons}, unmatched={unmatched}")
            if os.environ.get("ACELAB_SHAC_DEBUG_DONE_REASONS") == "1" and self.rollout_invalid_counts:
                invalid_reasons = ", ".join(
                    f"{name}={count}" for name, count in sorted(self.rollout_invalid_counts.items())
                )
                print(f"[DEBUG] rollout invalid reasons: {invalid_reasons}")
            if os.environ.get("ACELAB_SHAC_DEBUG_ACTION_STATS") == "1":
                profile = self.actor_action_profile
                print(
                    "[DEBUG] actor tanh action stats: "
                    f"count={profile['count']} min={profile['min']:.6g} "
                    f"max={profile['max']:.6g} mean={profile['mean']:.6g}"
                )

            self.writer.flush()
        
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                if has_episode_stats and np.isfinite(mean_policy_loss):
                    self.save(self.name + "policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))
                else:
                    self.save(self.name + "policy_iter{}".format(self.iter_count))

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)

        self.time_report.end_timer("algorithm")

        self.time_report.report()
        
        self.save('final_policy')

        # save reward/length history
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(open(os.path.join(self.log_dir, 'episode_loss_his.npy'), 'wb'), self.episode_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_discounted_loss_his.npy'), 'wb'), self.episode_discounted_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_length_his.npy'), 'wb'), self.episode_length_his)

        # Final evaluation is optional because callers such as ACELab run checkpoint-reload
        # evaluation outside the algorithm with task-specific safety gates.
        if self.run_final_eval:
            self.run(self.num_envs)

        self.close()
    
    def play(self, cfg):
        self.load(cfg['params']['general']['checkpoint'])
        self.run(cfg['params']['config']['player']['games_num'])
        
    def checkpoint_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "obs_rms": self.obs_rms,
            "ret_rms": self.ret_rms,
            "iter_count": getattr(self, "iter_count", 0),
            "step_count": getattr(self, "step_count", 0),
            "best_policy_loss": getattr(self, "best_policy_loss", np.inf),
        }

    def save(self, filename = None):
        if filename is None:
            filename = 'best_policy'
        torch.save(self.checkpoint_state(), os.path.join(self.log_dir, "{}.pt".format(filename)))
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["target_critic"])
            self.obs_rms = checkpoint.get("obs_rms")
            self.ret_rms = checkpoint.get("ret_rms")
            self.iter_count = checkpoint.get("iter_count", self.iter_count)
            self.step_count = checkpoint.get("step_count", self.step_count)
            self.best_policy_loss = checkpoint.get("best_policy_loss", self.best_policy_loss)
        else:
            self.actor = checkpoint[0].to(self.device)
            self.critic = checkpoint[1].to(self.device)
            self.target_critic = checkpoint[2].to(self.device)
            self.obs_rms = checkpoint[3].to(self.device) if checkpoint[3] is not None else checkpoint[3]
            self.ret_rms = checkpoint[4].to(self.device) if checkpoint[4] is not None else checkpoint[4]

    def close(self):
        self.writer.close()
