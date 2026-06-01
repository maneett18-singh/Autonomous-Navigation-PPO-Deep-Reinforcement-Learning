import argparse
import os
import random
import time
try:
    from distutils.util import strtobool  # Python <3.12
except Exception:  # pragma: no cover
    def strtobool(val):
        val = str(val).strip().lower()
        if val in {"y", "yes", "t", "true", "on", "1"}:
            return 1
        if val in {"n", "no", "f", "false", "off", "0"}:
            return 0
        raise ValueError(f"invalid truth value {val!r}")
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from train_agent import RLWrapper
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--env-type", type=str, default="parcour",
        help="the type of race environment: 'straight' or 'curvy'")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Reward shaping (passed to RLWrapper)
    parser.add_argument("--w-dist", type=float, default=2.0, help="reward weight for delta distance")
    parser.add_argument("--w-dist-abs", type=float, default=0.01, help="reward weight for absolute distance penalty")
    parser.add_argument("--w-align", type=float, default=0.1, help="reward weight for gate alignment")
    parser.add_argument("--w-speed", type=float, default=0.03, help="reward weight for forward speed")
    parser.add_argument("--w-ang", type=float, default=0.05, help="penalty weight for angular velocity")
    parser.add_argument("--w-time", type=float, default=0.02, help="per-step time penalty weight")
    parser.add_argument("--w-idle", type=float, default=0.1, help="penalty weight for idle/low-speed")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    # Early stopping (optional)
    parser.add_argument("--early-stop", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="enable early stopping based on episode metrics")
    parser.add_argument("--early-stop-patience", type=int, default=200,
        help="number of consecutive 'solved' episodes required to stop")
    parser.add_argument("--early-stop-return", type=float, default=None,
        help="stop if episodic_return >= this value (used with patience)")
    parser.add_argument("--early-stop-require-all-gates", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if True, only count an episode as 'solved' when gates_passed == gates_total")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def _save_model(agent: nn.Module, args, run_name: str) -> str:
    os.makedirs("models", exist_ok=True)
    model_path = f"models/PPO_{args.env_type}_{run_name}.pth"
    torch.save(
        {
            "actor_mean": agent.actor_mean.state_dict(),
            "actor_logstd": agent.actor_logstd,  # tensor/Parameter
            "critic": agent.critic.state_dict(),
            "args": vars(args),
            "run_name": run_name,
        },
        model_path,
    )
    return model_path


def make_env(env_type, seed, idx, capture_video, run_name):
    def thunk():
        # Create RLWrapper instead of gym.make
        render_mode = "rgb_array" if (capture_video and idx == 0) else None
        env = RLWrapper(
            env_type=env_type,
            render_mode=render_mode,
            w_dist=args.w_dist,
            w_dist_abs=args.w_dist_abs,
            w_align=args.w_align,
            w_speed=args.w_speed,
            w_ang=args.w_ang,
            w_time=args.w_time,
            w_idle=args.w_idle,
        )  # "straight" or "curvy" or "parcour"
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/eval/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10),
            observation_space=env.observation_space,
        )
        # NormalizeReward can produce large values early if return var is tiny;
        # clip + sanitize after normalization for stability.
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(np.nan_to_num(r, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)))
        # Note: RLWrapper uses gymnasium, so use reset(seed=seed) instead of env.seed()
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        action_mean = self.actor_mean(x)
        action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=0.0, neginf=0.0)
        action_logstd = self.actor_logstd.expand_as(action_mean).clamp(-20.0, 2.0)
        action_std = torch.exp(action_logstd).clamp(1e-6, 1e2)
        probs = Normal(action_mean, action_std, validate_args=False)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_type}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # wandb's gym monitor is not compatible with gymnasium's RecordVideo wrapper
            # and can break env.close() with AttributeError: 'RecordVideo' object has no attribute 'enabled'
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_type, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    # Early stopping state
    solved_streak = 0
    solved_returns = deque(maxlen=max(1, args.early_stop_patience))
    should_stop_early = False
    steps_to_first_solve = None
    steps_to_best_solve = None
    best_solved_ep_len = None
    solved_episodes = 0
    steps_to_stable_solve = None
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Log episode stats when episode ends
            if done.any() and "episode" in info:
                # print("Episode finished!",info)
                ep = info["episode"]
                ep_return = float(ep["r"][0])
                ep_len = int(ep["l"][0])

                print(f"global_step={global_step}, episodic_return={ep['r'][0]}")
                writer.add_scalar("charts/episodic_return", ep["r"][0], global_step)
                writer.add_scalar("charts/episodic_length", ep["l"][0], global_step)
                
                # Log custom reward components
                if "episode_dist_reward" in info:
                    writer.add_scalar("charts/episode_dist_reward", info["episode_dist_reward"][0], global_step)
                if "episode_alignment_reward" in info:
                    writer.add_scalar("charts/episode_alignment_reward", info["episode_alignment_reward"][0], global_step)
                if "episode_raw_reward" in info:
                    writer.add_scalar("charts/episode_raw_reward", info["episode_raw_reward"][0], global_step)
                if "episode_gates_passed" in info:
                    gp = int(info["episode_gates_passed"][0])
                    gt = int(info["episode_gates_total"][0])
                    writer.add_scalar("charts/gates_passed", gp, global_step)
                    writer.add_scalar("charts/gates_total", gt, global_step)
                else:
                    gp, gt = None, None

                # -------- First-solve tracking (for sweeps) --------
                if gp is not None and gt is not None and gp >= gt:
                    solved_episodes += 1
                    writer.add_scalar("charts/solved_episodes", solved_episodes, global_step)

                    if steps_to_first_solve is None:
                        steps_to_first_solve = global_step
                        writer.add_scalar("charts/steps_to_first_solve", steps_to_first_solve, global_step)

                    # Stable solve: require K solved episodes (reduces lucky one-offs)
                    # if steps_to_stable_solve is None and solved_episodes >= int(args.stable_solve_count):
                    #     steps_to_stable_solve = global_step
                    #     writer.add_scalar("charts/steps_to_stable_solve", steps_to_stable_solve, global_step)

                    # -------- Best-solve tracking (minimum solved episode length) --------
                    if best_solved_ep_len is None or ep_len < best_solved_ep_len:
                        best_solved_ep_len = ep_len
                        steps_to_best_solve = global_step
                        writer.add_scalar("charts/best_solved_ep_len", best_solved_ep_len, global_step)
                        writer.add_scalar("charts/steps_to_best_solve", steps_to_best_solve, global_step)
                # -------- Early stopping check --------
                if args.early_stop:
                    meets_gates = True
                    if args.early_stop_require_all_gates:
                        meets_gates = (gp is not None and gt is not None and gp >= gt)

                    meets_return = True
                    if args.early_stop_return is not None:
                        meets_return = (ep_return >= float(args.early_stop_return))

                    solved = bool(meets_gates and meets_return)

                    if solved:
                        solved_streak += 1
                        solved_returns.append(ep_return)
                    else:
                        solved_streak = 0
                        solved_returns.clear()

                    writer.add_scalar("charts/solved_streak", solved_streak, global_step)

                    if solved_streak >= args.early_stop_patience:
                        mean_ret = float(np.mean(solved_returns)) if len(solved_returns) else ep_return
                        print(
                            f"Early stop: solved_streak={solved_streak} "
                            f"(patience={args.early_stop_patience}), mean_return={mean_ret:.3f}"
                        )
                        should_stop_early = True
                        break
            # --------------------------------------

        if should_stop_early:
            break


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                if not torch.isfinite(loss):
                    writer.add_scalar("debug/nonfinite_loss", 1.0, global_step)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Save the trained model
    # Ensure sweep metric exists even if never solved
    if steps_to_first_solve is None:
        writer.add_scalar("charts/steps_to_first_solve", args.total_timesteps, global_step)
    if steps_to_stable_solve is None:
        writer.add_scalar("charts/steps_to_stable_solve", args.total_timesteps, global_step)
    if steps_to_best_solve is None:
        writer.add_scalar("charts/steps_to_best_solve", args.total_timesteps, global_step)
    if best_solved_ep_len is None:
        writer.add_scalar("charts/best_solved_ep_len", -1, global_step)

    # model_path = _save_model(agent, args, run_name)
    # print(f"Model saved to {model_path}")

    envs.close()
    writer.close()