# pylint: disable=not-callable
# pylint: disable=no-member

import sys
import time
import pathlib
import argparse

import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder


sys.path.append(str(pathlib.Path(__file__).parent.parent))

from pmbrl.envs import GymEnv
from pmbrl.envs import Env
from pmbrl.training import Normalizer, Buffer, Trainer
from pmbrl.models import EnsembleModel, RewardModel
from pmbrl.control import Planner, Agent
from pmbrl.utils import Logger
from pmbrl import get_config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")






def main(args):


    logger = Logger(args.logdir, args.seed)
    logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
    logger.log(args)

    # rate_buffer = None
    # if args.coverage:
    #     from pmbrl.envs.envs.ant import rate_buffer

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    env = GymEnv(
        args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed
    )
    # env = Env(
    #     args.env_name, args.max_episode_len, action_repeat=args.action_repeat, seed=args.seed
    # )

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    normalizer = Normalizer()
    buffer = Buffer(state_size, action_size, args.ensemble_size, normalizer, device=DEVICE)

    ensemble = EnsembleModel(
        state_size + action_size,
        state_size,
        args.hidden_size,
        args.ensemble_size,
        normalizer,
        device=DEVICE,
    )
    reward_model = RewardModel(state_size + action_size, args.hidden_size, device=DEVICE)
    trainer = Trainer(
        ensemble,
        reward_model,
        buffer,
        n_train_epochs=args.n_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        grad_clip_norm=args.grad_clip_norm,
        logger=logger,
    )

    planner = Planner(
        ensemble,
        reward_model,
        action_size,
        args.ensemble_size,
        plan_horizon=args.plan_horizon,
        optimisation_iters=args.optimisation_iters,
        n_candidates=args.n_candidates,
        top_candidates=args.top_candidates,
        use_reward=args.use_reward,
        use_exploration=args.use_exploration,
        use_mean=args.use_mean,
        expl_scale=args.expl_scale,
        reward_scale=args.reward_scale,
        strategy=args.strategy,
        device=DEVICE,
    )
    agent = Agent(env, planner, logger=logger)
    # 1. Agent首先与环境互动执行一段时间，目的是获得experience对，或者说是先初始化buffer
    agent.get_seed_episodes(buffer, args.n_seed_episodes)
    msg = "\nCollected seeds: [{} episodes | {} frames]"
    logger.log(msg.format(args.n_seed_episodes, buffer.total_steps))
    # 自己改
    # trainer.reset_models()
    eposide_rewads = []
    for episode in range(1, args.n_episodes):
        logger.log("\n=== Episode {} ===".format(episode))
        start_time = time.time()

        msg = "Training on [{}/{}] data points"
        logger.log(msg.format(buffer.total_steps, buffer.total_steps * args.action_repeat))
        # 2.对transition model 和reward model进行初始化,为何要每轮都要进行模型的初始化？？？，其实这里是训练了n_episodes 个model,
        trainer.reset_models()
        # 3. 对上面的模型进行训练
        ensemble_loss, reward_loss = trainer.train()
        logger.log_losses(ensemble_loss, reward_loss)

        recorder = None
        if args.record_every is not None and args.record_every % episode == 0:
            filename = logger.get_video_path(episode)
            recorder = VideoRecorder(env.unwrapped, path=filename)
            logger.log("Setup recoder @ {}".format(filename))

        logger.log("\n=== Collecting data [{}] ===".format(episode))
        # 4. Agent 跑一局
        reward, steps, stats = agent.run_episode(
            buffer, action_noise=args.action_noise, recorder=recorder
        )
        eposide_rewads.append(reward)
        logger.log_episode(reward, steps)
        logger.log_stats(stats)

        # if args.coverage:
        #     coverage = rate_buffer(buffer=buffer)
        #     logger.log_coverage(coverage)

        logger.log_time(time.time() - start_time)
        logger.save()

    logger._save_fig(eposide_rewads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="Pendulum-v1")  # Pendulum-v1, SparseMountainCar-v0,FrozenLake-v1
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()   # args, unknown = parser.parse_know_args()
    config = get_config(args)
    main(config)
