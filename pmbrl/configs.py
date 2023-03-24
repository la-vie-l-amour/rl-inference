import pprint

MOUNTAIN_CAR_CONFIG = "SparseMountainCar-v0"
PENDULUM_CONFIG = "Pendulum-v1"
FROZENLAKE_CONFIG = "FrozenLake-v1"

def print_configs():
    print(f"[{MOUNTAIN_CAR_CONFIG}, {PENDULUM_CONFIG} ,{FROZENLAKE_CONFIG}]")

def get_config(args):

    if args.config_name == PENDULUM_CONFIG:
        config = PendulumConfig()
    elif args.config_name == MOUNTAIN_CAR_CONFIG:
        config = MountainCarConfig()
    elif args.config_name == FROZENLAKE_CONFIG:
        config = FrozenLakeConfig()
    else:
        raise ValueError("`{}` is not a valid config ID".format(args.config_name))

    # config.set_logdir(args.logdir)
    config.set_seed(args.seed)
    config.set_strategy(args.strategy)
    return config


class Config(object):
    def __init__(self):
        self.logdir = "log"
        self.seed = 0
        self.n_episodes = 50
        self.n_seed_episodes = 5
        self.record_every = None
        self.coverage = False

        self.env_name = None
        self.max_episode_len = 500   # max_episode_len表示一轮的最大步数
        self.action_repeat = 3       #走一步，走多少步
        self.action_noise = None

        self.ensemble_size = 10
        self.hidden_size = 200

        self.n_train_epochs = 100
        self.batch_size = 50
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.grad_clip_norm = 1000

        self.plan_horizon = 30
        self.optimisation_iters = 5
        self.n_candidates = 500
        self.top_candidates = 50

        self.expl_strategy = "information"
        self.use_reward = True
        self.use_exploration = True
        self.use_mean = False

        self.expl_scale = 1.0
        self.reward_scale = 1.0

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_seed(self, seed):
        self.seed = seed

    def set_strategy(self, strategy):
        self.strategy = strategy

    def __repr__(self):
        return pprint.pformat(vars(self))


class PendulumConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "Pendulum-v1"
        self.env_name = "Pendulum-v1"
        self.n_episodes = 5
        self.max_episode_len = 100
        self.hidden_size = 64
        self.plan_horizon = 5
        self.record_every = 0



# 调整超参，训练过程很不尽如人意
class FrozenLakeConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "FrozenLake-v1"
        self.env_name = "FrozenLake-v1"
        self.max_episode_len = 500
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.expl_scale = 1.0
        self.ensemble_size = 30
        self.record_every = 0  # 改，从None改为0
        self.n_episodes = 50


class MountainCarConfig(Config):
    def __init__(self):
        super().__init__()
        self.logdir = "SparseMountainCar-v0"
        self.env_name = "SparseMountainCar-v0"
        self.max_episode_len = 500
        self.n_train_epochs = 100
        self.n_seed_episodes = 1
        self.expl_scale = 1.0
        self.n_episodes = 30
        self.ensemble_size = 25
        self.record_every = 0  #改，从None改为0
        self.n_episodes = 50

