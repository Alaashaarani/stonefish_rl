import os
import argparse
import sys
import datetime as dt
from copy import deepcopy
import time
import math
import yaml
from pathlib import Path

from stable_baselines3 import SAC,PPO,TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from library.training_seeding import training_seed
from library.docking_env import dsEnv
from library.env_factory import make_env_instance
from library.common import resolve_model_path, resolve_path
from library.configuration import load_config as load_workflow_config, DEFAULT_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEARNING_RATES = {
    "PPO": 3e-4,
    "SAC": 3e-4,
    "TD3": 1e-3,
}


class StopTrainingOnFileCallback(BaseCallback):
    """Stop SB3 learning cleanly when a request file appears."""

    def __init__(self, stop_file, check_freq=100, verbose=1):
        super().__init__(verbose=verbose)
        self.stop_file = Path(stop_file).expanduser() if stop_file else None
        self.check_freq = max(1, int(check_freq))

    def _on_step(self) -> bool:
        if self.stop_file is None:
            return True
        if self.n_calls % self.check_freq != 0:
            return True
        if self.stop_file.exists():
            if self.verbose:
                print(f"[INFO] Stop request detected: {self.stop_file}")
                print("[INFO] Finishing model.learn() cleanly...")
            return False
        return True


class StopTrainingOnSuccessRate(BaseCallback):
    """Stop on cumulative high-success evaluations or a later performance drop."""

    def __init__(self, threshold, verbose=1, required_evaluations=3, drop_threshold=0.6):
        super().__init__(verbose=verbose)
        if type(required_evaluations) is not int or required_evaluations < 1:
            raise ValueError('success_rate_required_evaluations must be a positive integer.')
        if (isinstance(drop_threshold, bool) or not isinstance(drop_threshold, (int, float))
                or not math.isfinite(drop_threshold) or not 0 <= drop_threshold <= threshold):
            raise ValueError('success_rate_drop_threshold must be between 0 and the success threshold.')
        self.threshold = threshold
        self.required_evaluations = required_evaluations
        self.drop_threshold = drop_threshold
        self.success_evaluations = 0
        self.stopped_on_drop = False
        self.last_success_rate = None
        self.stopped_on_success = False

    def _on_step(self) -> bool:
        if not isinstance(self.parent, EvalCallback):
            raise RuntimeError('Success-rate stopping must run after an EvalCallback.')
        successes = self.parent._is_success_buffer
        expected = self.parent.n_eval_episodes
        if len(successes) != expected:
            raise RuntimeError(
                f'Evaluation returned {len(successes)} success flags for {expected} episodes; '
                "every completed episode must provide info['is_success']."
            )
        count = sum(bool(success) for success in successes)
        self.last_success_rate = count / expected
        if self.last_success_rate > self.threshold:
            self.success_evaluations += 1
        self.stopped_on_success = self.success_evaluations >= self.required_evaluations
        self.stopped_on_drop = self.success_evaluations > 0 and self.last_success_rate < self.drop_threshold
        if self.verbose:
            print(f'[INFO] Evaluation success {count}/{expected} ({self.last_success_rate:.1%}); '
                  f'evaluations above {self.threshold:.1%}: '
                  f'{self.success_evaluations}/{self.required_evaluations}.')
            if self.stopped_on_success or self.stopped_on_drop:
                reason = 'success target reached' if self.stopped_on_success else 'success dropped below the floor after a high-success evaluation'
                print(f'[INFO] {reason}; finishing this stage and saving its model.')
        return not (self.stopped_on_success or self.stopped_on_drop)



def training_success_threshold(config):
    value = config.get('train', {}).get('success_rate_threshold')
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError('library.training.success_rate_threshold must be a number in [0, 1), or null to disable.')
    StopTrainingOnSuccessRate(
        float(value), verbose=0,
        required_evaluations=config['train'].get('success_rate_required_evaluations', 3),
        drop_threshold=config['train'].get('success_rate_drop_threshold', min(0.6, float(value))))
    return float(value)


def make_training_eval_callback(config, eval_env, log_dir):
    _, episodes = training_evaluation_settings(config)
    threshold = training_success_threshold(config)
    return EvalCallback(
        eval_env,
        callback_after_eval=StopTrainingOnSuccessRate(
            threshold,
            required_evaluations=config['train'].get('success_rate_required_evaluations', 3),
            drop_threshold=config['train'].get('success_rate_drop_threshold', min(0.6, threshold)),
        ) if threshold is not None else None,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=config['train']['eval_freq'],
        n_eval_episodes=episodes,
        deterministic=True,
        render=False,
    )


def load_config(config_path):
    return load_workflow_config(config_path, "training")


def resolve_repo_path(path_value, default_path=None):
    if path_value in (None, "", "default"):
        return Path(default_path)

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def make_env(rank, config):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    :param config: Yaml file contains all required parameters
    """
    def _init():
        seed = training_seed(config)
        if seed is not None:
            set_random_seed(seed + rank)
        # Paths to your configs
        env = make_env_instance(config.get("env", {}).get("type", "docking"), rank, config)
        if seed is not None:
            env.action_space.seed(seed + rank)
        return env
    return _init

def linear_schedule(initial_lr, final_lr):
    def schedule(progress_remaining):
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


def is_auto(value):
    return isinstance(value, str) and value.strip().lower() in ("auto", "default")


def parse_learning_rate_value(value, algorithm):
    if is_auto(value):
        return DEFAULT_LEARNING_RATES.get(str(algorithm).upper(), 3e-4)
    return float(value)


def make_learning_rate(config):
    """Build SB3 learning_rate from YAML.

    Supported forms:
      model.learning_rate: auto | 0.0003
      model.learning_rate_start/end: auto | numeric

    If start and end differ, returns the existing linear schedule. If they are
    equal, returns a constant float.
    """
    model_config = config["model"]
    algorithm = str(model_config.get("algorithm", "PPO")).upper()

    if "learning_rate" in model_config:
        learning_rate = parse_learning_rate_value(model_config["learning_rate"], algorithm)
        print(f"[INFO] Learning rate: {learning_rate} ({model_config['learning_rate']})")
        return learning_rate

    start_value = model_config.get("learning_rate_start", "auto")
    end_value = model_config.get("learning_rate_end", start_value)
    start = parse_learning_rate_value(start_value, algorithm)
    end = parse_learning_rate_value(end_value, algorithm)

    if start == end:
        print(f"[INFO] Constant learning rate: {start}")
        return start

    print(f"[INFO] Linear learning rate schedule: start={start}, end={end}")
    return linear_schedule(start, end)


def training_evaluation_settings(config):
    train = config["train"]
    values = (train.get("eval_instances", 1), train.get("eval_episodes", 5))
    for key, value in zip(("eval_instances", "eval_episodes"), values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"library.training.{key} must be a positive integer")
    return values


def make_training_eval_env(config, log_dir):
    workers, _ = training_evaluation_settings(config)
    # Keep evaluation ranks beyond all training ranks (and preserve the old gap).
    first_rank = config["env"]["instances"] + 1
    factories = []
    for i in range(workers):
        eval_config = deepcopy(config)
        # Only the first evaluation simulator may render, independently of training.
        eval_config["sim"]["graphical_interface"] = (
            i == 0 and bool(config["sim"]["evaluation_graphical_interface"])
        )
        factories.append(make_env(first_rank + i, eval_config))
    env = SubprocVecEnv(factories)
    env = VecMonitor(env, str(Path(log_dir) / "evaluation"))
    seed = training_seed(config)
    if seed is not None:
        env.seed(seed + first_rank)  # Seed first reset only; subsequent resets advance the RNG.
    return env


def env_loop(config,num_instances): 
    list = []
    
    for i in range(num_instances): 
        list.append(make_env(i,config))

    return list

def run_training(config):
    eval_instances, eval_episodes = training_evaluation_settings(config)
    training_success_threshold(config)  # Validate before creating simulator workers.
    seed = training_seed(config)
    if seed is not None:
        set_random_seed(seed)
        print(f"[INFO] Training seed: {seed}; workers {seed}..{seed + config['env']['instances'] - 1}; "
              f"evaluation seed: {seed + config['env']['instances'] + 1}")

    # 1. Configuration
    log_dir = resolve_repo_path(config["log"].get("log_dir"), REPO_ROOT / "logs")
    models_dir = resolve_repo_path(config["train"].get("save_path"), REPO_ROOT / "models")
    wandb_dir = log_dir / "wandb"
    tensorboard_dir = log_dir / "runs"
    wandb_model_dir = log_dir / "wandb_models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(wandb_model_dir, exist_ok=True)
        
    num_instances = config["env"]["instances"]

    train_env = eval_env = run = None
    try:
        # 2. Create Parallel Environments
        # VecMonitor logs episode rewards/lengths for tensorboard

        train_env = SubprocVecEnv(env_loop(config,num_instances))

        train_env = VecMonitor(train_env, str(log_dir))

        # 3. Create parallel evaluation environments on separate simulator ports.
        time.sleep(2)  # Ensure different ports
        print(f"[INFO] Evaluation: {eval_instances} workers, {eval_episodes} total episodes per callback")
        eval_env = make_training_eval_env(config, log_dir)

        # 4. Callback to evaluate and save the best model
        eval_callback = make_training_eval_callback(config, eval_env, log_dir)

        # Record the resolved configuration separately for each training invocation.
        if config["log"]["enable_wandb"]:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            name = config["model"]["algorithm"]+"_"+config["model"]["policy"]+"_"+ dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run = wandb.init(
                        id = name if config["log"]["run_name"]=="default" else config["log"]["run_name"],
                        project=config["log"]["project_name"],
                        config=deepcopy(config),
                        tags = config["log"]["tags"],
                        dir=str(wandb_dir),
                        sync_tensorboard=True,  # This automatically uploads your SB3 logs
                        monitor_gym=True,       # This upload videos of the robot
                        save_code=True, )
            tensorboard_log = str(tensorboard_dir / run.id)

            callback = [eval_callback, WandbCallback(
                        gradient_save_freq=100,
                        model_save_path=str(wandb_model_dir / run.id),
                        verbose=2,    )]
        else:
            run = None
            tensorboard_log = str(tensorboard_dir / dt.datetime.now().strftime("local_%Y%m%d_%H%M%S"))
            callback = eval_callback

        stop_file = config.get("train", {}).get("stop_file")
        if stop_file:
            stop_file = resolve_repo_path(stop_file)
            stop_check_freq = config.get("train", {}).get("stop_check_freq", 100)
            stop_callback = StopTrainingOnFileCallback(stop_file, stop_check_freq)
            callback = callback if isinstance(callback, list) else [callback]
            callback.append(stop_callback)
            print(f"[INFO] Graceful stop file enabled: {stop_file}")

        learning_rate = make_learning_rate(config)
        # 5. IModel Initilization
        # MlpPolicy is standard for vector/sensor observations
        if config["model"]["algorithm"]=="SAC":
            if config["model"]["pretrained"]:
                model = SAC.load(resolve_model_path(config["model"]["model_path"]),
                                env=train_env,
                                seed=seed,
                                tensorboard_log=tensorboard_log,
                                weights_only=config["model"].get("weights_only", True)
                                )
            else:
                model = SAC(
                    config["model"]["policy"],
                    train_env,
                    train_freq=(1, "step"), # Collect steps from each env before updating
                    gradient_steps=config["model"]["gradient_steps"],       # Do  gradient updates
                    verbose=1,
                    learning_rate=learning_rate,
                    seed=seed,
                    tensorboard_log=tensorboard_log,
                    buffer_size=config["model"]["buffer_size"],
                    learning_starts=config["model"]["learning_starts"],
                    ent_coef = config["model"]["ent_coef"]
                )
        elif config["model"]["algorithm"]=="PPO":
            if config["model"]["pretrained"]:
                model = PPO.load(resolve_model_path(config["model"]["model_path"]),
                                env=train_env,
                                seed=seed,
                                tensorboard_log=tensorboard_log,
                                weights_only=config["model"].get("weights_only", True)
                                )
            else:
                model = PPO(config["model"]["policy"],
                            env=train_env,
                            learning_rate= learning_rate,
                            n_steps= config["model"]["n_steps"],
                            batch_size= config["model"]["batch_size"],
                            # ent_coef= config["model"]["ent_coef"],
                            clip_range= config["model"].get("clip_range", 0.2),
                            seed=seed,
                            tensorboard_log=tensorboard_log
                            )

        elif config["model"]["algorithm"]=="TD3":
            if config["model"]["pretrained"]:
                model = TD3.load(resolve_model_path(config["model"]["model_path"]),
                                env=train_env,
                                seed=seed,
                                tensorboard_log=tensorboard_log,
                                 weights_only=config["model"].get("weights_only", True)
                                )
            else:
                model = TD3(config["model"]["policy"],
                            env=train_env,
                            learning_rate= learning_rate,
                            buffer_size=config["model"]["buffer_size"],
                            batch_size= config["model"]["batch_size"],
                            policy_delay= config["model"]["policy_delay"],
                            seed=seed,
                            tensorboard_log=tensorboard_log
                            )


        # 6. Train the model
        print(f"Starting training with {num_instances} instances...")
        starting_time = time.time()
        try:
            model.learn(
                total_timesteps=config["train"]["total_timesteps"],
                reset_num_timesteps=config["train"].get("reset_num_timesteps", False),
                callback= callback
                        )

        except KeyboardInterrupt:
            print("Training interrupted by user.")
        finally:
            output_model_path = config["train"].get("output_model_path")
            if output_model_path:
                name = resolve_repo_path(output_model_path)
            else:
                model_name_prefix = config["train"].get(
                    "warm_start_path",
                    config["model"]["algorithm"] + "_",
                )
                name = models_dir / (
                    model_name_prefix
                    + config["train"]["save_name"] + "_"
                    + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                )
            # 7. Final Save and Cleanup
            name.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(name))
            print(f"[INFO] Saved model to: {name}")

    finally:
        # Also clean up if checkpoint loading, W&B initialization or saving fails.
        try:
            if eval_env is not None:
                eval_env.close()
        finally:
            try:
                if train_env is not None:
                    train_env.close()
            finally:
                if run is not None:
                    run.finish()

    end_time = time.time()
    elapsed_time = end_time - starting_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes.")


