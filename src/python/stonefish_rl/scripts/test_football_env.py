import argparse

import yaml

from stonefish_rl.envs.football_env import FootballEnv
from stonefish_rl.utils.utils import resolve_path


def load_config(config_path):
    with open(resolve_path(config_path), "r") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run random-action football env test episodes.")
    parser.add_argument(
        "config",
        nargs="?",
        default="include/parameters/football_param.yaml",
        help="Path to football parameter YAML.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run. Overrides testing.episodes in the YAML.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    num_episodes = args.episodes
    if num_episodes is None:
        num_episodes = int(config.get("testing", {}).get("episodes", 1))
    num_episodes = max(1, int(num_episodes))
    step_per_print = int(config.get("testing", {}).get("step_per_print", 10))
    
    env = FootballEnv(0, config)
    max_steps = int(
        config["env"]["episode_duration"] * config["env"]["rl_observation_freq"]
    )
    all_episode_rewards = []

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Initial observation shape: {obs.shape}")

            total_reward = 0.0
            terminated = False
            truncated = False

            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if step_per_print > 0 and step % step_per_print == 0:
                    print(
                        f"episode={episode + 1} step={step} "
                        f"reward={reward:.3f} total={total_reward:.3f} "
                        f"robot_ball={info['robot_ball_distance']:.3f} "
                        f"ball_goal={info['ball_goal_distance']:.3f}"
                    )

                if terminated or truncated:
                    break

            all_episode_rewards.append(total_reward)
            print(
                f"Episode {episode + 1} finished: "
                f"terminated={terminated}, truncated={truncated}, "
                f"total_reward={total_reward:.3f}"
            )
    finally:
        env.close()

    mean_reward = sum(all_episode_rewards) / len(all_episode_rewards)
    print(f"\nCompleted {len(all_episode_rewards)} episode(s). Mean reward: {mean_reward:.3f}")
