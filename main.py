from typing import Dict, List
import gymnasium as gym
from stable_baselines3 import SAC
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback
import pandas as pd


def get_target_ranks_search_space(
    rank: int, min_rank: int, number_of_samples: int
) -> torch.tensor:
    assert (
        isinstance(rank, int) and rank >= 1
    ), f"rank parameter must be a non-negative integer, given: {type(rank)}"
    assert (
        isinstance(min_rank, int) and min_rank >= 1
    ), f"min_rank parameter must be a non-negative integer, given: {type(min_rank)}"
    assert (
        isinstance(number_of_samples, int) and number_of_samples >= 1
    ), f"number_of_samples parameter must be a non-negative integer, given: {type(number_of_samples)}"
    assert (
        rank > min_rank
    ), f"Rank parameter ({rank}) must be grater than min_rank ({min_rank})"

    rank_range = rank - min_rank
    if number_of_samples >= rank_range:
        print(
            f"Cannot generate {number_of_samples} samples in the given rank range. Defaulting to using rank-min_rank={rank_range} samples"
        )
        number_of_samples = rank_range
    step_size = rank_range // number_of_samples
    print(f"Using step_size={step_size}")

    return torch.range(
        start=min_rank, end=rank, step=step_size, dtype=torch.int
    ).tolist()


def get_low_rank_approximation(matrix: torch.Tensor, target_rank: int) -> torch.Tensor:
    u, s, v = torch.svd_lowrank(matrix, q=target_rank)

    return u @ torch.diag(s) @ v.T


ENVIRONMENT = "Walker2d-v4"
TOTAL_TIMESTEPS = 1e6
SEED = 42
HIDDEN_SIZE = 256
N_EVAL_EPISODES = 100  # using high number of eval episodes for better approximation
TRAIN_FROM_SCRATCH = False
MODEL_SAVE_PATH = "./tmp/best_model"
TENSORBOARD_PATH = "./tmp/tensorboard/SAC"
RESULTS_FILE_PATH = "./tmp/intervention_results.csv"

POLICY_KWARGS = {
    "net_arch": dict(
        pi=[HIDDEN_SIZE, HIDDEN_SIZE],
        vf=[HIDDEN_SIZE, HIDDEN_SIZE],
        qf=[HIDDEN_SIZE, HIDDEN_SIZE],
    ),
    "activation_fn": torch.nn.ReLU,
}
# Hyperparameters taken from SAC-CEPO paper [3] (see README.md)
HYPERPARAMETERS = {
    "learning_rate": 0.0003,
    "gamma": 0.99,  # Called discount in the paper
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,  # Polyak-Ruppert smoothing coef
}

if __name__ == "__main__":
    env = gym.make(ENVIRONMENT)
    eval_env = gym.make(ENVIRONMENT)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=POLICY_KWARGS,
        **HYPERPARAMETERS,
        tensorboard_log=TENSORBOARD_PATH,
    )
    if TRAIN_FROM_SCRATCH:

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=MODEL_SAVE_PATH,
            log_path="./logs/",
            eval_freq=1000,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, progress_bar=True, callback=[eval_callback]
        )

    model = SAC.load(
        f"{MODEL_SAVE_PATH}/best_model.zip",
        env=env,
    )

    BASELINE_MEAN_REWARD, BASELINE_STD_REWARD = evaluate_policy(
        model, eval_env, n_eval_episodes=N_EVAL_EPISODES
    )
    print(
        f"Not modified model: mean_reward={BASELINE_MEAN_REWARD}, std_reward={BASELINE_STD_REWARD}"
    )

    results: List[Dict[str, float]] = []

    # Model's policy is represented as the two NNs: latent_pi and mu
    for layer in [4, 0, 2]:
        if layer != 4:
            LAYER_WEIGHTS = torch.clone(model.policy.actor.latent_pi[layer].weight)
        else:
            LAYER_WEIGHTS = torch.clone(model.policy.actor.mu.weight)

        layer_rank = min(
            LAYER_WEIGHTS.size()
        )  # SGD produces full rank matricies so we don't have to calculate it from definition.
        print(f"Layer {layer} has rank: {layer_rank}")
        layer_rank_search_space = get_target_ranks_search_space(
            rank=layer_rank, min_rank=1, number_of_samples=10
        )

        print(
            f"Processing layer number: {layer} with rank of weight matrix = {layer_rank}"
        )
        print(
            f"Layer rank search space: {layer_rank_search_space} number of points: {len(layer_rank_search_space)}"
        )
        for target_rank in layer_rank_search_space:
            # Low rank aproxximation
            low_rank_approx = get_low_rank_approximation(
                matrix=LAYER_WEIGHTS, target_rank=target_rank
            )
            if layer != 4:
                model.policy.actor.latent_pi[layer].weight = torch.nn.Parameter(
                    low_rank_approx.to(model.device)
                )
            else:
                model.policy.actor.mu.weight = torch.nn.Parameter(
                    low_rank_approx.to(model.device)
                )
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=N_EVAL_EPISODES
            )

            mean_reward_delta, std_reward_delta = (
                mean_reward - BASELINE_MEAN_REWARD,
                std_reward - BASELINE_STD_REWARD,
            )
            intervention_results = {
                "layer": layer,
                "target_rank": target_rank,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_reward_delta": mean_reward_delta,
                "std_reward_delta": std_reward_delta,
            }
            print(intervention_results)
            results.append(intervention_results)

    results_df = pd.DataFrame.from_records(results)
    results_df.to_csv(RESULTS_FILE_PATH, header=True, sep=";", index=False)
    print("Results saved to: f{RESULTS_FILE_PATH}")
