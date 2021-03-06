from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sys
import argparse

import joblib


from osim_http_client import Client


def run_task(*_):
    global params
    env = normalize(Client())

    policy = params['policy']# GaussianMLPPolicy(
    #     env_spec=env.spec,
    #     # The neural network policy should have two hidden layers, each with 32 hidden units.
    #     hidden_sizes=(100, 50, 25)
    # )

    baseline = params['baseline'] #LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=50000,
        max_path_length=500,
        n_itr=100,
        discount=0.99,
        step_size=0.005,
        threads=7,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from', type=str,
                        help='path of pickled file')
    parser.add_argument(
        '--exp_name', type=str, help='Name of the experiment.')
    args = parser.parse_args()
    params = joblib.load(args.resume_from)
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=4,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
        exp_name=args.exp_name,
    #     log_dir='my_exps'
    )
