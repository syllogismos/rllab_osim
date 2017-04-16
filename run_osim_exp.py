


import argparse
from rllab.misc.ext import is_iterable, set_seed
from rllab import config
import rllab.misc.logger as logger
import os.path as osp
import datetime
import dateutil.tz
import ast
import uuid
import pickle
import base64
import joblib

import logging

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sys
import joblib

from osim_http_client import Client

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='Name of the exp folder')
    parser.add_argument('--n_parallel', type=int, default=1)
    parser.add_argument('--snapshot_mode', type=str, default='all')
    parser.add_argument('--snapshot_gap', type=int, default=1)
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv')
    parser.add_argument('--text_log_file', type=str, default='debug.log')
    parser.add_argument('--params_log_file', type=str, default='params.json')
    # parser.add_argument('--variant_log_file', type=str, default='variant.json')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=50000)
    parser.add_argument('--max_path_length', type=int, default=500)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--step_size', type=float, default=0.005)
    parser.add_argument('--n_itr', type=int, default=100)
    parser.add_argument('--args_data', default=None)
    # parser.add_argument('--variant_data', type=str)


    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
    rand_id = str(uuid.uuid4())[:5]
    exp_name = '%s_%s_%s' % (args.exp_name, timestamp, rand_id)
    log_dir = osp.join(config.LOG_DIR, exp_name)
    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)
    params_log_file = osp.join(log_dir, args.params_log_file)

    # if args.variant_data is not None:
    #     variant_data = pickle

    # wtf is cloud pickle and what is it doing in run exp lite???
    logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)    
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    prev_gap = logger.get_snapshot_gap()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_snapshot_gap(args.snapshot_gap)
    logger.set_log_tabular_only(False)
    logger.push_prefix("[%s] " % args.exp_name)
    

    # training happens here based on args.resume_from
    env = normalize(Client())

    if args.resume_from is not None:
        params = joblib.load(args.resume_from)
        policy = params['policy']
        baseline = params['baseline']
    else:
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(100, 50, 25)
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    ### woo hooo training happening soo fasttt.. ultra fasttttt
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.batch_size,
        max_path_length=args.max_path_length,
        n_itr=args.n_itr,
        discount=args.discount,
        step_size=args.step_size,
        threads=args.n_parallel
    )

    algo.train()

    logger.set_snapshot_mode(prev_mode)
    logger.set_snapshot_dir(prev_snapshot_dir)
    logger.set_snapshot_gap(prev_gap)
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()