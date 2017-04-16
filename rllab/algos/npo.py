from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from multiprocessing import Pool
from rllab.envs.normalized_env import normalize
from osim_http_client import Client
from rllab.sampler.utils import rollout
from itertools import chain
import numpy as np
import time
from osim_helpers import start_env_server
import psutil

class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            threads=1,
            first_env_server=None,
            ec2=False,
            **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.threads = threads
        self.ec2 = ec2
        super(NPO, self).__init__(**kwargs)

        print("Creating multiple env threads")
        self.env_servers = [first_env_server] + list(map(lambda x: start_env_server(x, self.ec2), range(1, threads)))
        time.sleep(10)

        self.parallel_envs = [self.env] #+ list(map(lambda x: normalize(Client(x)), range(1, threads)))
        for i in range(1, threads):
            while True:
                try:
                    temp_env = normalize(Client(i))
                    self.parallel_envs.append(temp_env)
                    break
                except Exception:
                    print("Exception while creating env of port ", i)
                    print("Trying to create env again")



    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            # env=self.env,
        )

    @overrides
    def get_paths(self, itr):
        p = Pool(self.threads)
        parallel_paths = p.map(self.get_paths_from_env, enumerate(self.parallel_envs))
        for job in parallel_paths:
            print(list(map(lambda x: tot_reward(x), job)))
            print(list(map(lambda x: len_path(x), job)))
        print(len(parallel_paths), 'no of parallel jobs')
        print(len(parallel_paths[0]), 'no of paths in first job')
        print(len(parallel_paths[0][0]['rewards']), 'no of rewards in first path of first job')
        print(np.sum(parallel_paths[0][0]['rewards']), 'total reward')
        p.close()
        p.join()
        all_paths = list(chain(*parallel_paths)) 
        print(len(all_paths), 'total no of paths in this itr')
        return all_paths
    
    def get_paths_from_env(self, thread_env):
        # num_episodes_per_thread = self.batch_size // self.max_path_length // self.threads
        # print(num_episodes_per_thread, 'no of episodes in this thread')
        # ext.set_seed(thread_env[0])
        # paths = [rollout(thread_env[1], self.policy, self.max_path_length)\
        #     for x in range(num_episodes_per_thread)]
        thread_batch_size = self.batch_size // self.threads
        paths = []
        count = 0
        episodes = 0
        ext.set_seed(thread_env[0])
        while count < thread_batch_size:
            cur_path = rollout(thread_env[1], self.policy, self.max_path_length)
            count += len(cur_path['rewards'])
            episodes += 1
            paths.append(cur_path)
        print(episodes, 'no of episodes in this thread')
        print([len(x['rewards']) for x in paths], 'episode lenghths in this thread')
        return paths

    @overrides
    def destroy_envs(self):
        print("Destroying env servers")
        for pid in self.env_servers:
            try:
                process = psutil.Process(pid)
                process.terminate()
            except:
                print("process doesnt exist", pid)
                pass
        pass

    @overrides
    def create_envs(self):
        print("Creating new env servers")
        self.env_servers = list(map(lambda x: start_env_server(x, self.ec2), range(0, self.threads)))
        time.sleep(10)
        print("Creating new envs")
        self.parallel_envs = []
        for i in range(0, self.threads):
            while True:
                try:
                    temp_env = normalize(Client(i))
                    self.parallel_envs.append(temp_env)
                    break
                except Exception:
                    print("Exception while creating env of port ", i)
                    print("Trying to create env again")
        pass

def tot_reward(path):
    return np.sum(path['rewards'])

def len_path(path):
    return len(path['rewards'])
