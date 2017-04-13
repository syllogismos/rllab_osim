
from osim_grader_client import Client
from collections import defaultdict
from tqdm import tqdm
import joblib
import argparse
import numpy as np

# get pickle location, and submit true or false

remote_base = 'http://grader.crowdai.org'
local_base = 'http://127.0.0.1:5000'
token = 'a6e5f414845fafd1063253a11429c78f'



policy = 'bla bla'


def submit_policy(policy):
    infos = defaultdict(list)
    client = Client(remote_base)
    ob = client.env_create(token)
    tot_rew = 0.0
    with tqdm(total=2500) as reward_bar:
        for i in tqdm(range(501)):
            a, _info = policy.get_action(ob)
            ob, rew, done, info = client.env_step(a.tolist(), True)
            tot_rew += rew
            rew_new = max([0, rew])
            reward_bar.update(rew_new)
            if done:
                print("terminated after %s timesteps"%i)
                break
            for k, v in info.items():
                infos[k].append(v)
            infos['ob'].append(ob)
            infos['reward'].append(rew)
            infos['action'].append(a)
    print("Total reward", tot_rew)
    x = input("type yes to submit to the server")
    if x.strip() == 'yes':
        client.submit()
    return infos, tot_rew

def local_run(policy):
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policyfile")
    parser.add_argument("--timestep_limit",type=int)
    parser.add_argument("--submit", type=int, default=0)
    args = parser.parse_args()

    params = joblib.load(args.policyfile)

    policy = params['policy']

    while True:
        if args.submit == 1:
            infos, tot_rew = submit_policy(policy)
        else:
            infos, tot_rew = local_run(policy)
        # for (k,v) in infos.items():
        #     if k.startswith("reward"):
        #         print(k, np.sum(v))
        # print("Total reward", tot_rew)
        input("press enter to continue")

if __name__ == '__main__':
    main()