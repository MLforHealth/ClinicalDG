# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid
import tqdm

import numpy as np
import torch

from clinicaldg import datasets
from clinicaldg import hparams_registry
from clinicaldg import algorithms
from clinicaldg.lib import misc
from clinicaldg import command_launchers

import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir, slurm_pre, delete_model = False):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'clinicaldg.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        if delete_model:
            command.append('--delete_model')
        self.command_str = ' '.join(command)
        if slurm_pre is not None:
            self.command_str = f'sbatch {slurm_pre} --wrap "{self.command_str}"'

        print(self.command_str)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
                    self.train_args['algorithm'],
                    self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def all_test_env_combinations(n):
    """For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs."""
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]


def make_args_list(n_trials, dataset_names, algorithms, n_hparams, steps, hparams, es_method):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                for hparams_seed in range(n_hparams):
                    train_args = {}
                    train_args['dataset'] = dataset
                    train_args['algorithm'] = algorithm
                    train_args['hparams_seed'] = hparams_seed
                    train_args['trial_seed'] = trial_seed
                    train_args['seed'] = misc.seed_hash(dataset,
                                                        algorithm, hparams_seed, trial_seed)
                    train_args['es_method'] = es_method
                    if steps is not None:
                        train_args['steps'] = steps
                    if hparams is not None:
                        train_args['hparams'] = hparams
                    args_list.append(train_args)
    return args_list


def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=[
                        'launch', 'delete_incomplete', 'delete_all'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--es_method', choices=['train', 'val', 'test'])
    parser.add_argument('--algorithms', nargs='+', type=str,
                        default=algorithms.ALGORITHMS)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--delete_model', action = 'store_true', 
        help = 'delete model weights after training to save disk space')
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--slurm_pre', type=str, required=False)
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams=args.n_hparams,
        steps=args.steps,
        hparams=args.hparams,
        es_method=args.es_method
    )

    jobs = [Job(train_args, args.output_dir, args.slurm_pre, args.delete_model)
            for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state in [
            Job.NOT_LAUNCHED, job.INCOMPLETE]]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)

    elif args.command == 'delete_all':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE or j.state == job.DONE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
