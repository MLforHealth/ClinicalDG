# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import sys
import unicodedata
import getpass
import time

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """Doesn't run anything; instead, prints each command.
    Useful for testing."""
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')
                
def slurm_launcher(commands):
    MAX_SLURM_JOBS = 400
    for cmd in commands:
        block_until_running(MAX_SLURM_JOBS, getpass.getuser())
        subprocess.call(cmd, shell=True)     

def get_num_jobs(user):
    # returns a list of (# queued and waiting, # running)
    out = subprocess.run(['squeue -u ' + user], shell = True, stdout = subprocess.PIPE).stdout.decode(sys.stdout.encoding)
    a = list(filter(lambda x: len(x) > 0, map(lambda x: x.split(), out.split('\n'))))
    queued, running = 0,0
    for i in a:
        if i[0].isnumeric():
            if i[4].strip() == 'PD':
                queued += 1
            else:
                running += 1
    return (queued, running)

def block_until_running(n, user):
    while True:
        if sum(get_num_jobs(user)) < n:
            time.sleep(0.2)
            return True
        else:
            time.sleep(10)        
        
REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'slurm': slurm_launcher
}

