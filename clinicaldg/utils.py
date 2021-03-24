import getpass
import os
import torch
from pathlib import Path

class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, step, state_dict, path):  # lower loss is better
        score = -val_loss 

        if self.best_score is None:
            self.best_score = score
            self.step = step
            save_model(state_dict, path)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            save_model(state_dict, path)
            self.best_score = score
            self.step = step
            self.counter = 0
    
def save_model(state_dict, path):
    torch.save(state_dict, path)              

# functions for checkpoint/reload in case of job pre-emption on our slurm cluster
# will have to customize if you desire this functionality
# otherwise, the training script will still work fine as-is
def save_checkpoint(model, optimizer, sampler_dicts, start_step, es, rng):   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')        
    
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/').exists():        
        torch.save({'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'sampler_dicts': sampler_dicts,
                    'start_step': start_step,
                    'es': es,
                    'rng': rng
        } 
                   , 
                   Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').open('wb')                  
                  )
        
        
def has_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').exists():
        return True
    return False           

def load_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    fname = Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    if slurm_job_id is not None and fname.exists():
        return torch.load(fname)       
