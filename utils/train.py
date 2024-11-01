import numpy as np
import os
import torch
import shutil
import torch.nn as nn

from tqdm import tqdm

def train(args, accelerator, model, train_loader, criterion):
    

                