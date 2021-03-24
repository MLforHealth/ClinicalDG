# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import uuid
import json
import os
from pathlib import Path


mnist_dir = '/scratch/ssd001/home/haoran/domainbed_data/MNIST/'

# MNIST #######################################################################

def download_mnist():
    # Original URL: http://yann.lecun.com/exdb/mnist/
    Path(mnist_dir).mkdir(exist_ok = True, parents = True)
    MNIST(mnist_dir, download=True)

if __name__ == "__main__":
    download_mnist()