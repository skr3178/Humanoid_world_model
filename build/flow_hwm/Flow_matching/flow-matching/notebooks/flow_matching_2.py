import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(''))))

from src.flow_matching import FlowMatching
from src.utils import generate_dynamic_conditioning
from src.visualization import visualize_flow_matching

# Initialize model and data
model = FlowMatching()
target_data = model.generate_two_circles_targets()

# Visualize the initial vector field
visualize_flow_matching(model, target_data, step=0)