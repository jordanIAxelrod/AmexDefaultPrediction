"""
 config some variables for the analysis
"""
import torch

categorical_vars = {'B_30': [], 'B_38': [], 'D_114': [], 'D_116': [], 'D_117': [], 'D_120': [], 'D_126': [], 'D_63': [],
                    'D_64': [], 'D_66': [], 'D_68': []}
non_features = ['target', 'S_2', 'customer_ID']
batch_size = 50
in_features = 223
device = "cuda" if torch.cuda.is_available() else "cpu"

