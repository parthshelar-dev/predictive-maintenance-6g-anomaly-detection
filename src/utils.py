"""
Utility functions for the 6G Predictive Maintenance and Anomaly Detection project.
"""

import os
import random
import numpy as np
import nbformat
from nbclient import NotebookClient

def set_seed(seed=42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def execute_notebook(notebook_path, output_path=None, timeout=600):
    """
    Executes a jupyter notebook top-to-bottom and saves it with all outputs.
    
    Args:
        notebook_path: Path to the input .ipynb file
        output_path: Path to save the executed notebook (defaults to overwriting input)
        timeout: Maximum seconds per cell
    """
    if output_path is None:
        output_path = notebook_path
        
    print(f"Executing {notebook_path} ...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name='python3',
        resources={'metadata': {'path': os.path.dirname(os.path.abspath(notebook_path))}}
    )
    client.execute()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print(f"Successfully executed and saved to {output_path}")
    return nb
