"""
Configuration file untuk GNN experiments
Ubah parameter di sini untuk eksperimen yang berbeda
"""

# ========== MODEL CONFIGURATION ==========
MODEL_CONFIG = {
    'model_type': 'GCN',  # Options: 'GCN', 'GAT', 'GraphSAGE'
    'hidden_channels': 16,
    'num_layers': 2,
    'dropout': 0.5,
}

# GAT specific config
GAT_CONFIG = {
    'heads': 8,  # Number of attention heads
    'attention_dropout': 0.6,
}

# GraphSAGE specific config
SAGE_CONFIG = {
    'aggr': 'mean',  # Options: 'mean', 'max', 'lstm'
}

# ========== TRAINING CONFIGURATION ==========
TRAINING_CONFIG = {
    'max_epochs': 200,
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
    'early_stopping_patience': 10,
    'log_interval': 10,
}

# ========== OPTIMIZER CONFIGURATION ==========
OPTIMIZER_CONFIG = {
    'optimizer': 'Adam',  # Options: 'Adam', 'SGD', 'AdamW'
    'momentum': 0.9,  # For SGD
    'betas': (0.9, 0.999),  # For Adam/AdamW
}

# ========== SCHEDULER CONFIGURATION ==========
SCHEDULER_CONFIG = {
    'use_scheduler': True,
    'scheduler_type': 'StepLR',  # Options: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
    'step_size': 50,  # For StepLR
    'gamma': 0.5,  # For StepLR
    'T_max': 200,  # For CosineAnnealingLR
    'patience': 10,  # For ReduceLROnPlateau
    'factor': 0.5,  # For ReduceLROnPlateau
}

# ========== DATASET CONFIGURATION ==========
DATASET_CONFIG = {
    'dataset_name': 'Cora',  # Options: 'Cora', 'CiteSeer', 'PubMed'
    'normalize_features': True,
    'dataset_path': '../dataset',
}

# ========== VISUALIZATION CONFIGURATION ==========
VIZ_CONFIG = {
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    'save_plots': True,
    'plot_dpi': 300,
    'output_dir': '../model',
}

# ========== DEVICE CONFIGURATION ==========
DEVICE_CONFIG = {
    'use_cuda': True,  # Set False to force CPU
    'cuda_device': 0,  # GPU device number
}

# ========== REPRODUCIBILITY ==========
SEED_CONFIG = {
    'use_seed': True,
    'seed': 42,
}

# ========== ADVANCED FEATURES ==========
ADVANCED_CONFIG = {
    'use_batch_norm': False,
    'use_residual': False,
    'use_layer_norm': False,
    'gradient_clipping': False,
    'max_grad_norm': 1.0,
}

# ========== EXPERIMENT TRACKING ==========
EXPERIMENT_CONFIG = {
    'experiment_name': 'gcn_cora_baseline',
    'use_tensorboard': False,
    'tensorboard_dir': 'runs',
    'save_embeddings': True,
}

# ========== CLASS NAMES ==========
CLASS_NAMES = {
    'Cora': [
        'Case_Based', 
        'Genetic_Algorithms', 
        'Neural_Networks',
        'Probabilistic_Methods', 
        'Reinforcement_Learning',
        'Rule_Learning', 
        'Theory'
    ],
    'CiteSeer': [
        'Agents', 
        'AI', 
        'DB',
        'IR', 
        'ML', 
        'HCI'
    ],
    'PubMed': [
        'Diabetes_Mellitus_Experimental',
        'Diabetes_Mellitus_Type_1',
        'Diabetes_Mellitus_Type_2'
    ]
}

# ========== HELPER FUNCTIONS ==========
def get_config():
    """Return all configurations as a dictionary"""
    return {
        'model': MODEL_CONFIG,
        'gat': GAT_CONFIG,
        'sage': SAGE_CONFIG,
        'training': TRAINING_CONFIG,
        'optimizer': OPTIMIZER_CONFIG,
        'scheduler': SCHEDULER_CONFIG,
        'dataset': DATASET_CONFIG,
        'visualization': VIZ_CONFIG,
        'device': DEVICE_CONFIG,
        'seed': SEED_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'class_names': CLASS_NAMES,
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("=" * 70)
    print("CURRENT CONFIGURATION")
    print("=" * 70)
    for section, params in config.items():
        if section != 'class_names':  # Skip class_names for brevity
            print(f"\n[{section.upper()}]")
            for key, value in params.items():
                print(f"  {key}: {value}")
    print("=" * 70)

if __name__ == '__main__':
    print_config()
