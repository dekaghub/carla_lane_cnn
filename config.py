import torch

class Config:
    # Data Configuration
    DATA_DIR = "dataset"
    TRAIN_DIR = f"{DATA_DIR}/train"
    TRAIN_LABEL_DIR = f"{DATA_DIR}/train_label"
    VAL_DIR = f"{DATA_DIR}/val"
    VAL_LABEL_DIR = f"{DATA_DIR}/val_label"
    
    # Model Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMAGE_SIZE = (1024, 512)
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-3
    
    # Training Configuration
    LOSS_FUNCTION = 'combined'  # Options: 'bce', 'dice', 'focal', 'combined'
    SAVE_DIR = 'checkpoints'
    
    # Visualization Configuration
    FIGSIZE = (20, 8)
