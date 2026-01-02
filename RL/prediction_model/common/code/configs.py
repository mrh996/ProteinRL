from yacs.config import CfgNode as CN

_C = CN()

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True
_C.PROTEIN.THERSHOLD = 8
_C.PROTEIN.IN_CHANNEL = 16
_C.PROTEIN.BOXSIZE = 20
_C.PROTEIN.IN_CHANNEL = 128
_C.PROTEIN.NUM_FEATURE_NODE = 33
_C.PROTEIN.NUM_FEATURE_EDGE = 1

# GNN
_C.GNN = CN()
_C.GNN.DIM = 128
_C.GNN.DIM2 = 128
_C.GNN.DEPTH = 3
_C.GNN.HEAD = 2
_C.GNN.OUT_CHANNEL = 1

# ENCODER DECODER
_C.MODEL = CN()
_C.MODEL.MLP_LAYER = 3
_C.MODEL.DROPOUT = 0.1
_C.MODEL.lr_decay = 0.5
_C.MODEL.decay_interval = 10
_C.MODEL.NUM_EPOCHS = 50
_C.MODEL.BATCH_SIZE = 2
_C.MODEL.LR = 0.001
_C.MODEL.WEIGHT_DECAY = 1e-4
_C.MODEL.SEED = 2048

# DIR
_C.DIR = CN()
_C.DIR.DATASET = '/mnt/scratch2/users/3057228/reinforcement/dataset/largescale/largescale.txt'
_C.DIR.GRAPH = '/mnt/scratch2/users/3057228/reinforcement/dataset/largescale/'
_C.DIR.OUTPUT_DIR = '/mnt/scratch2/users/3057228/reinforcement/dataset/largescale/transfer_2'
_C.DIR.SOURCE = '/mnt/scratch2/users/3057228/reinforcement/dataset/megascale/results_nodssp_2'



def get_cfg_defaults():
    return _C.clone()
