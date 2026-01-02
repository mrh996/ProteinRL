from yacs.config import CfgNode as CN

_C = CN()

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.THERSHOLD = 8
_C.PROTEIN.BOXSIZE = 20
_C.PROTEIN.TEMP_RM = False
_C.PROTEIN.TEMP = '/LOCAL2/mur/MRH/protein_RL/prediction_model/pdb/mut_temp' 
_C.PROTEIN.WT_FOLDER = '/LOCAL2/mur/MRH/protein_RL/prediction_model/pdb/wt'

# DDG
_C.DDG_MODEL = CN()
_C.DDG_MODEL.DIR = '/LOCAL2/mur/MRH/protein_RL/prediction_model/model_largenew' #model_large(nerblock encoder), model_mega,model_transfer_large

# DIR
_C.DIR = CN()
# RL
_C.THERSHOLD=0.5
_C.STEPS=4



def get_cfg_defaults():
    return _C.clone()