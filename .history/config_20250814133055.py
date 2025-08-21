import pickle
import torch
import pandas as pd

def get_default_config(args):
    class Config: pass
    cfg = Config()
    cfg.root = "data/TCGA_GBMLGG/"
    cfg.pathway_mask = torch.tensor(
        pd.read_csv(cfg.root+"pathway_mask.csv", index_col=0).values,
        dtype=torch.float32
    )
    data_cv_path = cfg.root + "splits/0506_single_split_all_st.pkl"
    cfg.data_cv = pickle.load(open(data_cv_path, 'rb'))
    cfg.data_cv.pop("data_pd")
    cfg.input_size = 4088
    cfg.output_size = 1
    return cfg
