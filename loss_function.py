import torch
import numpy as np

def CoxLoss(survtime, censor, hazard_pred, device):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
