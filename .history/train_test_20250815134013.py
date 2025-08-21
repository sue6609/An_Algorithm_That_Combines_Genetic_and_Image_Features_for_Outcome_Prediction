import numpy as np
from lifelines.utils import concordance_index
import torch

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    num_batches = 0
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])

    for batch, (x_omic, x_path, e, t) in enumerate(dataloader):
        x_omic, x_path, e, t = x_omic.to(device), x_path.to(device), e.to(device), t.to(device)
        pred = model(x_omic, x_path)
        loss = loss_fn(t, e, pred, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, e.cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, t.cpu().numpy().reshape(-1)))
    avg_loss = total_loss / num_batches
    c_index = concordance_index(survtime_all, -risk_pred_all, censor_all)
    return avg_loss, c_index

def test(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        for x_omic, e, t in dataloader:
            x_omic, e, t = x_omic.to(device), e.to(device), t.to(device)
            pred = model(x_omic)
            total_loss += loss_fn(t, e, pred, device).item()
            num_batches += 1
            risk_pred_all = np.concatenate((risk_pred_all, pred.cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, e.cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, t.cpu().numpy().reshape(-1)))
    avg_loss = total_loss / num_batches
    c_index = concordance_index(survtime_all, -risk_pred_all, censor_all)
    return avg_loss, c_index

def evaluate_test_set(model, dataloader, loss_fn, device):
    avg_loss, c_index = test(dataloader, model, loss_fn, device)
    print(f"Test C-index: {c_index:.4f} | Avg Loss: {avg_loss:.4f}")
    return avg_loss, c_index
