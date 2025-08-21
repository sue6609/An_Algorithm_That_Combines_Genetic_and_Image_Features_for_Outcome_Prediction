# %%
for name in dir():
    if not name.startswith('_'):
        del globals()[name]


# %%
# 套件
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.nn import init, Parameter
import numpy as np
import math
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from utils import mixed_collate


from PIL import Image

# %%
root = "data\\TCGA_GBMLGG\\"
pathway_mask = pd.read_csv(root+"pathway_mask.csv", index_col=0)
pathway_mask = torch.tensor(pathway_mask.values, dtype=torch.float32)
print(pathway_mask.shape)

# %%
# checkpoints_dir = './checkpoints/TCGA_GBMLGG'
# exp_name = '單影像'
# # del_name = 'path'
mode =  "omic"
# num_epochs = 50
# task = 'surv'
# gpu_ids = "0"
# verbose = 1
# lambda_cox  = 1
# print_every  = 10

# %%
# dataroot = './data/TCGA_GBMLGG'
# ignore_missing_moltype = 1
# ignore_missing_histype = 0
# use_vgg_features = 0
# use_rnaseq = 1
# use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if use_vgg_features else ('_', 'all_st')
# use_rnaseq = '_rnaseq' if use_rnaseq else ''

# %%
# data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, use_vgg_features, use_rnaseq)

data_cv_path = "data\\TCGA_GBMLGG\\splits\\0506_single_split_all_st.pkl"
# data_cv_path = "data\\TCGA_GBMLGG\\splits\\gbmlgg15cv_all_st_1_0_0_rnaseq.pkl"
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv.pop("data_pd")
data_cv.keys()  
# 隨機產一數1~15

# i = np.random.randint(1, 16)
# data_cv_splits = data_cv['cv_splits'][i]
# print(i)
# results = []

# %%
# device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
# print("Using device:", device)

# %%
class DatasetLoader():
    def __init__(self, data, split, mode):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
       
        self.mode = mode
        
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            # transforms.RandomCrop(512),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        

        if self.mode == "path" or self.mode == 'pathpath':
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            return (self.transforms(single_X_path), 0, single_e, single_t)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (single_X_omic, single_e, single_t)
        elif self.mode == "pathomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), single_X_omic, single_e, single_t)
        

    def __len__(self):
        return len(self.X_path)

# %%
# train_loader = torch.utils.data.DataLoader(dataset=DatasetLoader(data_cv, split='train', mode = mode),
#                                             batch_size=110 ,shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=DatasetLoader(data_cv, split='test', mode = mode),
#                                             batch_size=110 ,shuffle=True)
# train_loader = torch.utils.data.DataLoader(dataset=DatasetLoader(data_cv_splits, split='train', mode = mode),
#                                             batch_size=32 ,shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=DatasetLoader(data_cv_splits, split='test', mode = mode),
#                                             batch_size=32 ,shuffle=True)

# %%
from torch.utils.data import DataLoader, random_split

train_loader = DataLoader(DatasetLoader(data_cv, 'train', mode), batch_size = 8, shuffle=True,collate_fn=mixed_collate)
full_val_dataset = DatasetLoader(data_cv, 'test', mode)

val_size = int(0.5 * len(full_val_dataset))
test_size = len(full_val_dataset) - val_size

val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size])

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,collate_fn=mixed_collate)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,collate_fn=mixed_collate)

print("Train dataset size:", len(train_loader.dataset))
print("Validation dataset size:", len(val_loader.dataset))
print("Test dataset size:", len(test_loader.dataset))

# %%
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
print(f"Using {device} device")
# Define model
def init_max_weights(module):
  """
    初始化神經網路中所有 nn.Linear 層的權重和bias。

    - 權重：使用平均為 0，標準差為 1 / sqrt(輸入特徵數) 的高斯分布初始化，確保訓練時的數值穩定性。
    - bias：初始化為 0。

    Args:
        module (nn.Module): 需要進行初始化的神經網路模型或模組。

  """
  for m in module.modules():
    if type(m) == nn.Linear:
      stdv = 1. / math.sqrt(m.weight.size(1))
      m.weight.data.normal_(0, stdv)
      m.bias.data.zero_()


class SNN_Model(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate ):
        super(SNN_Model, self).__init__()
        hidden = [ 24,16]
        # hidden = [64,48,32]    

        # Part 1: gene data with Pathway
        # self.pathway_mask = pathway_mask.to(device)
        # self.fc1 = nn.Linear(input_size, 32)  # Gene -> Pathway

        encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        # encoder3 = nn.Sequential(
        #     nn.Linear(hidden[1], hidden[2]),
        #     nn.ELU(),
        #     nn.AlphaDropout(p=dropout_rate, inplace=False))
        # self.encoder = nn.Sequential(encoder1, encoder2, encoder3)
        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Dropout(0.7),nn.Linear(16, output_size))
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


        init_max_weights(self)

    def forward(self, gene_data):
        # Gene data forward pass with Pathway
        # self.fc1.weight.data = self.fc1.weight.data.mul(self.pathway_mask)  # Apply pathway mask
        # gene_data = self.fc1(gene_data)
        gene_data = self.encoder(gene_data)
        gene_data = self.classifier(gene_data)
       # gene_data = torch.sigmoid(gene_data)
        gene_data = gene_data * self.output_range + self.output_shift

        return gene_data
    
# class SNN_Model(nn.Module):
#     def __init__(self, input_size, output_size,dropout_rate ):
#         super(SNN_Model, self).__init__()
#         hidden = [48,32,16]
#         hidden = [64,48,32]


#         encoder1 = nn.Sequential(
#             nn.Linear(input_size,hidden[0]),
#             nn.ELU(),
#             nn.AlphaDropout(p=dropout_rate, inplace=False))

#         encoder2 = nn.Sequential(
#             nn.Linear(hidden[0], hidden[1]),
#             nn.ELU(),
#             nn.AlphaDropout(p=dropout_rate, inplace=False))

#         encoder3 = nn.Sequential(
#             nn.Linear(hidden[1], hidden[2]),
#             nn.ELU(),
#             nn.AlphaDropout(p=dropout_rate, inplace=False))
        
#         self.encoder = nn.Sequential(encoder1, encoder2, encoder3)
#         # self.encoder = nn.Sequential(encoder1, encoder2)



#         self.classifier = nn.Sequential(nn.Linear(32, output_size))
#         self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
#         self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


#         init_max_weights(self)

#     def forward(self, gene_data):
        
#         gene_data = self.encoder(gene_data)
#         gene_data = self.classifier(gene_data)
#         gene_data = torch.sigmoid(gene_data)
#         gene_data = gene_data * self.output_range + self.output_shift

#         return gene_data


# 單基因模態(SNN)
# class CombinedModel(nn.Module):
#     def __init__(self, input_size, output_size, pathway_mask, pathway_nodes,  dropout_rate):
#         super(CombinedModel, self).__init__()
#         self.model_a = SNN_Model(input_size, output_size,pathway_mask, pathway_nodes, dropout_rate)
#         # self.fc = nn.Sequential(nn.Linear(output_size, 1))

#     def forward(self, gene_data):
#         gene_output = self.model_a(gene_data)  # 輸入通過模型A
#         # gene_output = self.fc(gene_output)
#         return gene_output

class CombinedModel(nn.Module):
    def __init__(self, input_size, output_size,  dropout_rate):
        super(CombinedModel, self).__init__()
        self.model_a = SNN_Model(input_size, output_size, dropout_rate)
        # self.fc = nn.Sequential(nn.Linear(output_size, 1))
        

    def forward(self ,gene_data):
        gene_output = self.model_a(gene_data)  # 輸入通過模型A
        # gene_output = self.fc(gene_output)

        return gene_output
    

    
    
pathway_nodes = pathway_mask.shape[0]
# input_size = train_loader.dataset.X_omic.shape[1] # 基因數據輸入特徵數量
input_size = 4088
# input_size = 320
output_size = 1
dim_1 = output_size
dim_2 = output_size
output_dim = 1 # 輸出維度
dropout_rate = 0.5

#model = CombinedModel(input_size, output_size, dropout_rate).to(device)
model = CombinedModel(input_size, output_size, dropout_rate).to(device)
# model = CombinedModel(input_size, output_size, dim_1, dim_2, output_dim).to(device)
# model = CombinedModel(input_size, output_size, dim_1, dim_2, output_dim, pathway_nodes, pathway_mask).to(device)
print(model)

# %%
# 所有參數總數
total_params = sum(p.numel() for p in model.parameters())

# 可訓練參數總數（requires_grad=True 的）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"總參數量：{total_params:,}")
print(f"可訓練參數量：{trainable_params:,}")

# %%
def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

# %%
# from torch.optim.lr_scheduler import StepLR
# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-4)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.9, 0.999))
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# l1_lambda = 0

##圖片


# %%
from torch.optim.lr_scheduler import StepLR
optimizer = torch.optim.Adam(model.parameters())

# optimizer = torch.optim.Adam(model.parameters(),lr=5e-4,weight_decay=4e-4, betas=(0.9, 0.999))
# scheduler = StepLR(optimizer, step_size=25, gamma=0.2)

# l1_lambda = 2e-3

# optimizer = torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=4e-4, betas=(0.9, 0.999))
#Cscheduler = StepLR(optimizer, step_size=25, gamma=0.5)
# l1_lambda = 2e-4
l1_lambda = 0



# %%
def train(dataloader, model, CoxLoss, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    total_loss = 0.0  # 累積所有 batch 的 loss
    num_batches = 0   # 計算 batch 數量
    current = 0
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([]) 
    c_index = 0
   
    for batch, ( x_omic, e, t) in enumerate(dataloader):
        x_omic = x_omic.to(device)
        e = e.to(device)
        t = t.to(device)

        # optimizer.zero_grad()

        pred = model(x_omic)
        loss = CoxLoss(t, e, pred, device)

        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        

        batch_loss = loss.item()
        total_loss += batch_loss
        num_batches += 1
        current += len(x_omic)

        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
        censor_all = np.concatenate((censor_all, e.detach().cpu().numpy().reshape(-1)))   # Logging Information
        survtime_all = np.concatenate((survtime_all, t.detach().cpu().numpy().reshape(-1))) 


        if batch % 6 == 0:
            print(f"Batch: {batch}  loss: {batch_loss:>7f}  [{current:>5d}/{size:>5d}]")  

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    c_index = concordance_index(survtime_all, -np.array(risk_pred_all), censor_all)

    # scheduler.step() 
    
    return average_loss, c_index


def test(dataloader, model, CoxLoss):
  
    num_batches = len(dataloader)
    model.eval()
    test_loss, c_index = 0, 0
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([]) 
    with torch.no_grad():
        for  x_omic, e, t in dataloader:
           
            x_omic = x_omic.to(device)
            e = e.to(device)
            t = t.to(device)

            pred = model(x_omic)
            test_loss += CoxLoss(t,e, pred, device).item()


            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
            censor_all = np.concatenate((censor_all, e.detach().cpu().numpy().reshape(-1)))   # Logging Information
            survtime_all = np.concatenate((survtime_all, t.detach().cpu().numpy().reshape(-1))) 
          
# all_censorships, all_times) = np.array(all_censorships), np.array(all_times)

    test_loss /= num_batches
    c_index = concordance_index(survtime_all, -np.array(risk_pred_all), censor_all)
  
    
    print(f"Test Error: \n c-index: {(c_index):>0.2f}, Avg loss: {test_loss:>8f} \n")
    return test_loss, c_index
    


# %%

train_losses = []
test_losses = []
c_indices = []
train_cindexs = []


# %%
epochs = 50
best_loss = float('inf')
import copy

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss, train_cindex = train(train_loader, model, CoxLoss, optimizer)
    train_losses.append(loss)
    test_loss, c_index = test(test_loader, model, CoxLoss)
    test_losses.append(test_loss)
    c_indices.append(c_index)
    train_cindexs.append(train_cindex)
    if test_loss < best_loss:
        best_loss = test_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"Best model updated at epoch {t+1} with test loss {best_loss:.6f}")
print("Done!")

# %%
def evaluate_test_set(model, dataloader, CoxLoss, device):
    model.eval()
    all_risk_preds = []
    all_times = []
    all_censorships = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for x_omic, e, t in dataloader:
            x_omic, e, t =  x_omic.to(device), e.to(device), t.to(device)
            pred = model( x_omic)
            loss = CoxLoss(t, e, pred, device)
            total_loss += loss.item()
            num_batches += 1

            all_risk_preds.extend(pred.cpu().numpy())
            all_times.extend(t.cpu().numpy())
            all_censorships.extend(e.cpu().numpy())

    avg_loss = total_loss / num_batches
    c_index = concordance_index(all_times, -np.array(all_risk_preds), all_censorships)

    print(f"Test C-index: {c_index:.4f} | Avg Loss: {avg_loss:.4f}")
    return avg_loss, c_index


# %%
# 訓練結束後，載入最佳模型
model.load_state_dict(best_model_state)

avg_loss, c_index = evaluate_test_set(model, val_loader, CoxLoss, device)

# %%
print("max c-index: ", max(c_indices))

# %%
#設定style
plt.style.use('ggplot')
# 確保 losses 都是數值
train_losses = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_losses]
test_losses = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in test_losses]

fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))  # 一列兩圖
epochs = len(train_losses)
# 子圖 1：Loss
#加入節點

axs[0].plot(range(epochs), train_losses, label='Train Loss' , color='blue' )#加點
axs[0].plot(range(epochs), test_losses, label='Validation Loss', color='orange')
axs[0].set_title('Loss over epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')


axs[0].legend()
axs[0].grid(True)

# 子圖 2：C-index
axs[1].plot(range(epochs), c_indices, label='Validation C-index', color='red')
axs[1].plot(range(epochs), train_cindexs, label='Train C-index', color='green')
axs[1].set_title('C-index over epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('C-index')
axs[1].legend()
axs[1].grid(True)
axs[1].text(0.8, 0.1, f'Test loss: {avg_loss:.4f}\n Test c-index: {c_index:.4f}',
         fontsize=12, ha='center', va='center', transform=axs[1].transAxes,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))


# 加總標題
fig.suptitle(f"{epochs} epochs on SNN", fontsize=16)
plt.tight_layout()
# plt.savefig('plot/4088_SNN_50_no_path_3.png', dpi=300)  # 儲存圖片 
plt.show()

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
stop

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
#torch.save(model.state_dict(), './snn_model_2.pt')

# state_dict = torch.load('./snn_model_2.pt')   



