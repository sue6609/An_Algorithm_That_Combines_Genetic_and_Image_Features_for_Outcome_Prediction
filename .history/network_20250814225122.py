import torch
import torch.nn as nn
import math
from torch.nn import Parameter

def init_max_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


################
# SNN
################
class SNN_Model(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(SNN_Model, self).__init__()
        hidden = [24, 16]
        encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))
        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Dropout(0.7), nn.Linear(16, output_size))
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        init_max_weights(self)

    def forward(self, gene_data):
        gene_data = self.encoder(gene_data)
        gene_data = self.classifier(gene_data)
        return gene_data * self.output_range + self.output_shift
    


################
# PSNN
################
class PSNN_Model(nn.Module):
    def __init__(self, device ,input_size, output_size, pathway_mask, pathway_nodes ,dropout_rate ):
        super(PSNN_Model, self).__init__()
        hidden = [ 24,16]
        # hidden = [64,48,32]    

        # Part 1: gene data with Pathway
        self.pathway_mask = pathway_mask.to(device)
        self.fc1 = nn.Linear(input_size, pathway_nodes)  # Gene -> Pathway

        encoder1 = nn.Sequential(
            nn.Linear(pathway_nodes, hidden[0]),
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
        self.fc1.weight.data = self.fc1.weight.data.mul(self.pathway_mask)  # Apply pathway mask
        gene_data = self.fc1(gene_data)
        gene_data = self.encoder(gene_data)
        gene_data = self.classifier(gene_data)
       # gene_data = torch.sigmoid(gene_data)
        gene_data = gene_data * self.output_range + self.output_shift

        return gene_data
    


################
# CNN
################    
class CNN(nn.Module):
    def __init__(self, output_size,):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(64 * 128 * 128, output_size)  # Assuming input image size is 1024x1024
        self.relu = nn.ReLU() # Define ReLU as an attribute
        init_max_weights(self)

    def forward(self, images_data):

        # Image data forward pass
        images_data = self.pool(self.relu(self.conv1(images_data))) # Use self.relu here
        images_data = self.pool(self.relu(self.conv2(images_data))) # Use self.relu here
        images_data = self.pool(self.relu(self.conv3(images_data))) # Use self.relu here
        images_data = images_data.view(-1, 64 * 128 * 128)  # Flatten the tensor
        images_data = self.fc(images_data)  # CNN output size

        return images_data
    
    
class CNN_Model(nn.Module):
    def __init__(self, output_size, dropout_rate):
        super(CNN_Model, self).__init__()

        self.features = nn.Sequential(
          
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
         

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,32 ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_size)  
        )
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)    
        

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        x = torch.sigmoid(x) 
        x = x * self.output_range + self.output_shift
        return x



################
# FUSED_MODEL
################
class GatedMultimodalUnit(nn.Module):
    def __init__(self, dim_1, dim_2, output_dim):
        super(GatedMultimodalUnit, self).__init__()

        self.fc_modality1 = nn.Sequential(nn.Linear(dim_1, dim_1), nn.ReLU())  # 模態1的轉換
        self.fc_modality2 = nn.Sequential(nn.Linear(dim_2, dim_2), nn.ReLU())  # 模態2的轉換

        self.linear_z1 = nn.Bilinear(dim_1, dim_2, dim_1)  # 基於兩個模態的聯合
        self.linear_z2 = nn.Bilinear(dim_1, dim_2, dim_2)  # 基於兩個模態的聯合

        self.linear_o1 = nn.Sequential(nn.Linear(dim_1, dim_1), nn.ReLU())
        self.linear_o2 = nn.Sequential(nn.Linear(dim_2, dim_2), nn.ReLU())

        self.post_fusion_dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d((dim_1 + 1) * (dim_2 + 1))
        self.encoder1 = nn.Sequential(nn.Linear((dim_1 + 1) * (dim_2 + 1), output_dim), nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        init_max_weights(self)

    def forward(self, modality1, modality2):
        # 分別轉換兩個模態
        modality1 = torch.nn.functional.normalize(modality1, p=2, dim=1)
        h1 = self.fc_modality1(modality1)

        modality2 = torch.nn.functional.normalize(modality2, p=2, dim=1)
        h2 = self.fc_modality2(modality2)

        # 計算 z1 和 z2
        z1 = self.sigmoid(self.linear_z1(modality1, modality2))  # 基於兩個模態的聯合信息
        z2 = self.sigmoid(self.linear_z2(modality1, modality2))  # 基於兩個模態的聯合信息

        # 計算 o1 和 o2
        o1 = self.linear_o1(z1 * h1)  # 使用 z1 的權重
        o2 = self.linear_o2(z2 * h2)  # 使用 z2 的權重

        # Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)  # 添加一行值全為 1
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)  # 添加一行值全為 1
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)  # 外積

        # 輸出處理
        out = self.bn1(o12)
        out = self.post_fusion_dropout(out)
        out = self.encoder1(out)  # 使用全連接層進行處理

        return out
    
class CNN_SNN_Model(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, p): # 帶改
        super(CNN_SNN_Model, self).__init__()

        hidden = [24,16]

        encoder1 = nn.Sequential(
            nn.Linear(input_size, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))


        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Linear(16, output_size))

        init_max_weights(self)

        # Part 2: Image data (CNN)
        self.features = nn.Sequential(
          
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
         

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.classifier_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,32 ),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, output_size)  
        )


    def forward(self, gene_data, x):
     
        gene_data = self.encoder(gene_data)
        gene_data = self.classifier(gene_data)

        # Image data forward pass
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier_2(x)

        return gene_data, x
    
class FUSED_Model(nn.Module):
    def __init__(self, input_size, output_size, dim_1, dim_2, output_dim, dropout_rate, p):
        super(FUSED_Model, self).__init__()
        self.model_a = CNN_SNN_Model(input_size, output_size, dropout_rate,p)  # 模型A
        self.model_b = GatedMultimodalUnit(dim_1, dim_2, output_dim)  # 模型B

    def forward(self, gene_data, images_data):
        gene_output, image_output = self.model_a(gene_data, images_data)  # 輸入通過模型A
        x_b = self.model_b(gene_output, image_output)  # 模型A的輸出進入模型B
        return x_b



class CNN_PSNN_Model(nn.Module):
    def __init__(self, ㄝinput_size, output_size, pathway_nodes, pathway_mask, dropout_rate, p): # 帶改
        super(CNN_PSNN_Model, self).__init__()

        hidden = [24,16]

        # Part 1: gene data with Pathway
        self.pathway_mask = pathway_mask.to(device)

        self.fc1 = nn.Linear(input_size, pathway_nodes)  # Gene -> Pathway

        encoder1 = nn.Sequential(
            nn.Linear(pathway_nodes, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))


        self.encoder = nn.Sequential(encoder1, encoder2)
        self.classifier = nn.Sequential(nn.Linear(16, output_size))

        init_max_weights(self)

        # Part 2: Image data (CNN)
        self.features = nn.Sequential(
          
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
         

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.classifier_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,32 ),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32, output_size)  
        )


    def forward(self, gene_data, x):
        # Gene data forward pass with Pathway
        self.fc1.weight.data = self.fc1.weight.data.mul(self.pathway_mask)  # Apply pathway mask
        gene_data = self.fc1(gene_data)
        gene_data = self.encoder(gene_data)
        gene_data = self.classifier(gene_data)


        # Image data forward pass
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier_2(x)

        return gene_data, x
    
class PFUSED_Model(nn.Module):
    def __init__(self, input_size, output_size, dim_1, dim_2, output_dim, pathway_nodes, pathway_mask, dropout_rate, p):
        super(PFUSED_Model, self).__init__()
        self.model_a = CNN_PSNN_Model(input_size, output_size, pathway_nodes, pathway_mask, dropout_rate,p)  # 模型A
        self.model_b = GatedMultimodalUnit(dim_1, dim_2, output_dim)  # 模型B

    def forward(self, gene_data, images_data):
        gene_output, image_output = self.model_a(gene_data, images_data)  # 輸入通過模型A
        x_b = self.model_b(gene_output, image_output)  # 模型A的輸出進入模型B
        return x_b    
def get_model(name, args):
    name = name.lower()
    if name == "snn":
        return SNN_Model(input_size = args.input_size, output_size = args.output_size, dropout_rate = args.dropout).to(args.device)
    elif name =="psnn":
        return PSNN_Model(device = args.device ,input_size = args.input_size, output_size = args.output_size, pathway_mask = args.pathway_mask, pathway_nodes = args.pathway_mask.shape[0], dropout_rate = args.dropout).to(args.device)
    elif name == "cnn":
        return CNN_Model(output_size = args.output_size, dropout_rate = args.dropout).to(args.device)
    elif name == "fused":
        return FUSED_Model(
            input_size=args.input_size,
            output_size=args.output_size, #記得設大於output_dim，原設定16
            dim_1=args.dim_1,
            dim_2=args.dim_2,
            output_dim=args.output_dim,
            dropout_rate=args.dropout,
            p=args.dropout
        ).to(args.device)
    elif name == "pfused":
        return PFUSED_Model(
            input_size=args.input_size,
            output_size=args.output_size,
            dim_1=args.dim_1,
            dim_2=args.dim_2,
            output_dim=args.output_dim,
            pathway_nodes=args.pathway_mask.shape[0],
            pathway_mask=args.pathway_mask,
            dropout_rate=args.dropout,
            p=args.dropout
        ).to(args.device)
    else:
        raise ValueError(f"Unknown model name: {name}")

