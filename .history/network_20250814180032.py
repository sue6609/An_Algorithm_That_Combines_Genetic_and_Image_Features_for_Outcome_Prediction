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
# Regularization
################

def get_model(name, args):
    name = name.lower()
    if name == "snn":
        return SNN_Model(input_size = args.input_size, output_size = args.output_size, dropout_rate = args.dropout).to(args.device)
    elif name =="psnn":
        return PSNN_Model(device = args.device ,input_size = args.input_size, output_size = args.output_size, pathway_mask = args.pathway_mask, pathway_nodes = args.pathway_mask.shape[0], dropout_rate = args.dropout).to(args.device)
    elif name == "cnn":
        return CNN_Model(output_size = args.output_size, dropout_rate = args.dropout).to(args.device)
    elif name == "fused":
        return FUSED_Model(**kwargs)
    elif name == "pfused":
        return PFUSED_Model(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")

