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


def get_model(name, **kwargs):
    name = name.lower()
    if name == "snn":
        return SNN_Model(**kwargs)
    elif name =="psnn":
        return PSNN_Model(device ,input_size, output_size, pathway_mask, pathway_nodes ,dropout_rate)
    elif name == "cnn":
        return CNNModel(**kwargs)
    elif name == "fused":
        return FUSED_Model(**kwargs)
    elif name == "pfused":
        return PFUSED_Model(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")

