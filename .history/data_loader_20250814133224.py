import torch
from torchvision import transforms
from torch_geometric.data import Batch
from torch.utils.data import default_collate
from PIL import Image

class DatasetLoader():
    def __init__(self, data, split, mode):
        self.X_path = data[split]['x_path']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.mode = mode
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).float()
        single_t = torch.tensor(self.t[index]).float()
        if self.mode in ["path", "pathpath"]:
            img = Image.open(self.X_path[index]).convert('RGB')
            return (self.transforms(img), 0, single_e, single_t)
        elif self.mode in ["omic", "omicomic"]:
            return (torch.tensor(self.X_omic[index]).float(), single_e, single_t)
        elif self.mode == "pathomic":
            img = Image.open(self.X_path[index]).convert('RGB')
            return (self.transforms(img), torch.tensor(self.X_omic[index]).float(), single_e, single_t)

    def __len__(self):
        return len(self.X_path)

def mixed_collate(batch):
    transposed = zip(*batch)
    return [Batch.from_data_list(samples, []) if isinstance(samples[0], torch_geometric.data.Data) 
            else default_collate(samples) for samples in transposed]
