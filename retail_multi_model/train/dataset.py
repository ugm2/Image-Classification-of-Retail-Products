import torch

class FreiburgGroceriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.num_classes = len(set(labels))

    def __getitem__(self, idx):
        item = {key: val[idx].detach().clone() for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        print(item)
        return item

    def __len__(self):
        return len(self.data)
