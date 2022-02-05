import torch

class RetailDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.num_classes = len(set(labels))
        self.transform = transform

    def __getitem__(self, idx):
        item = {key: val[idx].detach().clone() for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return 'RetailDataset'

    def __str__(self):
        return str({
            'data': self.data['pixel_values'].shape,
            'labels': self.labels.shape,
            'num_classes': self.num_classes,
            'num_samples': len(self.labels)
        })