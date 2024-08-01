import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as T


class SignalDataset(data.Dataset):
    def __init__(self, npy_file, mode='train', segment_length=32):
        self.mode = mode
        self.segment_length = segment_length

        # Load the data
        self.data = np.load(npy_file)

        self.total_segments = self.data.shape[0] // self.segment_length
        print(f"Total segments: {self.total_segments}")

    def __getitem__(self, index):
        if self.mode == 'train' or self.mode == 'valid':
            # Randomly select an index for training or validation
            start_idx = random.randint(0, self.data.shape[0] - self.segment_length)
        else:
            # Sequentially select an index for testing
            start_idx = index * self.segment_length
            if start_idx + self.segment_length > self.data.shape[0]:
                start_idx = self.data.shape[0] - self.segment_length

        end_idx = start_idx + self.segment_length

        segment = self.data[start_idx:end_idx, :]

        # Split the segment into input and output
        input_data = segment[:, :3]
        label_data = segment[:, 3:]

        # Convert to tensors
        input_tensor = T.ToTensor()(input_data)
        label_tensor = T.ToTensor()(label_data)

        if self.mode == 'train' or self.mode == 'valid':
            return input_tensor, label_tensor
        else:
            return input_tensor, label_tensor  # Assuming you want labels for test as well

    def __len__(self):
        if self.mode == 'train' or self.mode == 'valid':
            return self.data.shape[0]
        else:
            return self.total_segments


def get_loader(npy_file, batch_size, shuffle=True, num_workers=0, mode='train'):
    dataset = SignalDataset(npy_file=npy_file, mode=mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    return data_loader
