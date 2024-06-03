# Import necessary libraries and modules
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch as t

# Check for GPU availability and set the device accordingly
device = "cuda" if t.cuda.is_available() else "cpu"

# Class for taking the dataframe and creating dataloaders to feed into model
class DataPrep(Dataset):
    def __init__(self, df, test_size, textprocessing, cfg):
        self.df = df
        self.test_size = test_size
        self.textprocess = textprocessing
        self.cfg = cfg

    # Function for returning a dataloaders dependent on whether its for training or inference time
    def prep(self, data, train):

        data = self.textprocess.padding_and_tokenizing(data)
        dataset = DataSet(data, self.cfg)

        if train:
            return DataLoader(dataset, self.cfg.batch_size, shuffle=True)
        else:
            return DataLoader(dataset)

    # Stratified splitting of data into test and train
    def train_test(self, combined):

        train, test = train_test_split(
            self.df, test_size=self.test_size, stratify=self.df.label, random_state=1
        )

        return self.prep(data=train, train=True), self.prep(data=test, train=False)


# Defining a custom Dataset class
class DataSet(Dataset):
    def __init__(self, mode, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_s = t.tensor(mode)

    def __getitem__(self, idx):
        data = self.data_s[idx]
        # The last element is the target
        seq = data[:-1]
        targets = t.zeros(self.cfg.num_classes, device=device)
        targets[data[-1]] = 1
        return seq, targets

    def __len__(self):
        return len(self.data_s)
