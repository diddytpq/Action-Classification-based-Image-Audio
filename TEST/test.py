from dataloader_custom import AC_Data_Loader
from torch.utils.data import TensorDataset, DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())

data_path_x = "./dataset/train/features/"
data_loader = AC_Data_Loader(data_path_x)

# train_loader = DataLoader(dataset = data_loader, num_workers = 4, batch_size=1, shuffle=True)
train_loader = DataLoader(dataset = data_loader, batch_size=4, shuffle=True)


for i, (x_data, y_data) in enumerate(train_loader):
    print(i)
    print(x_data.shape)
    print(y_data)
