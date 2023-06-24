from dataloader_custom import AC_Data_Loader
from torch.utils.data import TensorDataset, DataLoader
import torch
from custom_model import Img_Audio_Feature_Extraction, Action_Classification_Model
import argparse
import time
import numpy as np

def FocalLoss(y_pred, y_true):
    eps = 1e-7
    loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss)

def mse_loss_fn(output, target):
        return torch.mean((output-target)**2)

def train(model, train_loader, epoch, loss_fn, batch_size):
    model.train()
    train_loss = 0

    t0 = time.time()

    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()

        data_rgb = data[0].type(torch.FloatTensor).to(device)
        data_flow = data[1].type(torch.FloatTensor).to(device)
        data_audio = data[2].type(torch.FloatTensor).to(device)

        data = [data_rgb, data_flow, data_audio]

        label = label.type(torch.FloatTensor).to(device)

        y_pred = model(data)
        
        # print(y_pred, label)

        # loss = FocalLoss(y_pred, label)
        
        loss = loss_fn(y_pred, label.view(batch_size,-1))

        # loss = mse_loss_fn(y_pred, label.view(4,-1))


        train_loss += loss.data
        loss.backward()
        optimizer.step()

        t1 = time.time()
        print('Train Epoch" {} [{}/{} ({:.0f}%)]'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100.0 * (batch_idx+1) / len(train_loader))+'\tLoss :',format(float(loss.data.cpu().numpy()),'.1E'),"\t time : ",(t1 - t0),"sec")
        t0 = time.time()

    train_loss /= len(train_loader)

    return model, train_loss

parser = argparse.ArgumentParser(description='Action Classification')

parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--optimizer', type=str, default='Adadelta', help='optimizer')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--epochs', type=int, default=100, help='epochs')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())


data_path_x_normal = "./dataset/train/features/normal/"
data_path_x_abnormal = "./dataset/train/features/abnormal/"

batch_size = 32

data_loader = AC_Data_Loader(data_path_x_normal, data_path_x_abnormal)
train_loader = DataLoader(dataset = data_loader, batch_size = batch_size, shuffle=True, drop_last=True)

model = Action_Classification_Model(device).to(device)


if args.optimizer == 'Adadelta':
    # args.lr = 0.8  
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)

elif args.optimizer == 'SGD':
    lr = 0.0001
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)

train_loss = []

# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.BCELoss()
# loss_fn = torch.nn.L1Loss()


for epoch in range(1, args.epochs + 1):
    model, loss = train(model, train_loader, epoch, loss_fn, batch_size)
    train_loss.append(loss.cpu().numpy())
    scheduler.step()

print(np.mean(train_loss))

torch.save(model.state_dict(), f"./models/custom/e{str(args.epochs)}_{str(args.optimizer)}_{str(np.mean(train_loss))}.pt")

# model.eval()
# with torch.no_grad():
#     for data, label in data_loader:
#         data = data
    
#         # y_pred = model(torch.unsqueeze(data, dim = 0))
#         y_pred = model([torch.unsqueeze(data[0], dim = 0), torch.unsqueeze(data[1], dim = 0), torch.unsqueeze(data[2], dim = 0)])

#         print(label , y_pred)

# show(train_loss)