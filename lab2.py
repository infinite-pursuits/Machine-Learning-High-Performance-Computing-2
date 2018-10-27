import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropconv = nn.Dropout2d()
        self.l1 = nn.Linear(2304, 256)
        self.dropfc = nn.Dropout()
        self.l2 = nn.Linear(256, 17)

    def forward(self, image):

        conv1out = self.conv1(image)
        conv1out = F.relu(conv1out)
        conv1out = F.max_pool2d(conv1out, kernel_size=2)

        conv2out = self.conv2(conv1out)
        conv2out = F.relu(conv2out)
        dropped = self.dropconv(conv2out)
        conv2out = F.max_pool2d(conv2out, kernel_size=2)

        fc_in = conv2out.view(-1, 2304)
        fc1 = self.l1(fc_in)
        fc1 = F.relu(fc1)
        fc1 = self.dropfc(fc1)

        fc2 = self.l2(fc1)
        fc2 = F.sigmoid(fc2)
        return fc2


class MyDataset(Dataset):
    def __init__(self, size, images, tags, data_dir, transform=None):
        super(MyDataset, self).__init__()
        self.size = size
        self.images = images
        self.tags = tags
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        label = [0]*17
        for i in range(len(self.tags[index])):
            label[self.tags[index][i]] = 1
        image_name = self.images[index]
        image_path = self.data_dir + '/train-jpg/' + image_name +'.jpg'
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        image_rgb_transformed = self.transform(image_rgb)
        label = torch.FloatTensor(label)
        return image_rgb_transformed, label


def precision(k, sigs, target, batchsize):
    topk = sigs.detach().topk(k)[1]
    precision_sum = 0.0
    for i in range(batchsize):
        true_labels = 0
        for j in range(k):
            if int(target[i][int(topk[i][j])]) == 1:
                true_labels += 1
        precision_sum += (float(true_labels)/float(k))
    return precision_sum/float(batchsize)


def train(model, loss_criterion, optimizer, data_loader, epochs, device):
    model.train()
    time_epoch = []
    load_batch = []
    batch_comp = []
    for epoch in range(0, epochs):
        time_epoch_start = time.monotonic()
        running_loss_epoch = 0.0
        running_loss_mean_epoch_list = []
        precision_sum_epoch_1 = 0.0
        precision_sum_epoch_3 = 0.0

        batch_comp_start = time.monotonic()
        load_start = time.monotonic()
        for batch_no, (imagergb, target) in enumerate(data_loader):
            load_end = time.monotonic()
            imagergb = imagergb.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            sigs = model(imagergb)
            loss = loss_criterion(sigs, target)
            running_loss_epoch += loss.item()
            loss.backward()
            """if batch_no % 10 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.4f}'.format(
                    epoch, batch_no * len(target), len(data_loader.dataset), loss.item()), flush=True)"""
            optimizer.step()
            batch_comp_end = time.monotonic()
            precision_sum_epoch_1 += precision(1, sigs, target, len(target))
            precision_sum_epoch_3 += precision(3, sigs, target, len(target))
            load_batch.append(load_end - load_start)
            batch_comp.append(batch_comp_end - batch_comp_start)
            batch_comp_start = time.monotonic()
            load_start = time.monotonic()
        time_epoch_end = time.monotonic()
        time_epoch.append(time_epoch_end - time_epoch_start)
        running_loss_epoch_mean = running_loss_epoch/float(120)
        precision_mean_1 = precision_sum_epoch_1 / float(120)
        precision_mean_3 = precision_sum_epoch_3 / float(120)
        running_loss_mean_epoch_list.append(running_loss_epoch_mean)
        print("Epoch {}:\n Loss Mean: {:.4f}\n Precision@1: {:.3f}\n Precision@3: {:.3f}".format(epoch, running_loss_epoch_mean, precision_mean_1, precision_mean_3), flush=True)

    time_epoch_sum = float(sum(time_epoch))
    time_epoch = time_epoch_sum/float(len(time_epoch))
    load_batch_sum = float(sum(load_batch))
    load_batch = load_batch_sum/float(len(load_batch))
    batch_comp_sum = float(sum(batch_comp))
    batch_comp = batch_comp_sum /float(len(batch_comp))
    epoch_loss_mean = float(sum(running_loss_mean_epoch_list))/float(len(running_loss_mean_epoch_list))

    print("Average Loss: {}".format(epoch_loss_mean), flush=True)
    print("Total Time for Epochs: {} secs".format(time_epoch_sum), flush=True)
    print("Total Time for data loading: {} secs".format(load_batch_sum), flush=True)
    print("Total Time for batch execution: {} secs".format(batch_comp_sum), flush=True)
    print("Average Time for an epoch: {} secs".format(time_epoch), flush=True)
    print("Average Time for data loading: {} secs".format(load_batch), flush=True)
    print("Average Time for one batch execution: {} secs".format(batch_comp), flush=True)

    return loss


def main(config):
    print("Exp name: %s \nWorkers: %s \nDevice: %s \nOptimizer: %s" % (str(config.expname), str(config.num_workers), str(config.device),  str(config.optimizer)), flush=True)
    images = []
    tags = []
    data_file = config.data_dir + '/train.csv'
    device = torch.device(config.device)
    with open(data_file, 'r') as file:
        lines = file.readlines()
        lines = lines[1:]
        for i in range(len(lines)):
            image_temp, tags_temp = lines[i].split(",")
            images.append(image_temp)
            t = tags_temp.split("\n")
            tags_temp = list(map(int, t[0].split(" ")))
            tags.append(tags_temp)
    train_tranform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    train_data = MyDataset(30000, images, tags, config.data_dir, transform=train_tranform)
    train_data_loader = DataLoader(train_data, batch_size=config.batchsize, num_workers=config.num_workers)

    model = Network().to(device)

    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.mval)
    elif config.optimizer == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.mval, nesterov=True)
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters())
    elif config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())

    loss_criterion = nn.BCELoss()
    loss_val = train(model, loss_criterion, optimizer, train_data_loader, config.epochs, device)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expname", help = "Name of exp")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers")
    parser.add_argument("--data_dir", default="/Users/chhaviyadav/PycharmProjects/HPC", type= str, help="Data Directory")
    parser.add_argument("--device", default='cpu',type=str, help="Device")
    parser.add_argument("--optimizer", default='sgd', type=str, help="Type of Optimizer as String")
    parser.add_argument("--lr", default=0.01, type=float, help="Input Learning Rate")
    parser.add_argument("--batchsize", default=250, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
    parser.add_argument("--m", action='store_true', help="Momentum?")
    parser.add_argument("--mval", default=0.9, type=float, help="momentum value")
    config = parser.parse_args()
    main(config)