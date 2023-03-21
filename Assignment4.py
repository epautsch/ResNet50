import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import time


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.5):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dropout_rate=0.5):
        super(ResNet, self).__init__()
        self.in_channels = 128 #64

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False) #64
        self.bn1 = nn.BatchNorm2d(128)
        self.layer1 = self.make_layer(block, 128, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self.make_layer(block, 256, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self.make_layer(block, 512, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self.make_layer(block, 1024, num_blocks[3], stride=2, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(1024, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, dropout_rate=dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

data_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.7 * len(data_set))
test_size = len(data_set) - train_size
train_set, test_set = random_split(data_set, [train_size, test_size])

PRINTED_MESSAGES = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', 'GPU' if device.type == 'cuda' else 'CPU')


def worker_init_fn(worker_id):
    global PRINTED_MESSAGES
    if not PRINTED_MESSAGES:
        print('Files already downloaded and verified')
        PRINTED_MESSAGES = True


# import multiprocessing
#
# num_cpu_cores = multiprocessing.cpu_count()
# print(f'Number of CPU cores: {num_cpu_cores}')

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

net = ResNet(ResNetBlock, [3, 4, 6, 3], dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


def train(net, data_loader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)


def test(net, data_loader, device, return_last_10=False):
    net.eval()
    correct = 0
    total = 0
    last_10_true = []
    last_10_pred = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_last_10 and i >= len(data_loader) - 10:
                last_10_true.extend(labels.cpu().numpy())
                last_10_pred.extend(predicted.cpu().numpy())

    if return_last_10:
        return 100 * correct / total, last_10_true, last_10_pred
    else:
        return 100 * correct / total


def plot_training_history(train_losses, val_accuracies, save_name):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training Loss (Filters double)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'b', label='Validation Accuracy (Filters double)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_name)
    plt.show()


if __name__ == '__main__':
    num_epochs = 50
    train_losses = []
    val_accuracies = []

    print('Starting training...')
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train(net, train_loader, criterion, optimizer, device)
        epoch_end_time = time.time()
        epoch_total_time = epoch_end_time - epoch_start_time

        val_accuracy = test(net, test_loader, device)

        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Time: {epoch_total_time:.2f}s')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f}s')

    plot_training_history(train_losses, val_accuracies, 'resnet_filters_double.png')

    accuracy, last_10_true, last_10_pred = test(net, test_loader, device, return_last_10=True)
    print(f'Final Test Accuracy: {accuracy:.2f}%')
    print('Last 10 image results:')

    results_last_10 = ''
    for i, (true_label, pred_label) in enumerate(zip(last_10_true, last_10_pred)):
        results_last_10 += f'Image {i + 1}: True Class: {true_label}, Pred Class: {pred_label}\n'

    print(results_last_10)


#### Filters normal ####
# Total training time: 2941.27s
# Final Test Accuracy: 89.77%
# Last 10 image results:
# Image 1: True Class: 4, Pred Class: 4
# Image 2: True Class: 6, Pred Class: 2
# Image 3: True Class: 4, Pred Class: 4
# Image 4: True Class: 3, Pred Class: 3
# Image 5: True Class: 9, Pred Class: 9
# Image 6: True Class: 6, Pred Class: 6
# Image 7: True Class: 1, Pred Class: 1
# Image 8: True Class: 8, Pred Class: 8
# Image 9: True Class: 1, Pred Class: 1
# Image 10: True Class: 4, Pred Class: 4

#### Filters double ####
# Total training time: 4945.22s
# Final Test Accuracy: 91.23%
# Last 10 image results:
# Image 1: True Class: 3, Pred Class: 3
# Image 2: True Class: 1, Pred Class: 1
# Image 3: True Class: 9, Pred Class: 0
# Image 4: True Class: 0, Pred Class: 0
# Image 5: True Class: 1, Pred Class: 1
# Image 6: True Class: 2, Pred Class: 2
# Image 7: True Class: 3, Pred Class: 3
# Image 8: True Class: 0, Pred Class: 0
# Image 9: True Class: 3, Pred Class: 3
# Image 10: True Class: 3, Pred Class: 5





