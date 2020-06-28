import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import wandb

import matplotlib.pyplot as plt

wandb.init(job_type='train', project='CVAE')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bs = 100

train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size =bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

class CVAE(nn.Module):
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim, c_dim):
        super(CVAE, self).__init__()

        #encoder
        self.fc1 = nn.Linear(x_dim + c_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc31 = nn.Linear(h2_dim, z_dim)
        self.fc32 = nn.Linear(h2_dim, z_dim)

        #decoder
        self.fc4 = nn.Linear(z_dim + c_dim, h2_dim)
        self.fc5 = nn.Linear(h2_dim, h1_dim)
        self.fc6 = nn.Linear(h1_dim, x_dim)
    
    def encoder(self, x, c):
        concat_input = torch.cat([x.view(-1,784),c],1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling_z(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    
    def decoder(self, z, c):
        concat_input = torch.cat([z,c],1)
        h = F.relu(self.fc4(concat_input))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))
    
    def forward(self, x, c):
        mu, log_var = self.encoder(x,c)
        z = self.sampling_z(mu, log_var)
        return self.decoder(z,c), mu, log_var



c_dim = train_loader.dataset.train_labels.unique().size(0)
cvae = CVAE(x_dim=784, h1_dim=512, h2_dim=512, z_dim=2, c_dim=c_dim)
cvae.to(device)

optimizer = optim.Adam(cvae.parameters())

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1,784), reduction = 'sum')
    KLD = -0.5*torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var))
    
    return BCE + KLD

def one_hot_encoding(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i,label] = 1
    
    return targets

def train(epoch):
    cvae.train()
    train_loss = 0
    for batch_id, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = one_hot_encoding(label, c_dim).to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = cvae(data, label)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if batch_id % 100 == 0:
            print('Train epoch {} \t Loss {}'.format(batch_id/len(data), loss.item()/len(data)))
    
    
    print('Average Loss: ', train_loss/len(train_loader.dataset))
    wandb.log({'Train Loss ':train_loss/len(train_loader.dataset)}, step=epoch)


def test():
    cvae.eval()
    test_loss = 0

    with torch.no_grad():
        for data, label in test_loader:
            label = one_hot_encoding(label, c_dim).to(device)
            data = data.to(device)
            recon_batch, mu, log_var = cvae(data, label)

            test_loss += loss_function(recon_batch, data, mu, log_var)

    print('Test Loss: ', test_loss/len(test_loader))


for epoch in range(1,51):
    train(epoch)
    test()

    z = torch.randn(bs,2).to(device)
    c = torch.eye(bs,c_dim).to(device)
    sample = cvae.decoder(z, c)
    sample =sample.view(bs,1,28,28)
    wandb.log({"Images": [wandb.Image(sample, caption="Images for epoch: "+str(epoch))]}, step=epoch)
    

torch.save(cvae.state_dict(), './ckpt/cvae.pth')

with torch.no_grad():
    z = torch.randn(bs,2).to(device)
    c = torch.eye(bs, c_dim).cuda()
    sample = cvae.decoder(z, c).to(device)
    save_image(sample.view(bs,1,28,28), './sample.png')