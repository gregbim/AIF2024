import argparse
from statistics import mean
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import MNISTNet

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 #pour la sauvegarde des loss
writer = SummaryWriter()

def train(net, optimizer, loader, writer, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)  # permet d'avoir une barre d'avancement
        for x, y in t:                         # boucle sur les batch
            x, y = x.to(device), y.to(device)  #par défaut, envoie sur CPU. Permet de mettre sur GPU (avec le modèle)= tout mettre sur même mémoire
            outputs = net(x)                   # les vafiables explicatives passent par le réseau
            loss = criterion(outputs, y)       # est un tenseur de réels
            running_loss.append(loss.item())   #stockage des résultats de la loss  loss.item() permet de récupérer nombre réel du tenseur
            optimizer.zero_grad()              #commande pour ré-initialiser explicitement le gradient (sinon accumule les gradients. Utile pour les jedis)
            loss.backward()                    #calcul du gradient
            optimizer.step()                   #mise à jours des poids
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0  #incrémentation d'un compteur
    total = 0
    with torch.no_grad():  #pas de calcul de gradient en test
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)       #index du plus grand selon dimension du tenseur (0=batch, 1=vecteur de classe)
            test_corrects += y_hat.eq(y).sum().item()
            #y_hat.eq(y) : comparaison des valeurs prédites aux réelles, puis on en fait la somme
            total += y.size(0)
    return test_corrects / total
#permet d'obtenir l'accuracy sur le jeu de test

if __name__=='__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument('--batch_size', type=int, default = 16, help='taille du batch')  # ne pas mettre 1 car peut faire des erreurs
  parser.add_argument('--lr', type=float, default = 0.01, help='learning rate')
  parser.add_argument('--epochs', type=  int, default=10, help='number of raining epochs')

  args = parser.parse_args()
  exp_name = args.exp_name
  epochs = args.epochs
  batch_size = args.batch_size
  lr = args.lr

# objectif : instantiates two data loaders: one loading data from the training set, the other one from the test set.
  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])     #pour chaque channel ((moyenne),(écart type))

  # datasets
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  net = MNISTNet()
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(net.parameters(),lr=0.1)

  train(net, optimizer, trainloader, writer,epochs)  #(model, optimizer, train_dataloader, nb_epochs) net, optimizer, loader, writer, epochs
  test_acc = test(net,testloader)
  print(f'Test accuracy:{test_acc}')

  #sauvegarde des poids du modèle
  if not os.path.exists('./models'):
  	os.mkdir('./models')
  torch.save(net.state_dict(), 'weights/mnist_net.pth')  #dossier 'weights' à cérer avant de lancer l'apprentissage 

  #add embeddings to tensorboard
  perm = torch.randperm(len(trainset.data))
  images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
  images = images.unsqueeze(1).float().to(device)
  
  with torch.no_grad():
    embeddings = net.get_features(images)
    writer.add_embedding(embeddings, metadata=labels,label_img=images, global_step=1)
                  
  # save networks computational graph in tensorboard
  writer.add_graph(net, images)
  # save a dataset sample in tensorboard
  img_grid = torchvision.utils.make_grid(images[:64])
  writer.add_image('mnist_images', img_grid)