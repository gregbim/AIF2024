import argparse
from torchvision.transforms import ToTensor

if __name__=='__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument('--batch_size', type=int, default = 16, help='taille du batch')  # ne pas mettre 1 car peut faire des erreurs
  parser.add_argument('--lr', type=float, default = 0.01, help='learning rate')
  parser.add_argument('--epochs', type=  int, default=10, help='number of raining epochs')

  args = parser.parse_args()
  print('exp_name',args.exp_name)
  print('Batch',args.batch_size)
  print('Learning_rate',args.lr)
  print('TRaining_epochs',args.epochs)
  exp_name = args.exp_name
  epochs = args.epochs
  batch_size = args.batch_size
  lr = args.lr

# objectif : instantiates two data loaders: one loading data from the training set, the other one from the test set.
  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])

  # datasets
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  net = CPU
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(model.parameters(),lr=0.1)

  train(model, optimizer, trainloader, 10)  #(model, optimizer, train_dataloader, nb_epochs)
  test_acc = test(model,testloader)
  print(f'Test accuracy:{test_acc}')

  #sauvegarde des poids du modèle
  torch.save(net.state_dict(), 'weights/mnist_net.pth')