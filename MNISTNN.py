from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import  time
import os

from opzer.dirl_bfgs_nn import DirLBFGS 
from opzer.main_lbfgs import LBFGS
from opzer.bfgs import BFGS
# Training settings

def lbfgs_imp(args, opzer, hs, npp):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    result_path = args.result_path
    result_folder = args.result_folder
    result_folderPath = '{0}/{1}'.format(result_path, result_folder)
    if not os.path.exists(result_folderPath):
        os.makedirs(result_folderPath)
    loss_fileName = '{0}/{1}_log.txt'.format(result_folderPath, result_folder)
    accuracy_fileName = '{0}/{1}_accuracy.txt'.format(result_folderPath, result_folder)


    class Data(Dataset):
        def __init__(self, X_train, y_train):
            self.X = X_train.type(torch.float32)
            self.y = y_train.type(torch.LongTensor)
            # self.X = torch.from_numpy(X_train.astype(np.float32))
            # self.y = torch.from_numpy(y_train).type(torch.LongTensor)
            self.len = self.X.shape[0]

        def __getitem__(self, index):
            return self.X[index], self.y[index]

        def __len__(self):
            return self.len
        
        
    torch.manual_seed(5)
    if args.cuda:
        torch.cuda.manual_seed(5)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    kwargs = {'pin_memory': True} if args.cuda else {}
    mnist_data =  datasets.MNIST('./data', download=False, train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    X_train = mnist_data.data.reshape((mnist_data.data.shape[0],1,mnist_data.data.shape[1],mnist_data.data.shape[2])).to(torch.device("cuda"))
    Y_train = mnist_data.targets.to(torch.device("cuda"))
    traindata = Data(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, drop_last=True)


    X_test = mnist_data.data.reshape((mnist_data.data.shape[0],1,mnist_data.data.shape[1],mnist_data.data.shape[2])).to(torch.device("cuda"))
    Y_test = mnist_data.targets.to(torch.device("cuda"))
    testdata = Data(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=args.test_batch_size, shuffle=False)


   

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
            self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(300, 30)
            self.fc2 = nn.Linear(30, 10)

         


        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 300)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x)

    model = Net()
    if args.cuda:
        model.cuda()
    print(sum(p.numel() for p in model.parameters()))
   



    optimizer = None
    lr = 1
    tg=1e-7
    tc=1e-9
    if opzer == 'lbfgs':
        optimizer = LBFGS(model.parameters(), lr=lr, max_iter=1, history_size=hs, 
                          return_time=True, tolerance_grad=tg, tolerance_change=tc)
    elif opzer == 'bfgs':
        optimizer = BFGS(model.parameters(), lr=lr, max_iter=1, history_size=2, 
                         return_time=True, tolerance_grad=tg, tolerance_change=tc)
    else:
        optimizer = DirLBFGS(model.parameters(), lr=lr, max_iter=1, history_size=hs, preallocate_memory=hs+10,
                         fast_version=True, update_gamma=True, restart=True, 
                         return_time=True, tolerance_grad=tg, tolerance_change=tc)

   
    def train(epoch, times):
        
        model.train()
        avgloss = 0
        tll = time.time()
        puretime_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            times[3] = times[3] + (time.time()-tll)
            
            ttt = time.time()
            target = target.cuda()
            times[2] = times[2] + (time.time()-ttt)
          
            puretime = time.time()
            def closure():
                ttt = time.time()
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()     
                times[0] = times[0] + (time.time()-ttt)   
                return loss
            ttt = time.time()
            loss, tott = optimizer.step(closure)
            times[1] =  times[1] + (time.time()-ttt)
            
            avgloss = avgloss + loss
            if (batch_idx+1) % args.log_interval == 0:
                loss_str = 'Train Epoch: {} iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data)
                print(loss_str)
                print(100*'-')
            tll = time.time()
            puretime_sum = puretime_sum + tott
         
        return avgloss, puretime_sum

    def tstt(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        this_loader = test_loader
        for data, target in this_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(this_loader.dataset)
        accuracy_str = '\nTest set Epoch {:d}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            epoch, test_loss, correct, len(this_loader.dataset),
            100. * correct / len(this_loader.dataset))
        print(accuracy_str)
       
        return test_loss

    total_time = 0
    n_epochs = npp

    loss = []
    tim = []
    # tim.append(0)
    t0 = time.time()
    thresh = 0.000003
    ll = 1000000
    pure_time_list = []
    acctime = 0
    acctt = 0
    train_time = []
    for epoch in range(1, n_epochs + 1):
        times = [0,0,0,0,0]
        print(f'{100*"-"} \nepoch:{epoch}')
        t1 = time.time()
        ll, pt = train(epoch, times)
  
        acctime = acctime + pt
        pure_time_list.append(acctime)
        
        pt2 = time.time()-t1
        acctt = acctt + pt2
        train_time.append(acctt)
        
        if ll < thresh:
            loss.append(torch.tensor(thresh))
            # writer.add_scalar("Loss/train", torch.tensor(thresh), tt)
            break
        ll = tstt(epoch)
        loss.append(ll)
        # writer.add_scalar("Loss/train", ll, tt)  
        print(f'clos:{times[0]}')
        print(f'stp:{times[1]}')
        print(f'load cuda:{times[2]}')
        print(f'load data:{times[3]}')
        print(times[4])
        print(f'hs={len(optimizer.state[optimizer._params[0]]["old_dirs"])} loss={ll} ' 
              f'epoch_time={time.time()-t1 :0.3f} '
              f'total_time={time.time()-t0 :0.3f} '
              f'train_time={train_time[-1] :0.3f} '
              f'pure_time={pure_time_list[-1] :0.3f} ')
        
  
    name = ''
    name = str(opzer) + '_lr' + str(lr) + '_hs' + str(hs) + '_epchs'+ str(len(pure_time_list)) + '_time' + str(f'{train_time[-1]:0.3f}') + '_pime' + str(f'{pure_time_list[-1]:0.3f}')  + str(f'_loss{loss[-1]:.3f}')
    torch.save(loss, name + '_hist.pt')
    torch.save(train_time, name + '_time.pt')
    torch.save(pure_time_list, name + '_pime.pt')
    torch.save(model.state_dict(), name+'_weights.pth')
   


if __name__== "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR',
                        help='learning rate (default: 1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1000000000000000000000000000000000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--result_path', default=None,
                        help='path to store samples and models')
    parser.add_argument('--result_folder', default=None,
                        help='folder to store samples and models')

    args = parser.parse_args()
    
   
    opzer= ['lbfgs']
    hslist = [100]
    n_epochs = [50]
    
    for opz,hs,npp in zip(opzer,hslist,n_epochs):
        lbfgs_imp(args, opzer = opz, hs = hs, npp = npp)