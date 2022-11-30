import argparse
import os
import torch
import torch.optim as optim
from torchvision import datasets


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading
    from data import data_transforms

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images_shiyao',
                             transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    from model import ResNet, Vgg, ViT, resnet_ssl, deeplab, Resnet50
    # model = ResNet()
    # model = Vgg()
    # model = ViT()
    model = resnet_ssl()
    # model = Resnet50()
    # model = deeplab()
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    # optimizer = optim.SGD([{"params" : model.layers.parameters(), "lr" : 1e-4, "momentum" : 0.9},
    #                        {"params": model.pooling.parameters(), "lr": 1e-4, "momentum": 0.9},
    #                        {"params" : model.classifier.parameters(), "lr" : 1e-4, "momentum" : 0.9},])
    optimizer = optim.Adam([{"params" : model.layers.parameters(), "lr" : 5e-5},
                                 {"params": model.pooling.parameters(), "lr": 5e-5},
                                 {"params": model.classifier.parameters(), "lr": 5e-5}
                            ])
    scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=args.epochs)
    def train(epoch):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # weights = ResNet152_Weights.IMAGENET1K_V2
            # preprocess = weights.transforms()
            # data = preprocess(data)

            output = model(data)

            optimizer.zero_grad()
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
        del data, target
        scheduler.step()

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return validation_loss, 100. * correct / len(val_loader.dataset)

    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        loss, accuracy = validation()
        # model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        model_file = args.experiment + '/best_model' + '.pth'
        if best_accuracy < accuracy:
            torch.save(model.state_dict(), model_file)
            best_accuracy = accuracy
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
        model_file2 = args.experiment + '/last_model' + '.pth'
        torch.save(model.state_dict(), model_file2)


def ssl():
    # from Grey to RGB using UNET whose encoder is ResNet
    from model import UNET
    from data import ssl_dataset, show
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(3047)

    train_dataset = ssl_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1, shuffle=False, num_workers=1)

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    model = UNET()
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    optimizer = optim.Adam([{"params": model.parameters(), "lr": 1e-4, },])

    def train(epoch):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            # optimizer.zero_grad()
            criterion = torch.nn.MSELoss(reduction='mean')
            loss = criterion(output, target)
            loss.backward()
            if batch_idx % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))
        optimizer.step()

    def eval():
        import numpy as np
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            data, target, output = data.cpu(), target.cpu(), output.detach().cpu()
            data, target, output = data[0].permute(1,2,0), target[0].permute(1,2,0), output[0].permute(1,2,0)
            data, target = np.clip(255 * data.numpy(), 0, 255).astype(np.uint8), \
                           (255 * target.numpy()).astype(np.uint8)
            output = np.clip(255 * output.numpy(), 0, 255).astype(np.uint8)
            # show(data)
            show(output)
            # show(target)
            break

    for epoch in range(1, 10 + 1):
        train(epoch)
        eval()
        model_file = "experiment" + '/ssl_last_model' + '.pth'
        torch.save(model.encoder.state_dict(), model_file)

if __name__ =="__main__":
    main()
    def eval():
        from data import show, data_transforms
        from model import ResNet

        use_cuda = torch.cuda.is_available()
        torch.manual_seed(1)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('bird_dataset' + '/val_images',
                                 transform=data_transforms),
            batch_size=64, shuffle=False, num_workers=1)
        model = ResNet()
        if use_cuda:
            print('Using GPU')
            model.cuda()
        else:
            print('Using CPU')

        state_dict = torch.load("experiment/last_model.pth")
        model.load_state_dict(state_dict)

        model.eval()
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            fake_mask = (pred != target.data.view_as(pred)).flatten()
            print(fake_mask)
            print(target)
            print("=====================")
    # eval()
    # ssl()

