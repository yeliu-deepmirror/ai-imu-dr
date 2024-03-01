import sys
sys.path.insert(0, '..')

import torch
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint
from model import DmVelNet
from dataset import InsDataset
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def test_model(model, testing_generator, criterion_mse):
    test_loss = 0
    # model.eval()
    with torch.no_grad():
        for input, target in testing_generator:
            input, target = input.to(args.device), target.to(args.device)
            output = model(input)
            loss = criterion_mse(output, target)
            test_loss += loss.item()
    return test_loss / len(testing_generator)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training_data_path', type=str, default="/ai-imu-dr/data_dm/ai_imu_trainning_data.npz")
    parser.add_argument('--testing_data_path', type=str, default="/ai-imu-dr/data_dm/ai_imu_trainning_data_test_1.npz")

    parser.add_argument('--model_path', type=str, default="/ai-imu-dr/temp/dmvelnet.p")

    # model parameters
    parser.add_argument('--block_size', type=int, default=50)

    # trainning parameters
    parser.add_argument('--plot_path', type=str, default="/ai-imu-dr/temp/dmvelnet_train.png")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--continue_training', type=bool, default=False)
    parser.add_argument('--save_every_epoch', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_grad', type=float, default=20.0)

    args, _ = parser.parse_known_args()

    cprint("Prepare data ...", 'green')
    training_set = InsDataset(args.training_data_path, True, args.block_size)
    testing_set = InsDataset(args.testing_data_path, True, args.block_size)

    dataloader_params = {'batch_size': 1024,
                         'shuffle': True,
                         'num_workers': 6}
    training_generator = torch.utils.data.DataLoader(training_set, **dataloader_params)
    testing_generator = torch.utils.data.DataLoader(testing_set, **dataloader_params)

    print(" training data size", len(training_set))

    cprint("Prepare model ...", 'green')

    torch_meanet = DmVelNet(args.device, training_set.block_size_)
    if args.continue_training:
        torch_meanet.load(args.model_path)
    if args.device == "cuda":
        torch_meanet.cuda()

    print(torch_meanet)

    optimizer = optim.Adadelta(torch_meanet.parameters(), lr=args.learning_rate, weight_decay=args.lr_weight_decay)
    criterion_mse = torch.nn.MSELoss(reduction="sum")

    cprint("Start training ...", 'green')
    losses = []
    test_losses = []
    g_norms = []
    plt.figure(figsize=(20, 8))

    # Loop over epochs
    for epoch in range(args.epochs):
        # Training
        loss_train = 0
        g_norm = 0

        # torch_meanet.train()
        for input, target in training_generator:
            input, target = input.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            output = torch_meanet(input)
            loss = criterion_mse(output, target)
            loss.backward()
            loss_train += loss.item() / input.shape[0]
            g_norm += torch.nn.utils.clip_grad_norm_(torch_meanet.parameters(), args.max_grad).cpu() / input.shape[0]
            optimizer.step()

        # Validation
        loss_test = test_model(torch_meanet, testing_generator, criterion_mse)

        test_losses.append(loss_test)
        losses.append(loss_train)
        g_norms.append(g_norm)
        cprint('  Epoch {:2d} \tLoss: {:.2f} Test Loss: {:.2f}  Gradient Norm: {:.2f}'.format(epoch, loss_train, loss_test, g_norm))

        if epoch%args.plot_every_epoch == 0:
            # plot the trainning to file
            # plt.figure(figsize=(20, 8))
            plt.subplot(1, 2, 1)
            plt.plot(losses, label="train")
            plt.plot(test_losses, label="test")
            plt.legend()
            plt.yscale('log')
            plt.title("train_loss")
            plt.subplot(1, 2, 2)
            plt.plot(g_norms)
            plt.yscale('log')
            plt.title("graident norm")
            plt.savefig(args.plot_path)
            plt.clf()

        if epoch%args.save_every_epoch == 0:
            torch.save(torch_meanet.state_dict(), args.model_path)
            print("  The DmVelNet nets are saved in the file " + args.model_path)

    torch.save(torch_meanet.state_dict(), args.model_path)
    print("  The DmVelNet nets are saved in the file " + args.model_path)
