import torch
from torch.autograd import Variable, grad
from preprocess import get_loader
from tqdm import tqdm
import torch.optim as optim
import os
import argparse
from lstm_CNN import UNet1D_LA
from Discriminator import Critic1DCNN
import matplotlib.pyplot as plt
from loss_func import PinballLoss
import numpy as np

parser = argparse.ArgumentParser()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def gradient_penalty(critic, real_data, input, fake_data, device):
    batch_size, seq_len, features = real_data.shape
    alpha = torch.rand(batch_size, 1, 1).expand_as(real_data).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    prob_interpolated = critic(input, interpolated)

    gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                     grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                     create_graph=True, retain_graph=True)[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def denormalize(data, lower_bound, upper_bound):
    return data * (upper_bound - lower_bound) + lower_bound


def visualize_and_save(output, label, epoch, idx):
    output_dir = 'visualization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(output)):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))

        for j in range(4):
            axs[j].plot(output[i, :, j], label='Output')
            axs[j].plot(label[i, :, j], label='Label')
            #axs[j].legend()
            #axs[j].set_title(f'Feature {j + 1}')

        # plt.tight_layout()
        # plt.suptitle(f'Epoch {epoch} - Sample {i + idx * 100}')
        # plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(output_dir, f'{i + idx * 100}.png'))
        plt.cla()
        plt.clf()


def main():
    train_loader = get_loader('train.npy', batch_size=256, mode='train')
    test_loader = get_loader('test.npy', batch_size=100, mode='valid')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = UNet1D_LA().to(device)
    critic = Critic1DCNN().to(device)

    g_optimizer = optim.AdamW(generator.parameters(), lr=2e-3)
    c_optimizer = optim.RMSprop(critic.parameters(), lr=1e-3)

    lambda_gp = 10
    lambda_l1 = 10
    pin_loss = PinballLoss(torch.tensor([0.9]).to(device)).to(device)

    bounds = np.load('bounds.npy')
    lower_bound = bounds[0]
    upper_bound = bounds[1]

    for epoch in range(500):
        generator.train()
        critic.train()
        for i, (input, label) in enumerate(tqdm(train_loader)):
            input = Variable(input.to(device))
            input = input.float().squeeze(1)
            label = Variable(label.to(device))
            label = label.float().squeeze(1)

            # Train Critic

            fake_output = generator(input)

            real_validity = critic(input, label)
            fake_validity = critic(input, fake_output.detach())

            gp = gradient_penalty(critic, label, input, fake_output, device)

            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            fake_output = generator(input)
            fake_validity = critic(input, fake_output)
            g_loss = -torch.mean(fake_validity) + lambda_l1 * pin_loss(fake_output, label)

            g_loss.backward()
            g_optimizer.step()

            if i % 500 == 0:
                print(f"Epoch [{epoch}/{1}], Step [{i}/{len(train_loader)}], "
                      f"D Loss: {c_loss.item()}, G Loss: {g_loss.item()}")

        if epoch >1 and epoch % 100 == 0:
            with torch.no_grad():
                generator.eval()
                for idx, (input, label) in enumerate(test_loader):
                    if idx < 1:
                        input = Variable(input.to(device))
                        input = input.float().squeeze(1)
                        label = Variable(label.to(device))
                        label = label.float().squeeze(1)

                        fake_output = generator(input)

                        # Denormalize
                        denormalized_output = denormalize(fake_output.cpu().numpy(), lower_bound[3:], upper_bound[3:])
                        denormalized_label = denormalize(label.cpu().numpy(), lower_bound[3:], upper_bound[3:])

                        # Visualize and save
                        visualize_and_save(denormalized_output, denormalized_label, epoch, idx)


if __name__ == '__main__':
    main()
