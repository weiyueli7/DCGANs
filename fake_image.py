from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from PIL import Image
from data import*
from model import*

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser(description = 'Generate Fake Images')
parser.add_argument("--label", type=int, default=6, help="which class of fake image to generate")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--image_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
hyperparameters = parser.parse_args()
hyperparameters = list(vars(hyperparameters).values())
hyperparameters = tuple(hyperparameters)
# print(hyperparameters)

label = hyperparameters[0]
# print(label)
if label == 6:
    roots = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
else:
    roots = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'][label]


# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = hyperparameters[2]

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = hyperparameters[8]

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = hyperparameters[1]

# Learning rate for optimizers
lr = hyperparameters[3]

# Beta1 hyperparam for Adam optimizers
beta1 = hyperparameters[4]

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


for r in roots:
    # Root directory for dataset
    dataroot = f"data/seg_train/seg_train/{r}"
    
    dataset = create_data(dataroot, image_size)
    
    dataloader = create_loader(dataset, batch_size, workers)
    
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    
    
    
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    
    
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    Epoch_G_loss = []
    Epoch_D_loss = []
    Epoch_D_acc = []
    all_img = []
    # all_labels = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in tqdm(range(num_epochs)):
        # For each batch in the dataloader
        E_G_loss, E_D_loss, E_D_acc, batch_count = 0, 0, 0, 0
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            output_label = torch.round(output)
            E_D_acc += ((output_label == label).sum().item()) / len(data[0])
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            E_G_loss += errG.item()
            E_D_loss += errD.item()

            
            if epoch == num_epochs-1:
                all_img.append(netG(torch.randn(len(data[1]), nz, 1, 1, device=device)).detach().cpu())
                
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
            batch_count += 1

        Epoch_G_loss.append(E_G_loss/batch_count)
        Epoch_D_loss.append(E_D_loss/batch_count)
        Epoch_D_acc.append(E_D_acc/batch_count)
        
        
    label_names = [r]

    if not os.path.exists('data/fake'):
        [os.makedirs(f'data/fake/{name}') for name in label_names]

    for lab in label_names:
        all_files = os.listdir(f'data/fake/{lab}')
        [os.remove(f'data/fake/{lab}/{f}') for f in all_files]



    im_name = 0
    for i in range(len(all_img)):
        for j in range(len(all_img[i])):
            lab = r
            image = all_img[i][j].numpy().transpose(1, 2, 0) 
            image = (image - image.min()) / (image.max() - image.min()) #to normalize
            image = (image * 255).astype(np.uint8)
            im = Image.fromarray(image, 'RGB')
            im.save(f'data/fake/{lab}/{im_name}.png')
            im_name += 1
    print(f'Fake {r} images done!')

print("All requested fake images done!")