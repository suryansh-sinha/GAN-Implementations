"""GAN Implementation.ipynb

# Implementation of Basic Generative Adversarial Network in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

"""Defining our Discriminator and Generator classes."""

# Creating the discriminator.
class Discriminator(nn.Module):
  # We create in_features. Since we're using the mnist dataset, it's gonna be 28*28 = 784
  def __init__(self, img_dim):
    super().__init__()
    self.disc = nn.Sequential(
        nn.Linear(img_dim, 128),  # Hidden Layer 1 with 128 nodes.
        nn.LeakyReLU(0.1),  # Activation for HL1.
        nn.Linear(128, 1),  # Output Node. This has only 1 node because it's output is real of fake.
        nn.Sigmoid(), # Activation Function for the output node.
    )

  # Forward Propagation for discriminator.
  def forward(self, x):
    return self.disc(x)

class Generator(nn.Module):
  # z_dim = dimension of the latent noise that will be the input.
  # img_dim = our output layer must have dimensions same as the real dataset image.
  def __init__(self, z_dim, img_dim):
    super().__init__()
    self.gen = nn.Sequential(
        nn.Linear(z_dim, 256),  #  Hidden Layer 1 - 256 nodes.
        nn.LeakyReLU(0.1),  # Activation Function
        nn.Linear(256, img_dim),  # Output layer. Nodes same dim as our img data.
        #  To make sure output of pixel values is b/w -1 and 1. This is because we normalize our input to be b/w -1 and 1.
        nn.Tanh(),  # Activation Function.
    )

  def forward(self, x):
    return self.gen(x)

# hyperparameters. GANs are very sensitive to hyperparameters.
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"
lr = 3e-4 # 3e-4 is the best learning rate for adam hands down ~ Andrej Karpathy
z_dim = 64  # can also try 128, 256. This is the dimension for latent space (random noise).
image_dim = 28 * 28 * 1 # 784
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)

# noise. tensor filled with random numbers from a normal distribution.
fixed_noise = torch.randn((batch_size, z_dim)).to(device) # Moving the tensor to GPU if available.

# Transforms is used for image transformations. It's basically for image transformation, augmentation.
# transforms.Compose can chain multiple transforms functions together.
# We need to pass in a list of Transform objects to the Compose function.
transforms = transforms.Compose(
    [transforms.ToTensor(), # Converts PIL image to numpy array.
     # The parameters passed are the mean and standard deviation, used for normalization.
     transforms.Normalize((0.5,), (0.5,))]
)

# Loading the data
dataset = datasets.MNIST(root='dataset/', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Creating the optimizers
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# Loss
criterion = nn.BCELoss()

# Tensorboard
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

# Training loop
for epoch in range(num_epochs):
  for batch_idx, (real, _) in enumerate(loader):
    
    real = real.view(-1, 784).to(device)
    batch_size = real.shape[0]

    ### Train Discriminator: max log(D(real)) + log(1 - D(G(z)))

    noise = torch.randn(batch_size, z_dim).to(device) # Generating a random noise distribution
    fake = gen(noise) # Passing that noise through generator to generate fake data. Basically p_g(x)
    
    # passing the real data into the discriminator, getting the probability D(real)
    # .view(-1) reshapes the probability tensor into a single dimension tensor.
    disc_real = disc(real).view(-1) 
    # Passing D(x) and a tensor of 1s (label for real data) to the loss function.
    lossD_real = criterion(disc_real, torch.ones_like(disc_real))

    # Passing the fake data through the discriminator, generating a tensor of probabilites
    # This is basically D(G(z))
    # Reshaping it to a single dimension tensor (flattening).
    # Using fake.detach() here because PyTorch destroys parts of computation graphs when backward is called and we want to use this later on for generator.
    # Another alternative we could do here would be to use retain_graph=True parameter during the .backward() call.
    disc_fake = disc(fake.detach()).view(-1)
    # Passing D(G(z)) and tensor of zeros(label for fake data) for loss calculation.
    lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

    lossD = (lossD_real + lossD_fake) / 2
    disc.zero_grad()  # setting gradients to zero.
    lossD.backward()  # Calculating the gradients for the loss.
    opt_disc.step()   # updating our weights using the optimizer.

    ### Training the Generator: max(log(D(G(z)))).
    # This is the better loss function as it prevents saturated gradients, slow training.
    output = disc(fake).view(-1)  # D(G(z)), reshaped to a single dimension tensor.
    lossG = criterion(output, torch.ones_like(output))
    gen.zero_grad() # setting the gradients of generator to 0.
    lossG.backward()  # calculating new gradients.
    opt_gen.step()  # optimizing the weights and biases of the generator.

    # Code for Tensorboard
    if batch_idx == 0:
      print(
          f'Epoch [{epoch}/{num_epochs}] \ '
          f'Loss D: {lossD: .4f}, Loss G: {lossG: .4f}'
      )

      with torch.no_grad():
        fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
        data = real.reshape(-1, 1, 28, 28)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(data, normalize=True)

        writer_fake.add_image(
            "Mnist Fake Images", img_grid_fake, global_step=step
        )

        writer_real.add_image(
            "Mnist Real Images", img_grid_real, global_step=step
        )

        step += 1

