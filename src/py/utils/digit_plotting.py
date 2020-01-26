import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

digit_size = 28


def plot_digits(*args, invert_colors=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    figure = np.zeros((digit_size * len(args), digit_size * n))

    for i in range(n):
        for j in range(len(args)):
            figure[j * digit_size: (j + 1) * digit_size,
            i * digit_size: (i + 1) * digit_size] = args[j][i].squeeze()

    if invert_colors:
        figure = 1 - figure

    plt.figure(figsize=(2 * n, 2 * len(args)))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


latent_dim=500


def draw_random_picture(generator):
    generator.eval()
    with torch.no_grad():
        normal_vec = torch.as_tensor(np.random.randn(latent_dim), dtype=torch.float).view(1, latent_dim)
        picture = generator(normal_vec)
        picture = torch.squeeze(picture).detach().numpy()
        plt.imshow(picture,  cmap='Greys_r')
        plt.show()
    generator.train()
