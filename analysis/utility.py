import torch
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal

import matplotlib.pyplot as plt
import numpy as np

# for spectral mixture kernel, pull out the spectral density from the kernel parameters
def spectral_density(smk) -> MixtureSameFamily:
    """Returns the Mixture of Gaussians thet model the spectral density
    of the provided spectral mixture kernel."""
    mus = smk.mixture_means.detach().reshape(-1, 1)
    sigmas = smk.mixture_scales.detach().reshape(-1, 1)
    mix = Categorical(smk.mixture_weights.detach())
    comp = Independent(Normal(mus, sigmas), 1)
    return MixtureSameFamily(mix, comp)

# get density peaks
def get_peaks(freq, density):
    peaks = (density > torch.cat([torch.tensor([0]), density[:-1]])) * (density > torch.cat([density[1:], torch.tensor([0])]))
    peaks_sortindex = density[peaks].argsort(descending=True)
    return freq.flatten()[peaks][peaks_sortindex], density[peaks][peaks_sortindex]

# newton inversion method 
def newton(func, start, target = torch.zeros(1),  
        threshold = 1e-4, iterations=10):
    guess = torch.autograd.Variable(start, requires_grad=True)
    value = func(guess)
    for _ in range(iterations):
        previous_guess = guess.clone()
        value = func(guess) - target
        value.backward()
        guess.data -= (value / guess.grad).data
        # print(previous_guess.item(), value.item(), guess.item(), guess.grad.item())
        guess.grad.data.zero_()
        if torch.abs(guess - previous_guess) < torch.tensor(threshold) or guess.isnan():
            break
    return guess.data

# plotting functions
def plot_losses(losses, figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(losses)
    ax.set_ylabel('log-objective')
    ax.set_yscale('log')
    ax.set_xlabel('epochs')
    ax.set_title('Training Log-Likelihood objective')
    ax.annotate(f'final loss: {losses[-1]:.2e}', 
        xy=(len(losses)-1, losses[-1]), xycoords='data', 
        xytext=(0.8, 0.5), textcoords='axes fraction', 
        arrowprops={'facecolor':'black', 'shrink':0.05, 'width': 0.3, 'headwidth': 7}, 
        horizontalalignment='right', verticalalignment='top',
    )
    return fig, ax

def plot_density(freq, density, ax):
    x = freq.numpy().flatten()
    y = density.numpy().flatten()
    ax.plot(x, y, color='tab:blue', lw=3)
    ax.fill_between(x, y, np.ones_like(x) * y.min(), color='tab:blue', alpha=0.5)
    ax.set_title('Kernel spectral density')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Log Density')

def plot_kernel(kernel, ax, xx=torch.linspace(-0.1, 0.1, 1000), col="tab:blue"):
    x0 = torch.zeros(xx.size(0))
    ax.plot(xx.numpy(), np.diag(kernel(xx, x0).numpy()), color=col)

def plot_transform(data, warp):
    if not type(data) is torch.Tensor: 
        data = torch.tensor(data).flatten()
    x = data.tolist()
    x.sort()
    wx = np.linspace(x[0], x[-1], 1000)
    wy = warp(torch.tensor(wx)).tolist()
    fx = warp(torch.tensor(x).flatten()).tolist()

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.plot(wx, wy)
    ax.set_xlabel('Original')
    ax.set_ylabel('Transformed')

    ax_histx.hist(x)
    ax_histy.hist(fx, orientation='horizontal')
    return fig, ax