# %%
from gpytorch.kernels.periodic_kernel import PeriodicKernel
from gpytorch.likelihoods import (
    GaussianLikelihood, 
    FixedNoiseGaussianLikelihood,
)
from gpytorch.means import (
    ZeroMean, 
    ConstantMean, 
    LinearMean,
)
# http://www.gaussianprocess.org/gpml/chapters/RW4.pdf
# https://www.cs.toronto.edu/~duvenaud/cookbook/
from gpytorch.kernels import (
    RBFKernel,
    RQKernel,  
    MaternKernel,
    SpectralMixtureKernel,
    AdditiveKernel,
    PolynomialKernel,
    ScaleKernel,
    ProductKernel,
    LinearKernel
)

import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from torch.nn.modules.loss import SoftMarginLoss

from data import (
    get_silso_data, 
    centre_x, 
    split_train_test, 
    to_tensor, 
    sample_random, sample_interval, sample_meaninterval, 
    plot_data,
)
from models import GP, train, data_loglik
from utility import (
    spectral_density, 
    get_peaks, 
    plot_losses, 
    plot_density, 
    plot_kernel, 
    plot_transform, 
)
from transforms import (
    identity, 
    log_transform, exp_transform, 
    invsoftplus_transform, softplus_transform, 
    power_transform, 
    TanhWarp, LogWarp, 
)

# %% data
data = get_silso_data('SN_m_tot_V2.0')
data = centre_x(data)

sampledata_rand = sample_random(data, 200)
sampling_interval = 4  # months
f_s = 12 / sampling_interval
sampledata_int = sample_interval(data, sampling_interval)
sampledata_mean = sample_meaninterval(data, sampling_interval, ['year_frac', 'ssn_total'])
sampledata = sampledata_mean

min_train_year = 1848  # 1900
min_test_year = 2018
traindata, testdata = split_train_test(sampledata, min_test_year=min_test_year, min_train_year=min_train_year, year_col='year_frac')

train_x, train_y = to_tensor(traindata['year_frac']), to_tensor(traindata['ssn_total'])
test_x, test_y = to_tensor(testdata['year_frac']), to_tensor(testdata['ssn_total'])

history_x = torch.cat([train_x, test_x])
future_x = torch.tensor([history_x.max() + 1/12*(i+1) for i in range(11*12)])
all_x = torch.cat([history_x, future_x])

#%% plot data
fig, ax = plot_data(data, 'year_frac', 'ssn_total')
ax.scatter(train_x, train_y)

# %% transforms
p = 1/2
pt = power_transform(p)
ipt = power_transform(1/p)
input_transform = lambda x: log_transform(pt(x))
output_transform = lambda x: ipt(exp_transform(x))
input_transform = pt
output_transform = ipt

transf_train_y = input_transform(train_y)
transf_train_y = input_transform(train_y)
transf_test_y = input_transform(test_y)

assert torch.allclose(train_y, output_transform(transf_train_y))
plot_transform(train_y, input_transform)

# %% define model
num_mixtures = 8
smk = SpectralMixtureKernel(num_mixtures)
# smk.initialize_from_data(train_x, transf_train_y)
smk.initialize_from_data_empspect(train_x, transf_train_y)

# kernel = ProductKernel(RBFKernel(), PeriodicKernel())
# kernel = smk
# kernel = AdditiveKernel(
#     smk, 
#     ScaleKernel(RBFKernel()),
# )
kernel = AdditiveKernel(
    smk, 
    ScaleKernel(RQKernel()),
)
# kernel = AdditiveKernel(
#     smk, 
#     ScaleKernel(MaternKernel(2.5)),
# )
# kernel = ProductKernel(
#     smk, 
#     LinearKernel()
# )

# mean = ConstantMean()
# mean = ConstantMean(prior=gpytorch.priors.NormalPrior(transf_train_y.mean(), 1))
mean = ConstantMean(prior=gpytorch.priors.NormalPrior(transf_train_y.mean(), transf_train_y.std()))

# lik = GaussianLikelihood()
# add prior to noise
noise_prior = gpytorch.priors.GammaPrior(1.05, 0.05)
noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
lik = GaussianLikelihood(
    noise_prior=noise_prior,
    noise_constraint=gpytorch.constraints.constraints.GreaterThan(
        1e-6,
        transform=None,
        initial_value=noise_prior_mode,
    ),
)
# this adds white noise to the kernel; see: https://github.com/cornellius-gp/gpytorch/issues/1128
# can add noise to specific observations, e.g. earlier observations with less reliable techniques/instruments
# lik = FixedNoiseGaussianLikelihood(torch.ones_like(transf_train_y), learn_additional_noise=True)

warp = None

model = GP(mean, kernel, train_x, transf_train_y, likelihood=lik, warp=warp)

print('parameters:')
for n,p in model.named_parameters():
    print(f'  {n}: {p.numel()}')

# %% train model and save
%%time 
torch.manual_seed(0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
training_iter = 2000

losses = train(model, optimizer, mll, train_x, transf_train_y, training_iter)
fig, ax = plot_losses(losses)

torch.save(model.state_dict(), f'./models/gp_{hash(str(model))}.model')

# %% load model
# model = GP(mean, kernel, train_x, transf_train_y, likelihood=lik, warp=warp)
# model.load_state_dict(torch.load('./models/gp_smk+rq_1800data.model'))

# %% analysis
def plot_model_fit(model, smk, f_s=12, figsize=(14, 12), transform=identity):
    pred, lower, upper = model.predict(all_x)

    if transform == identity: 
        ts = [(transform, 'ssn_total'), ]
        nplots = 2
    else:
        ts = [(identity, 'transformed ssn_total'), (transform, 'ssn_total')]
        nplots = 3
    fig, ax = plt.subplots(nplots, 1, figsize=figsize)
    for i,(t,ylbl) in enumerate(ts): 
        ax_t = ax[i]
        ax_t.set_title('Model fit')
        ax_t.set_xlabel('t')
        ax_t.set_ylabel(ylbl)
        ax_t.plot(all_x, t(pred.mean).numpy().flatten(), label='Predicted mean')
        ax_t.fill_between(all_x, t(lower).numpy(), t(upper).numpy(), alpha=0.5, label='2 Stdev.')
        data_ = data.loc[data.year_frac > min_train_year]
        ax_t.scatter(data_.year_frac, t(input_transform(data_.ssn_total)), label='Raw obs.', marker='x', alpha=0.3)
        ax_t.scatter(train_x, t(transf_train_y), label='Train')
        ax_t.scatter(test_x, t(transf_test_y), label='Test')
        ax_t.legend(loc='upper left', ncol=2)

    density = spectral_density(smk)
    nyquist = f_s/2
    freq = torch.linspace(0, nyquist, 5000).reshape(-1, 1)
    density2 = density.log_prob(freq).exp()
    plot_density(freq, density2, ax=ax[nplots-1])
    ax[nplots-1].set_ylabel('Density')
    fig.tight_layout()
    return freq, density2, fig, ax

print(f'train log-likelihood: {data_loglik(model, train_x, transf_train_y):.2f}')
print(f'test log-likelihood: {data_loglik(model, test_x, transf_test_y):.2f}')
freq, density, fig, ax = plot_model_fit(model, smk, f_s=f_s, transform=output_transform)
peak_freq, peak_density = get_peaks(freq, density)
print(f'freq peaks at: {", ".join([f"{1/f:.2f} ({d:.2f})" for f,d in zip(peak_freq, peak_density)])}')


# %%
def plot_model_kernel(kernel, figsize=(8, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_kernel(kernel, xx=torch.linspace(-2, 2, 1000), ax=ax)
    ax.set_title('Learned kernel')
    fig.tight_layout()
    return fig, ax

def plot_covariance(kernel, grid, figsize=(8, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(kernel(grid, grid).numpy())
    tickindex = [int(i) for i in ax.get_xticks()[1:-1]]
    ticklabels = grid.numpy()[tickindex].tolist() 
    ticklabels = [f'{int(l)}' for l in ticklabels]
    ax.set_xticks(tickindex)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(tickindex)
    ax.set_yticklabels(ticklabels)
    ax.set_title('Induced covariance matrix')
    fig.tight_layout()
    return fig, ax

fig, ax = plot_model_kernel(model.cov)
fig, ax = plot_model_kernel(smk)
fig, ax = plot_covariance(model.cov, history_x)




# %% scratch
warp = LogWarp()
# f = lambda x: warp(x) - torch.tensor([1.2])
f = warp
x = newton(f, torch.ones(1), target=1.3737)
print("x = %s, f(x) = %s" % (x.numpy()[0], f(x).numpy()[0]))


# %%

# https://math.stackexchange.com/questions/1671132/equation-for-a-smooth-staircase-function
def fstep(x, h, w, a):
    k = x/w
    return h * (
        torch.tanh(a * k - a * torch.div(k, 1, rounding_mode='trunc') - a/2) 
        / (2 * torch.tanh(a / 2)) 
        + 1/2 + torch.div(k, 1, rounding_mode='trunc')
    )

def fstep_cos(x, a, w, c):
    return a * torch.cos((x / w - c)* 2 * torch.pi) + a/w * 2 * torch.pi * x


x = torch.linspace(0, 40, 1000)
# y = fstep(x, torch.tensor(1), torch.tensor(11), torch.tensor(3))
y = fstep_cos(x, torch.tensor(0.8), torch.tensor(11), torch.tensor(1))
plt.plot(x, y)

# %%
fig, ax = plt.subplots(3, figsize=(12, 8))
ax[0].scatter(train_x, torch.sqrt(output_transform(train_y)))
ax[1].hist(output_transform(train_y).pow(3/7).tolist())
ax[2].scatter(fstep_cos(train_x, torch.tensor(0.8), torch.tensor(5.2), torch.tensor(0)), output_transform(train_y).pow(3/7).tolist())

# %%
pred, lower, upper = model.predict(to_tensor(data.year_frac))
fig, ax = plt.subplots(figsize=(30,10))
ax.plot(data.year_frac, output_transform(pred.mean))
ax.scatter(data.year_frac, data.ssn_total)
# plt.plot(data.year_frac, output_transform(lower))
# plt.plot(data.year_frac, output_transform(upper))
# %%
