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
from models import GP, train, data_loglik, calc_BIC
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
sampledata_mean['year_frac'] += sampling_interval / 12 / 2
sampledata = sampledata_mean

min_train_year = 1848  # 1900
min_test_year = 2018
traindata, testdata = split_train_test(sampledata, min_test_year=min_test_year, min_train_year=min_train_year, year_col='year_frac')

train_x, train_y = to_tensor(traindata['year_frac']), to_tensor(traindata['ssn_total'])
test_x, test_y = to_tensor(testdata['year_frac']), to_tensor(testdata['ssn_total'])

history_x = torch.cat([train_x, test_x])
future_x = torch.tensor([history_x.max() + sampling_interval * 1/12*(i+1) for i in range(11*12)])
future_x = torch.tensor([2022. + sampling_interval * 1/12*(i+1) for i in range(11*12//sampling_interval)])
all_x = torch.cat([history_x, future_x])

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

# %% define kernels
base_kernels = []

for num_mixtures in [2, 4, 6, 8, 10]:
    smk = SpectralMixtureKernel(num_mixtures)
    smk.initialize_from_data_empspect(train_x, transf_train_y)
    base_kernels.append(smk)

base_kernels.append(PeriodicKernel())
base_kernels.append(ScaleKernel(RBFKernel()))
base_kernels.append(ScaleKernel(RQKernel()))
base_kernels.append(ScaleKernel(LinearKernel()))

for x in [1.5, 2.5, 3.5, 4.5]:
    base_kernels.append(ScaleKernel(MaternKernel(x)))


base_compositions = [AdditiveKernel, ProductKernel]


# %%
def fit_model(x, y, kernel, training_iter = 2000, seed=0):
    mean = ConstantMean(prior=gpytorch.priors.NormalPrior(y.mean(), y.std()))
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

    model = GP(mean, kernel, x, y, likelihood=lik, warp=None)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    train(model, optimizer, mll, train_x, transf_train_y, training_iter)
    torch.save(model.state_dict(), f'./models/search/gp_{hash(str(model))}.model')
    return model

# %% 
def kernel_search(x, y, base_kernels, base_compositions, max_depth):
    kernel_cache = dict()
    bic_cache = dict()
    best_bic = np.inf
    best_kernel = None
    print('[INFO] beginning base kernel search')
    for kernel in base_kernels: 
        model = fit_model(x, y, kernel)
        k = hash(str(model))
        kernel_cache[0] = {k: kernel}
        bic = calc_BIC(model, x, y)
        bic_cache[0] = {k: bic}
        if bic < best_bic: 
            best_bic = bic
            best_kernel = kernel
    print(f'''[INFO] finished base kernel search
        best BIC: {best_bic}
        best kernel: {best_kernel}
    ''')

    print('[INFO] beginning composite kernel search')
    for depth in range(max_depth):
        best_kernel_hash, best_bic = max(bic_cache[depth].items(), key=lambda k,v: v)
        best_kernel = kernel_cache[depth][best_kernel_hash]
        new_best_bic = np.inf
        new_best_kernel = None
        for comp in base_compositions:
            for new_kernel in base_kernels:
                kernel = comp(best_kernel, new_kernel)
                model = fit_model(x, y, kernel)
                k = hash(str(model))
                kernel_cache[depth+1] = {k: kernel}
                bic = calc_BIC(model, x, y)
                bic_cache[depth+1] = {k: bic}
                if bic < new_best_kernel:
                    new_best_bic = bic
                    new_best_kernel = kernel
        print(f'''[INFO] finished base kernel search
            best BIC: {new_best_bic}
            best kernel: {new_best_kernel}
        ''')
        if new_best_bic > best_bic: 
            print(f'[INFO] could not find a better composite kernel at depth {depth}; aborting')
            break
    
    return kernel_cache, bic_cache