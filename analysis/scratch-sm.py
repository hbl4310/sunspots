# %%
import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

# %% data
data_path = '../data/SILSO data/SN_m_tot_V2.0.csv'

data = pd.read_csv(data_path, sep=';', names=['year', 'month', 'year_frac', 'ssn_total', 'ssn_stdev', 'nobs', 'marker'])

data['year_frac2'] = data['year'] + data['month'] * 1/12 - 1/24
min_year = data['year'].min()
min_train_year = 1900
min_test_year = 2018

data.index

traindata = data.loc[(min_train_year <= data.year) & (data.year < min_test_year)]
train_x = torch.tensor(traindata['year_frac2'].values, dtype=torch.float32) 
train_y = torch.tensor(traindata['ssn_total'].values, dtype=torch.float32)

# downsampling: random, interval, TODO summary e.g. average or median
random_sample = np.random.choice(traindata.index, 200)
random_sample.sort()
random_sample = traindata.index[::3]
sample_train_x = torch.tensor(traindata.loc[random_sample, 'year_frac2'].values, dtype=torch.float32) 
sample_train_y = torch.tensor(traindata.loc[random_sample, 'ssn_total'].values, dtype=torch.float32)

testdata = data.loc[data.year >= min_test_year]
test_x = torch.tensor(testdata['year_frac2'].values, dtype=torch.float32) 
test_y = torch.tensor(testdata['ssn_total'].values, dtype=torch.float32)

history_x = torch.cat([train_x, test_x])
future_x = torch.tensor([history_x.max() + 1/12*(i+1) for i in range(11*12)])
all_x = torch.cat([history_x, future_x])

#%% plot data
plt.figure(figsize=(30, 8))
plt.scatter(range(len(data)), data['ssn_total'])

# %%
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
# TODO look at heteroskedastic noise
# gpytorch.likelihoods.FixedNoiseGaussianLikelihood
model = SpectralMixtureGPModel(train_x, train_y, likelihood)

# %%
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# https://pytorch.org/docs/1.9.1/generated/torch.optim.LBFGS.html

# %%
%%time
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
losses = []
training_iter = 100

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)

# %%
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Make predictions
    observed_pred = likelihood(model(all_x))

    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(30,8))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(all_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(all_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

# %%
for n,p in model.named_parameters():
    print(n)
    print(p.data)

# %% access parameters
model.mean_module.constant
model.likelihood.noise_covar.raw_noise
model.covar_module.raw_mixture_weights
model.covar_module.raw_mixture_means
model.covar_module.raw_mixture_scales



# %% try with different model
%%time
from gpytorch_models import SMKernelGP

model = SMKernelGP(sample_train_x, sample_train_y, num_mixtures=10)

model.train()

lr = 0.05
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

training_iter = 2000
losses = []
loss = 0
iterator = tqdm(range(training_iter), desc="Epoch")
for _ in iterator:
    optimizer.zero_grad()
    output = model(sample_train_x)
    loss = -mll(output, sample_train_y)
    loss.backward()
    optimizer.step()
    iterator.set_postfix(loss=loss.item())
    losses.append(loss.item())

fig, ax = plt.subplots()
ax.plot(losses)
ax.set_ylabel('log-loss')
ax.set_yscale('log')
ax.set_xlabel('epochs')

# %% evaluate
from utility import spectral_density, plot_density, plot_kernel

pred, lower, upper = model.predict(all_x)

def plot_model_fit(f_s=12):
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(14, 12))
    ax_t.set_title("Model fit")
    ax_t.set_xlabel("t")
    ax_t.set_ylabel("y(t)")
    ax_t.plot(all_x, pred.numpy().flatten(), label="Predicted mean")
    ax_t.fill_between(all_x, lower.numpy(), upper.numpy(), alpha=0.5, label="Confidence")
    ax_t.scatter(train_x, train_y, label="Train obs.", marker='x', alpha=0.3)
    ax_t.scatter(sample_train_x, sample_train_y, label="Sample Train obs.")
    ax_t.scatter(test_x, test_y, label="Test obs.")
    ax_t.legend()

    density = model.spectral_density()
    nyquist = f_s/2
    freq = torch.linspace(0, nyquist, 5000).reshape(-1, 1)
    density2 = density.log_prob(freq).exp()
    plot_density(freq, density2, ax=ax_f)
    ax_f.set_ylabel("Density")
    fig.tight_layout()
    return freq, density2

def plot_model_kernel():
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_kernel(model.cov, xx=torch.linspace(-2, 2, 1000), ax=ax, col="tab:blue")
    ax.set_title("Learned kernel")
    fig.tight_layout()
    return fig, ax

def plot_covariance(kernel, grid):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.matshow(kernel(grid, grid).numpy())
    ax.set_title("Induced covariance matrix")
    fig.tight_layout()
    return fig, ax

freq, density = plot_model_fit()
plot_model_kernel()
plot_covariance(model.cov, all_x)
# %%
