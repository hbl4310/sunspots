import torch
import gpytorch as gp
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from tqdm import tqdm
from utility import newton
import numpy as np 

# Convenience class inheriting from ExactGP
#    https://docs.gpytorch.ai/en/v1.5.1/models.html#exactgp
class GP(gp.models.ExactGP):
    def __init__(self, mean, cov, train_x, train_y, 
            likelihood=GaussianLikelihood(), 
            warp=None,
        ):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean = mean
        self.cov = cov
        self.warp = warp

    def forward(self, x):
        if self.warp: 
            x = self.warp(x)
        return MultivariateNormal(self.mean(x), self.cov(x))

    def predict(self, x, conf=True, train=False):
        if not train: 
            self.eval(), self.likelihood.eval()
        else: 
            self.train(), self.likelihood.train()
        if self.warp: 
            self.warp(x)
        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        # See https://arxiv.org/abs/1803.06058
        # with torch.no_grad(), gp.settings.fast_pred_var():
        with torch.no_grad(), gp.settings.lazily_evaluate_kernels(False):
            pred = self.likelihood(self(x))
            if conf: 
                lower, upper = pred.confidence_region()
        if conf:
            return pred, lower, upper
        return pred
    
    # if the x passed to self.predict is different in shape to the training data, 
    # FixedNoiseGaussianLikelihood will complain since it won't have a noise for some obs.
    # def __hack_FixedNoiseGaussianLikelihood(self, x):
    #     if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
    #         return FixedNoiseGaussianLikelihood(torch.ones_like(x))
    #     else: 
    #         return self.likelihood
    
    def invert_warp(self, x):
        if self.warp:
            return torch.tensor([newton(self.warp, torch.ones(1), target=i) for i in x])
        return x

def train(model, optimizer, mll, train_x, train_y, iters):
    model.train(), model.likelihood.train()
    losses = []
    loss = 0
    iterator = tqdm(range(iters), desc="Epoch")
    for _ in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        iterator.set_postfix(loss=loss.item())
        losses.append(loss.item())
    return losses

def data_loglik(model, x, y, train=False):
    pred = model.predict(x, conf=False, train=train)
    return pred.log_prob(y)

def calc_BIC(model, x, y, train=False):
    # lower BIC better
    loglik = data_loglik(model, x, y, train=train).item()
    n = len(x)
    k = sum(p.numel() for p in model.parameters())
    return k*np.log(n) - 2 * loglik
