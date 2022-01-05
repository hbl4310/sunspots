import torch
import gpytorch
from gpytorch.priors import (
    GammaPrior, 
    NormalPrior, 
)

# transforms 
identity = lambda x: x 

log_transform = lambda x: torch.log(x)
exp_transform = lambda x: torch.exp(x)

invsoftplus_transform = lambda x: torch.log(torch.exp(x) - 1)
softplus_transform = lambda x: torch.log(torch.exp(x) + 1)

def power_transform(k):
    return lambda x: x.pow(k)


# https://github.com/cornellius-gp/gpytorch/issues/756
class TanhWarp(gpytorch.Module):
    def __init__(self, n, 
            scale_prior=GammaPrior(1, 1), 
            shift_prior=NormalPrior(0, 1)):
        super(TanhWarp, self).__init__()
        self.n = n
        for i in range(n): 
            for _ab in ['a', 'b']:
                pn = f'{_ab}_{i}'
                self.register_parameter(name=f'{pn}_raw', parameter=torch.nn.Parameter(scale_prior.sample()))
                self.register_constraint(f'{pn}_raw', gpytorch.constraints.Positive())
                self.register_prior(f'{pn}_prior', scale_prior, f'{pn}_raw')
                    # lambda m: getattr(m, f'{pn}_raw_constraint').transform(getattr(m, f'{pn}_raw')))
            pn = f'c_{i}'
            self.register_parameter(name=pn, parameter=torch.nn.Parameter(shift_prior.sample()))
            # self.register_prior(f'{pn}_prior', shift_prior, lambda m: getattr(m, pn))
            self.register_prior(f'{pn}_prior', shift_prior, pn)

    def forward(self, x):
        # clone x to avoid in-place modification; not sure if that is the case or is necessary
        y = x.clone()
        for i in range(self.n):
            a = getattr(self, f'a_{i}_raw_constraint').transform(getattr(self, f'a_{i}_raw'))
            b = getattr(self, f'b_{i}_raw_constraint').transform(getattr(self, f'b_{i}_raw'))
            c = getattr(self, f'c_{i}')
            y += a * torch.tanh(b * (x - c))
        return y

class LogWarp(gpytorch.Module):
    def __init__(self):
        super(LogWarp, self).__init__()
    
    def forward(self, x):
        return torch.log(x + 1)

# class BetaCDFWarp(gpytorch.Module):
    
#     def __init__(self, a=None, b=None):
#         super(BetaCDFWarp, self).__init__()
#         a = torch.rand(1) if a is None else torch.tensor([a])
#         b = torch.rand(1) if b is None else torch.tensor([b])

#         self.register_parameter(name="a_raw", parameter=torch.nn.Parameter(a))
#         self.register_constraint("a_raw", gpytorch.constraints.Positive())
#         self.register_prior("a_prior", gpytorch.priors.GammaPrior(1, 1), lambda: self.a)
        
#         self.register_parameter(name="b_raw", parameter=torch.nn.Parameter(b))
#         self.register_constraint("b_raw", gpytorch.constraints.Positive())
#         self.register_prior("b_prior", gpytorch.priors.GammaPrior(1, 1), lambda: self.b)

#     @property
#     def a(self):
#         return self.a_raw_constraint.transform(self.a_raw)
    
#     @property
#     def b(self):
#         return self.b_raw_constraint.transform(self.b_raw)
        
#     def forward(self, x):
#         # cannot detach otherwise gradient will not flow through 
#         # need to implement beta cdf in torch
#         a = self.a.detach()
#         b = self.b.detach()
#         warp_numpy = beta(a, b).cdf(x)
#         return torch.from_numpy(warp_numpy)