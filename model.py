import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianEmbedding(nn.Module):
    def __init__(self, embed_size, counts, window, batch_size, covariance_type,
                 device, verbose, mu_0=0.1, mu_c=2.0, sigma_min=0.5, sigma_max=2.0,
                 margin=0.1, power=0.75):
        super(GaussianEmbedding, self).__init__()

        self.verbose = verbose

        self.vocab_size = len(counts)
        self.embed_size = embed_size
        self.covariance_type = covariance_type
        self.device = device
        self.margin = margin

        self.mu_c = torch.tensor(mu_c).to(device)
        self.sigma_min = torch.tensor(sigma_min).to(device)
        self.sigma_max = torch.tensor(sigma_max).to(device)

        self.mu = nn.Embedding(self.vocab_size, embed_size)
        if covariance_type == 'diagonal':
            self.sigma = nn.Embedding(self.vocab_size, embed_size)
        elif covariance_type == 'spherical':
            self.sigma = nn.Embedding(self.vocab_size, 1)

        self.init_weights(mu_0, sigma_min, sigma_max)
        self.sample_iter = self.negative_sample_gen(counts,
                                                    power,
                                                    batch_size,
                                                    window - 1)

    def init_weights(self, mu_0, sigma_min, sigma_max):
        self.mu.weight.data.uniform_(0, mu_0)
        if self.covariance_type == 'diagonal':
            self.sigma.weight.data.uniform_(sigma_min, sigma_max)
        elif self.covariance_type == 'spherical':
            raise NotImplementedError

    def negative_sample_gen(self, counts, power, batch_size, sample_size):
        p = torch.pow(counts.float(), power)
        weights = p / p.sum()

        while True:
            yield torch.multinomial(weights.repeat(batch_size, 1),
                                    sample_size).to(self.device)

    def regularize_weights(self):
        l2_mu = torch.sqrt(self.mu.weight.data.pow(2.0).sum(dim=1)).unsqueeze(1)
        self.mu.weight.data = torch.where(l2_mu > self.mu_c, self.mu_c / l2_mu,
                                          self.mu.weight.data)
        self.sigma.weight.data = torch.max(self.sigma_min,
                                           torch.min(self.sigma.weight.data,
                                                     self.sigma_max))

    def forward(self, input):
        target, context = input
        batch_size = len(target)

        mean = self.mu(target)
        cov = (torch.eye(self.embed_size, device=self.device) *
               self.sigma(target).view(
                       batch_size, -1, 1, self.embed_size
                   )
               )
        target_dist = MultivariateNormal(mean, cov)

        context_pos = context
        context_neg = next(self.sample_iter)[:batch_size]

        if self.verbose:
            # note target and context are corpus_len - window
            print(f"target shape = {target.shape}")
            print(f"context shape = {context.shape}")

            # target repeats the current entity (window-1) times to allow for easy comparison
            # with the context [also (window-1) size] when computing KLs below (same size tensors)
            print(f"target = {target}")
            # +ve context maps to the (window-1) other entities in the window
            print(f"+ve context = {context_pos}")
            # -ve context maps to the entities not in the current window
            print(f"-ve context = {context_neg}")
            print(f"target_dist = {target_dist}")



        ###### CONTEXT DISTRIBUTIONS ######
        # context_dist has
        # mean shape [corpus_len-window, window-1, size]
        # cov shape [corpus_len-window, window-1, size, size]
        # the first dimension tracks which window we are referring to
        # the second dimension tracks which entities appear in that window
        # the third (and fourth) dimensions record the current mean and cov values

        # positive sample
        mean_pos = self.mu(context_pos)
        cov_pos = (torch.eye(self.embed_size, device=self.device) *
                   self.sigma(context_pos).view(
                           batch_size, -1, 1, self.embed_size))

        context_pos_dist = MultivariateNormal(mean_pos, cov_pos)

        # negative sample
        mean_neg = self.mu(context_neg)
        cov_neg = (torch.eye(self.embed_size, device=self.device) *
                   self.sigma(context_neg).view(
                           batch_size, -1, 1, self.embed_size
                       )
                   )
        context_neg_dist = MultivariateNormal(mean_neg, cov_neg)
        if self.verbose:
            print(f"mean_pos = {mean_pos.shape}")
            print(f"cov_pos = {cov_pos.shape}")
            print(f"context_pos_dist = {context_pos_dist}")



        ###### LOSS ######
        # loss = max(0, m - [KL_pos - KL_neg])
        # if KL_pos - KL_neg >= m then no loss accumulated
        # if KL_pos - KL_neg < m then m - [KL_pos - KL_neg] added to loss
        # this encourages the target distn to have a KL (overlap) with the
        # current window (positive context) that is at least m more than with
        # the negative context (entities outside the window)

        kl_pos = kl_divergence(target_dist, context_pos_dist)
        kl_neg = kl_divergence(target_dist, context_neg_dist)
        loss = torch.max(torch.tensor(0., device=self.device),
                         self.margin - kl_pos + kl_neg).sum(dim=1).mean()

        return loss
