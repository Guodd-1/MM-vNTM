import math
import torch
from torch.distributions.kl import register_kl

from ..ops.ive import ive, ive_fraction_approx, ive_fraction_approx2
from .hyperspherical_uniform import HypersphericalUniform

# Define the Von Mises-Fisher distribution class for modeling distributions on high-dimensional spheres
class VonMisesFisher(torch.distributions.Distribution):
    # Define parameter constraints: `loc` is real-valued, `scale` is positive
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real  # support domain
    has_rsample = True  # indicates that this distribution supports rsample (reparameterization trick)
    _mean_carrier_measure = 0

    # Define mean property, computed using different approximation methods
    @property
    def mean(self):
        # Option 1: compute mean using the `ive` function
        # return self.loc * (
        #     ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        # )
        # Option 2: use `ive_fraction_approx` for approximation
        return self.loc * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # Option 3:
        # return self.loc * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

    # Define standard deviation property, simply return the `scale` parameter
    @property
    def stddev(self):
        return self.scale

    # Initialize the distribution; `loc` is the mean direction, `scale` is the concentration (kappa),
    # `k` is a parameter used during sampling (e.g., number of proposals in rejection sampling)
    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype  # store data type
        self.loc = loc  # mean direction (unit vector)
        self.scale = scale  # concentration parameter (kappa > 0)
        self.device = loc.device  # device (CPU/GPU)
        self.__m = loc.shape[-1]  # dimensionality of the sphere (embedding dimension)
        self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)  # canonical basis vector e1
        self.k = k  # sampling parameter (e.g., max proposals in rejection sampling)

        # Call parent class initializer
        super().__init__(self.loc.size(), validate_args=validate_args)

    # Define `sample` method: delegate to `rsample` under no-gradient context
    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    # Define `rsample` method for reparameterized sampling (enables gradient flow)
    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        # Choose sampling method based on dimension `m`
        w = (
            self.__sample_w3(shape=shape)
            if self.__m == 3
            else self.__sample_w_rej(shape=shape)
        )

        # Sample from standard normal, normalize to unit vectors (for orthogonal complement)
        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        # Construct sample on the sphere: x = [w; sqrt(1−w²) * v]
        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)  # rotate to align with `loc`

        return z.type(self.dtype)

    # Specialized sampler for 3D case (m = 3), using inverse transform sampling
    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )
        return self.__w

    # Rejection sampler for general dimension (m > 3)
    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # Use Taylor expansion to avoid numerical instability for large kappa
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        # Perform rejection sampling; `k` controls the number of proposals per attempt
        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    # Helper function: find index of first non-zero element along a dimension
    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    # Rejection sampling loop: generate proposals until accepted
    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]
        w, e, bool_mask = (
            torch.zeros_like(b).to(self.device),
            torch.zeros_like(b).to(self.device),
            (torch.ones_like(b) == 1).to(self.device),
        )

        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)

        # Sample from Beta and Uniform distributions for rejection sampling
        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            # Acceptance condition: log acceptance probability > log(u)
            accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

            reject = accept_idx < 0
            accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e.reshape(shape), w.reshape(shape)

    # Householder rotation: reflect vector `x` to align with `loc` direction
    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    # Compute entropy: measure of uncertainty of the vMF distribution
    def entropy(self):
        # Option 1: compute using `ive`
        # output = (
        #     -self.scale
        #     * ive(self.__m / 2, self.scale)
        #     / ive((self.__m / 2) - 1, self.scale)
        # )
        # Option 2: use `ive_fraction_approx` for stable approximation
        output = - self.scale * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # Option 3:
        # output = - self.scale * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

        return output.view(*(output.shape[:-1]))  # + self._log_normalization()

    # Compute log-probability density
    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    # Compute unnormalized log-probability (dot product term)
    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    # Compute log-normalization constant (log C_d(kappa))
    def _log_normalization(self):
        output = -(
            (self.__m / 2 - 1) * torch.log(self.scale)
            - (self.__m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale)))
        )
        return output.view(*(output.shape[:-1]))


# Register KL divergence from VonMisesFisher to HypersphericalUniform
@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    # print(vmf.entropy(), hyu.entropy())
    return -vmf.entropy() + hyu.entropy()