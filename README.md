Proximal MCMC for constrained Gaussians
=======================================

This is an implementation of the [proximal MCMC](https://arxiv.org/abs/1612.07471) method for sampling from linearly constrained
multivariate normal distributions.

For a high-level mathematical documentation, see [here]().


TODO
----

9. Implement R_hat.
10. Implement w_value.
11. Implement autotune: Automatically adapt delta during warmup in order to achieve a 50% acceptance ratio.