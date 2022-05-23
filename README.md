Proximal MCMC for constrained Gaussians
=======================================

This is an implementation of the [proximal MCMC](https://arxiv.org/abs/1612.07471) method for sampling from linearly constrained
multivariate normal distributions.

For a high-level mathematical documentation, see [here]().


TODO
----

11. Implement automatic choice of sample size (using w-criterion).
12. Implement autotune: Automatically adapt delta during warmup in order to achieve a 50% acceptance ratio.