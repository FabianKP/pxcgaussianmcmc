Proximal MCMC for constrained Gaussians
=======================================

This is an implementation of the `proximal MCMC`_ method for sampling from linearly constrained
multivariate normal distributions.

.. _proximal MCMC: https://arxiv.org/abs/1612.07471

For a high-level mathematical documentation, see [here]().

.. math::
    \mathbf x \sim \mathcal N(\mathbf m, \mathbf \Sigma), \\
    \mathbf A \mathbf x = \mathbf b, \\
    \mathbf C \mathbf x \geq \mathbf d, \\
    \mathbf l \leq \mathbf x \leq \mathbf u.


TODO
----

1. Write mathematical documentation.
2. Make a demo-case.
3. Implement outer loop (MCMC).
4. Implement inner loop (proximal operator).