Proximal MCMC for constrained Gaussians
=======================================

This is an implementation of the [proximal MCMC](https://arxiv.org/abs/1612.07471) method for sampling from linearly constrained
multivariate normal distributions.

For a high-level mathematical documentation, see [here]().


TODO
----

4. Test proximal MCMC on demo problem.
   1. There is a big bug in PxMALA (even in mathematical documentation): y does not neccesarily satisfy constraints?!
5. Implement MYULA.
6. Test MYULA on demo problem.
7. Implement effective sample size.
8. Implement R_hat.