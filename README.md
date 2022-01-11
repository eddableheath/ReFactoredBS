# Refactored Boson Sampling

The old version of the Boson Sampling code was becoming too messy so this is a refactor.

Notable differences here:
* Use of the constructed unitary via matrix exponentiation of the Gramm matrix.
* Instead of simulating single runs of a Boson Sampler each time the entire output distribution is computed and sampled from. This is done for the following reasons:
  * Whilst explicitly computing the output distribution is a computationally complex task, sampling is essentially free - in comparison to the simlutation at least - so computing it once and sampling is more efficient over large sample numbers.
  * Since the final lattice sample is a sum over many individual samples of the Boson Sampler the number of samples needed is multiplied so this makes even more sense.
