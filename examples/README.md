# mlid_2024/examples/
This folder contains example scripts for identifying stochastic linear models (and linear augmented disturbance models) with eigenvalue constraints.

- siso_mle.py: An example identification experiment and model fit of a simple scalar system.
- siso_dmle.py: An example identification experiment and /disturbance model/ fit of a simple scalar system.
- tclab_mle.py: Identification of the TCLab benchmark temperature control laboratory, fit to data from [Kuntz and Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344).
- tclab_dmle.py: The case study models from Section VI.A of [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760), fit to data from [Kuntz and Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344).

# References

[Kuntz, S. J. and Rawlings, J. B. (2022).](https://ieeexplore.ieee.org/abstract/document/9867344) “Maximum likelihood estimation of linear disturbance models for offset-free model predictive control,” in American Control Conference, Atlanta, GA, June 8–10, 2022, pp. 3961– 3966.

[Kuntz, S. J. and Rawlings, J. B. (2024).](https://arxiv.org/pdf/2406.03760) Maximum likelihood identification of uncontrollable linear time-invariant models for offset-free control. arXiv preprint arXiv:2406.03760.