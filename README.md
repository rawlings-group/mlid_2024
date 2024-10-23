# [Maximum Likelihood Identification of Linear Models with Integrating Disturbances for Offset-Free Control](https://arxiv.org/pdf/2406.03760)

This repository contains code for the paper [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760). 

- `data/' contains example data from the papers [Kuntz and Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344) and [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760). 
- `examples/` contains example identification scripts, including the case study from Section VI.A of [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760). 
- `lib/` contains library files for the main algorithm and other identification tools.

# Prerequisites

```
numpy
scipy
casadi
ipopt
matplotlib
time
```

# References

[Kuntz, S. J. and Rawlings, J. B. (2022).](https://ieeexplore.ieee.org/abstract/document/9867344) “Maximum likelihood estimation of linear disturbance models for offset-free model predictive control,” in American Control Conference, Atlanta, GA, June 8–10, 2022, pp. 3961– 3966.

[Kuntz, S. J. and Rawlings, J. B. (2024).](https://arxiv.org/pdf/2406.03760) Maximum likelihood identification of uncontrollable linear time-invariant models for offset-free control. arXiv preprint arXiv:2406.03760.