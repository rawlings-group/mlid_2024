# [Maximum Likelihood Identification of Linear Models with Integrating Disturbances for Offset-Free Control](https://arxiv.org/pdf/2406.03760)

This repository contains code for the paper [Kuntz and Rawlings
(2024)](https://arxiv.org/pdf/2406.03760). The example files can be used to
fully reproduce the results from Section VI.A of this paper.

# Prerequisites

```
numpy
scipy
matplotlib
time
ipopt
casadi
cvxpy
mosek*
tclab**
```

$^*$ While many of the examples can be run with the default SDP solver in
`cvxpy`, some will fail at the initial guess generation step. The examples in
the paper were solved with [MOSEK](https://www.mosek.com/).

$^{**}$ The TCLab benchmark temperature control laboratory ([Park, et. al
  (2020)](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)) is
  only required if you wish to replicate the results with your own experiments.

# Usage

This repository is standalone and does not require installation as long as the
prerequisites are met. To run an example file, simply run from the main
directory,

```
python3 examples/<example>.py
python3 figures/<example>_plot.py
```

which will generate `data/<example>.pickle` and
`figures/<example>_plot.pdf` files.

# Contents

- `data/` contains example data from the papers [Kuntz and Rawlings
  (2022)](https://ieeexplore.ieee.org/abstract/document/9867344) and [Kuntz and
  Rawlings (2024)](https://arxiv.org/pdf/2406.03760). Data files from the
  `examples/` folder are also dumped here.
- `examples/` contains example identification scripts, including the case study
  from Section VI.A of [Kuntz and Rawlings
  (2024)](https://arxiv.org/pdf/2406.03760).
- `experimental_tools/` contains scripts for experimenting with the TCLab
  benchmark temperature control laboratory ([Park, et. al
  (2020)](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)).
- `idtools/` contains library files for the main algorithm and other
  identification tools.

## `data/`
This folder contains example data from experimenting with the TCLab benchmark
temperature control laboratory ([Park, et. al
(2020)](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)).

- `tclab_kuntz_rawlings_2022.mat`: A file containing the model from [Kuntz and
  Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344).
- `tclab_cl_prbs.pickle`: An example identification experiment. This data was
  used in [Kuntz and Rawlings
  (2022)](https://ieeexplore.ieee.org/abstract/document/9867344) and [Kuntz and
  Rawlings (2024)](https://arxiv.org/pdf/2406.03760).
- `tclab_cl.pickle`: An example closed-loop setpoint change test. This data was
  used in [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760).
- `tclab_cl_dist.pickle`: An example closed-loop disturbance rejection test.
  This data was used in [Kuntz and Rawlings
  (2024)](https://arxiv.org/pdf/2406.03760).

Example scripts for producing all but the last of these data files are included
in `experimental_tools/`.

The scripts in `examples/` also produce data files that are dumped here. See the
`examples/` section for more information about those files.

## `examples/`
This folder contains example scripts for identifying stochastic linear models
(and linear augmented disturbance models) with eigenvalue constraints.

- `siso_mle.py`: An example identification experiment and model fit of a simple
  scalar system.
- `siso_dmle.py`: An example identification experiment and /disturbance model/
  fit of a simple scalar system.
- `tclab_mle.py`: Identification of the TCLab benchmark temperature control
  laboratory, fit to data from [Kuntz and Rawlings
  (2022)](https://ieeexplore.ieee.org/abstract/document/9867344).
- `tclab_dmle.py`: The case study models from Section VI.A of [Kuntz and
  Rawlings (2024)](https://arxiv.org/pdf/2406.03760), fit to data from [Kuntz
  and Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344).

NOTE: `tclab_dmle.py` only runs reliably if `cvxpy` is configured with
[MOSEK](https://www.mosek.com/).

## `experimental_tools/`
This folder contains example scripts for experimenting with the TCLab benchmark
temperature control laboratory ([Park, et. al
(2020)](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)).

- `tclab_prbs.py`: An example identification experiment.
- `tclab_cl.py`: An example closed-loop setpoint change test.
- `tclab_cl_dist.py`: An example closed-loop disturbance rejection test.

Example output files for these scripts are included in `data/`.

## `figures/`
This folder contains plotting files for the data produced by the scripts in
`examples/` and `experimental_tools/`. See the `examples/` and
`experimental_tools/` sections for more information on those scripts.

## `idtools/`
This folder contains the identification methods. The main methods files are:

- `arx.py`: Autoregressive models.
- `ssid.py`: Subspace identification.
- `ssmle.py`: State space maximum likelihood identification and prediction error
  methods.

The remaining files contain helper methods:

- `linalg.py`: Linear algebra tools (mostly for safe function overloading with
  numpy and casadi).
- `plotter.py`: Plotting methods.
- `regression.py`: An efficient linear-equality-constrained linear regression
  module that uses CasADi when constraints are given.
- `util.py`: Everything else.

# References

[Kuntz, S. J., Rawlings, J. B.
(2022).](https://ieeexplore.ieee.org/abstract/document/9867344) “Maximum
likelihood estimation of linear disturbance models for offset-free model
predictive control,” in American Control Conference, Atlanta, GA, June 8–10,
2022, pp. 3961– 3966.

[Kuntz, S. J., Rawlings, J. B. (2024).](https://arxiv.org/pdf/2406.03760)
Maximum likelihood identification of uncontrollable linear time-invariant models
for offset-free control. arXiv preprint arXiv:2406.03760.

[Park, J., Martin, R. A., Kelly, J. D., & Hedengren, J. D.
(2020).](https://apm.byu.edu/prism/uploads/Members/2020_park_tclab.pdf)
Benchmark temperature microcontroller for process dynamics and control.
Computers & Chemical Engineering, 135, 106736.
