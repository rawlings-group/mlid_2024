# mlid_2024/data/
This folder contains example data from experimenting with the TCLab benchmark temperature control laboratory.

- tclab_kuntz_rawlings_2022.mat: A file containing the model from [Kuntz and Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344).
- tclab_cl_prbs.pickle: An example identification experiment. This data was used in [Kuntz and Rawlings (2022)](https://ieeexplore.ieee.org/abstract/document/9867344) and [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760).
- tclab_cl.pickle: An example closed-loop setpoint change test. This data was used in [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760).
- tclab_cl_dist.pickle: An example closed-loop disturbance rejection test. This data was used in [Kuntz and Rawlings (2024)](https://arxiv.org/pdf/2406.03760).

Example scripts for producing all but the last data file are included in `mlid_2024/experimental_tools/`.

# References

[Kuntz, S. J. and Rawlings, J. B. (2022).](https://ieeexplore.ieee.org/abstract/document/9867344) “Maximum likelihood estimation of linear disturbance models for offset-free model predictive control,” in American Control Conference, Atlanta, GA, June 8–10, 2022, pp. 3961– 3966.

[Kuntz, S. J. and Rawlings, J. B. (2024).](https://arxiv.org/pdf/2406.03760) Maximum likelihood identification of uncontrollable linear time-invariant models for offset-free control. arXiv preprint arXiv:2406.03760.