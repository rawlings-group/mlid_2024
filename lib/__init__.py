"""
TurnkeyMPC
==========

State-space models
------------------
StateSpace.py
  class StateSpace
  method sysid
  method mpc (tracking)
  method empc (economic)
  method mhe

NonlinearStateSpace.py
  class NonlinearStateSpace <- StateSpace
  method lmpc (tracking)
  method elmpc (economic)
  method ekf
  method lmhe

  class NLTI <- LinearStateSpace
  class NLTV <- LinearStateSpace

LinearStateSpace.py
  class LinearStateSpace <- StateSpace
  method lqr
  method kf

  class LTI <- LinearStateSpace
  class LTV <- LinearStateSpace

System identification
---------------------
arx.py
  func arx (LTI)
  func narx (NonlinearStateSpace)
subspace.py
  func hokalman
  func jointmle
  func subspace (cca, n4sid, and moesp)
  func pca_id
  func pls_id (like cca but with PLS instead of RRR)
nucnorm.py
  func n2sid
  func nnarx
mle.py
  func pem (LTI/LTV/NLTI/NLTV)
  func mle (LTI/LTV/NLTI/NLTV)

Other helper functions
----------------------
regression.py
  func multiple_regression
  func reduced_rank_regression
  func nuclear_norm_regression
  func nonlinear_regression
optimze.py
  func nlpsolve
  func nspdsolve
util.py
  func hankel
"""
