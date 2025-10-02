# AGENTS

## Overview
- The codebase builds and analyzes perturbation-theory power spectrum models for multiple tracers, targeting Fisher-forecast computations.
- Workflow: configure fiducial cosmology/bias settings, instantiate a `Model` that evaluates the multipole spectra (possibly using precomputed CLASS outputs), derive parameter sensitivities, assemble covariance matrices, and finally construct Fisher matrices.
- The `CLASS-PT` directory is treated as an external numerical backend that supplies perturbation-theory inputs; its internals are not covered here.

## Module Breakdown
### fiducial_cosmo.py
- Provides default fiducial values for cosmological parameters (`OMEGA_CDM_FID`, `H_FID`, `AS_FID`) and bias/counterterm amplitudes (`B1_VALUE`, `CCT_2_0_VALUE`, etc.).
- Exposes a large `FIXED_PRIOR` that serves as the default prior width for non-constrained parameters in other modules.

### modellib.py
- Imports the defaults from `fiducial_cosmo.py` and exposes helpers for building parameter dictionaries expected by `modelclass.Model`.
- `import_and_overwrite_globals(filepath)` executes a Python file to overwrite module-level constants, mirroring the helper inside `modelclass.py`.
- `create_bias_params(...)` and `create_stochastic_params(...)` assemble dictionaries for each tracer (and tracer pair) with metadata (`name`, `value`, `priorsigma`, `active`, `type`, `traceridx`). The function branches between real-space and RSD use-cases, enabling extra counterterms when RSD is on.
- `set_model(...)` orchestrates model construction: it creates cosmological, bias, and stochastic parameter sets for the requested number of tracers, configures shot-noise assumptions, and finally instantiates `modelclass.Model` with toggles controlling non-linearity, RSD, multipoles, caching behaviour, and fiducial offsets per tracer.

### modelclass.py
- Hosts the core `Model` class that holds the fiducial spectra, parameters, and derivative machinery.
- Initialization steps:
  - Accepts cosmological/bias/stochastic dictionaries and survey settings from `modellib.set_model`.
  - Stores bookkeeping such as tracer counts, which parameters are active, and the k-grid (`set_ks`). The k-grid is fixed to a hard-coded array that is masked to obey `kmin`/`kmax` and used to derive mode-count weights (`Nk_loc`).
  - Defines the fiducial parameter vector (`self.all_parameters_values`) and extracts active entries/priors for Fisher forecasting.
  - Calls `compute_cosmo_func(...)` to obtain multipole spectra and growth-rate information. This method either calls the CLASS Python API (`classy.Class`) or loads/saves cached pickles, depending on the `open_precomputed_files`/`save_precomputed_files` flags. The method returns the multipoles on the masked k-array and the normalisation wavenumber.
  - Builds the fiducial data vector via `get_data(...)`, looping over tracer pairs and calling either `spec_model_nlin` (real-space) or `spec_model_nlin_rsd` (RSD). These assemble linear plus 1-loop perturbation-theory pieces, bias combinations, counterterms, and stochastic contributions; optional Gaussian noise can be injected for mock data generation.
  - Precomputes analytic derivative templates: `get_analytic_deriv()` symbolically differentiates the real-space model with SymPy, while `get_analytic_deriv_rsd()` does the same for each RSD multipole.
- Public methods of interest:
  - `get_data(...)`: recomputes spectra for arbitrary parameter vectors, used when finite-differencing cosmological parameters.
  - `get_derivatives(...)`: returns derivatives of the data vector with respect to all active parameters. Cosmological parameters are differentiated numerically via a fourth-order finite difference (`numerical_derivative`), while bias/stochastic parameters use the cached symbolic derivatives evaluated through `eval_derivative(...)`. A `FLAG_NOCROSS_SPEC` path reshapes outputs when cross-spectra are excluded.
  - `get_cosmo_bias_stoch_set(...)`: slices the flattened parameter vector back into tracer-specific bias and stochastic blocks.
  - `spec_model_nlin(...)` / `spec_model_nlin_rsd(...)`: implement the perturbation-theory formulae for real-space and RSD multipoles respectively, combining bias parameters with precomputed perturbative kernels coming from CLASS-PT.
  - `compute_cosmo_func(...)`: interfaces with CLASS to obtain raw multipole kernels, optionally storing them on disk for later reuse.
  - Utility functions (`_initialize_tracer_matrix`, `_initialize_parameters`, `set_ks`, etc.) prepare internal structures for later computations.
- External dependencies: relies on `classy` for cosmology, SymPy for analytic derivatives, NumPy for numerics, and matplotlib/pickle for diagnostics and caching.

### covariance.py
- Builds the covariance matrix of the model’s data vector under Gaussian assumptions.
- `get_covariance(...)` is a dispatcher that selects real-space vs. RSD routines.
- `get_covariance_real(...)` loops over all tracer combinations and k-bins, populating the standard disconnected covariance terms `(PAC·PBD + PAD·PBC)/Nk`. It uses `model.ntracers_matrix` to map tracer-pair indices into contiguous blocks and supports a flag that drops cross-spectra.
- `cov_rsd_value(...)` evaluates the RSD covariance kernel using Wigner 3j symbols (from `sympy.physics.wigner`). It sums over coupling coefficients between multipoles according to perturbation-theory expressions.
- `get_covariance_rsd(...)` constructs the full multipole covariance tensor, iterating over tracer pairs and multipole combinations, flattening the result to match the data vector ordering used by `Model`.

### fisher.py
- Provides `fisher(model, ...)`, which consumes a preconfigured `Model` instance and returns the inverted Fisher matrix.
- Workflow inside `fisher`:
  - Requests derivatives from the model (optionally disabling analytic bias derivatives).
  - Builds the covariance through `covariance.get_covariance` (supports RSD, cross-spectrum exclusion, asymmetric k-cuts).
  - Inverts the covariance, checks positive-definiteness, and accumulates the Fisher matrix by summing over all spectra, multipoles, and k-bins with the usual `dP_i C^{-1} dP_j` contraction.
  - Adds diagonal priors based on the active parameter metadata, verifies Fisher positive-definiteness, and reports marginalized parameter uncertainties.
- The script footer demonstrates usage by instantiating a single-tracer model via `modellib.set_model(1)` and calling `fisher(model)` when the module is executed directly.

## Typical Usage Pattern
- Adjust fiducial settings either by editing `fiducial_cosmo.py` or by calling `modellib.import_and_overwrite_globals` / `modelclass.import_and_overwrite_globals` with a user-defined Python file.
- Build a survey configuration with `modellib.set_model(...)`, choosing the number of tracers, whether to include RSD multipoles, and other physical toggles.
- Use the resulting `Model` instance to generate spectra (`Model.Pdata`) or derivative information (`Model.get_derivatives`).
- Assemble covariances and Fisher forecasts via `fisher.fisher(model, ...)`, optionally customizing covariance options and priors.

## Notes
- The CLASS-PT directory supplies the perturbative kernels consumed by `Model` but is treated as an opaque external dependency in this summary.
- Several routines print diagnostic information (e.g., spectra slices, covariance shapes) during execution; these are useful for debugging but may be noisy in production runs.
