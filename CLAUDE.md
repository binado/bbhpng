# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fisher-forecast analysis pipeline for Black Hole Binary (BBH) parameter inference using primordial non-Gaussianity (PNG). The codebase builds perturbation-theory power spectrum models for multiple tracers and computes Fisher information matrices to forecast constraints on cosmological parameters (e.g., f_NL, ω_cdm, h, A_s) from BBH observations.

**Key Domain**: Cosmology/perturbation theory, Fisher forecasting, precision cosmology with gravitational waves.

## Architecture Overview

The codebase follows a modular workflow:

1. **Fiducial Setup** (`fiducial_cosmo.py`): Defines default cosmological parameters, bias parameters, and counterterms
2. **Model Construction** (`modellib.py`): Orchestrates building parameter dictionaries and Model instances
3. **Core Model** (`modelclass.py`): Main `Model` class that:
   - Interfaces with CLASS (Cosmic Linear Anisotropy Solving System) for cosmological perturbations
   - Evaluates multipole power spectra using 1-loop perturbation theory
   - Computes analytical derivatives (via SymPy) for bias/stochastic parameters
   - Handles numerical derivatives for cosmological parameters
   - Manages caching of precomputed CLASS outputs
4. **Covariance** (`covariance.py`): Builds Gaussian covariance matrices for real-space or RSD multipoles
5. **Fisher Analysis** (`fisher.py`): Computes Fisher information matrix and marginalized parameter uncertainties
6. **Configuration** (`config.py`): TOML-based configuration system with hierarchical key access

## Key Concepts

### Parameters
- **Cosmological**: ω_cdm, h, A_s, f_NL (PNG amplitude)
- **Bias**: b1, b2, bG2, bGamma3, bNabla2 (tracer bias coefficients)
- **Counterterms**: cct_2_0, cct_2_2, cct_2_4, cct_4_4, cct_4_6
- **Stochastic**: Shot noise and other nuisance parameters

### Power Spectrum Model
- Real-space: Combines linear + 1-loop terms with bias contributions
- RSD: Full redshift-space multipole decomposition (monopole, quadrupole, hexadecapole)
- Nonlinearity: Optional 1-loop corrections via CLASS-PT backend
- PNG: Coupling through f_NL parameter (local or other shapes)

### External Dependencies
- **CLASS**: Cosmological background/perturbations (via `classy` Python API)
- **CLASS-PT**: Perturbation-theory kernels (external directory, opaque to this codebase)
- **SymPy**: Symbolic differentiation for analytical derivatives
- **NumPy/SciPy**: Numerics (matrix inversion, finite differences)

## Configuration

All runtime settings are read from `config.toml` (default) or specified via CLI flag `-c/--config-file`.

**Key sections**:
- `[tracers]`: Number of tracers, noise, volume, shot noise (`nbar_value`)
- `[bias]`: Fiducial bias parameters for each tracer; `use_universality_relation` flag
- `[cosmology]`: Fiducial cosmological parameters; `includeAs` toggles A_s as an active parameter
- `[power_spectrum]`: RSD toggle, multipole selection, real-space vs RSD, active parameter masks
- `[caching]`: Whether to load/save precomputed CLASS outputs

To override defaults programmatically, use `modellib.import_and_overwrite_globals(filepath)` to load a Python file that redefines module-level constants.

## Common Development Tasks

### Running the Fisher Analysis
```bash
python fisher.py
```
This loads `config.toml` and performs a single-tracer Fisher forecast. Output includes Fisher matrix shape, parameter uncertainties, and intermediate diagnostics.

### Testing with Custom Configuration
```bash
python fisher.py -c custom_config.toml
```

### Working with Multi-Tracer Models
Use `modellib.set_model(ntracers, ...)` to instantiate models with multiple tracers. The function handles parameter bookkeeping and tracer-pair indexing.

### Computing Derivatives
```python
model = modellib.set_model(ntracers=1)
derivatives = model.get_derivatives(analytic_bias=True)
```
Returns shape `(len_param, len_k * ntracers_pairs * nmultipoles)`.

### Inspecting Spectra
```python
model = modellib.set_model(ntracers=1)
spectra = model.Pdata
```
Access fiducial power spectra and diagnostic plots via model attributes.

### Precomputing CLASS Outputs
Set `save_precomputed_files = true` in config to cache CLASS evaluations. These are stored in `savedspec/` as pickles and reused for faster iterations.

## Important Implementation Details

### Numerical Precision
- k-grid: Hard-coded array in `modelclass.py` masked to `[kmin, kmax]`
- Finite-difference step: `eps=0.04` (default, adjustable)
- Fourth-order finite difference for cosmological derivatives
- Covariance inversion: Checked for positive-definiteness

### Tracer Indexing
- Single-tracer cross-spectra indexed as `(0, 0)` (auto-spectrum)
- Multi-tracer models use `ntracerssum = n(n+1)/2` to store upper triangular pairs
- Flattening order matters: spectra are flattened as `(tracer_pairs, k_bins, multipoles)`

### Analytic vs Numerical Derivatives
- **Analytic**: Bias/stochastic parameters differentiated via SymPy expressions
- **Numerical**: Cosmological parameters (ω_cdm, h, A_s, f_NL) use fourth-order finite differences
- `model.get_derivatives(analytic_bias=False)` switches to numerical for all parameters

### RSD Covariance
- Coupling between multipoles via Wigner 3j symbols
- `cov_rsd_value()` evaluates kernels; results flattened to match model data vector ordering

## Code Style and Maintenance

- **Linting**: Code has been formatted with ruff; run `ruff check` to verify
- **Documentation**: AGENTS.md provides detailed module breakdown; refer to it for function signatures and workflows
- **Diagnostics**: Many routines print intermediate state (spectra shapes, covariance condition numbers, etc.); these are useful for debugging but can be verbose

## Debugging Tips

- **Model initialization fails**: Check `config.toml` parameter lists (e.g., `b1set`, `b2set`) have correct length
- **Fisher matrix singular**: Verify positive-definiteness of covariance; check if active parameters are too many or k-range too narrow
- **Derivative mismatch**: Compare analytic vs numerical derivatives with `analytic_bias=True/False` to isolate issues
- **CLASS errors**: Ensure cosmological parameters are physical (e.g., A_s > 0); check CLASS-PT availability if nonlin=true

## Data and Outputs

- **Redshift data**: `redshifts_COBA_BBH_1yr.npy` (BBH redshift samples)
- **Cached spectra**: `savedspec/` directory (pickled CLASS/PT outputs)
- **Volume/source counts**: Computed from BBH 1-year catalogs; see `compute_volume.py` and `get_redshift_bins.py`
- **Test files**: `test_v*.txt` contain diagnostic output from development iterations
