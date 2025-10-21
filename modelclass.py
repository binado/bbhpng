import pickle

import classy
import numpy as np
import sympy as sp

from fiducial_cosmo import *


def import_and_overwrite_globals(filepath):
    """
    Imports a Python file and overwrites all existing global variables with the ones from the file.

    Parameters:
    filepath (str): Path to the Python file to be imported.
    """
    #    print ("Modelclass:", CCT_2_2_VALUE)
    global_vars = globals()

    # Read the file contents
    with open(filepath, "r") as f:
        file_contents = f.read()

    # Execute the file contents in the global scope
    exec(file_contents, global_vars)


#    print ("Modelclass:", CCT_2_2_VALUE)


class Model:
    def __init__(
        self,
        cosmo_parameters,
        bias_parameters,
        stoch_parameters,
        tracers_shot_noise,
        kmin,
        kmax,
        Vbox,
        z,
        p1loop=1.0,
        noise_amplitude=0,
        eps=0.04,
        RSD=False,
        multipoles=[True, True, True],
        open_precomputed_files=True,
        save_precomputed_files=False,
    ):
        """
        Initialize the Model class with key cosmological, bias, stochastic, and configuration parameters.

        Parameters:
        - cosmo_parameters: Dictionary of cosmological parameters.
        - bias_parameters: List of bias parameter dictionaries for tracers.
        - stoch_parameters: List of stochastic parameter dictionaries for tracers.
        - tracers_shot_noise: List of shot noise values for tracers.
        - kmin, kmax: Minimum and maximum wavenumbers.
        - Vbox: Survey volume.
        - z: Redshift of the survey.
        - h_fid: Fiducial Hubble constant.
        - omega_cdm_fid: Fiducial cold dark matter density.
        - p1loop: Non-linear correction term.
        - noise_amplitude: Amplitude for noise modeling.
        - eps: Step size for numerical derivatives.
        """

        # Initialize key attributes
        self.cosmo_parameters = cosmo_parameters
        self.bias_parameters = bias_parameters
        self.stoch_parameters = stoch_parameters
        self.ntracers = len(self.bias_parameters)
        self.ntracerssum = int(self.ntracers * (self.ntracers + 1) / 2)
        self.RSD = RSD

        self.open_precomputed_files = open_precomputed_files
        self.save_precomputed_files = save_precomputed_files

        if not self.RSD:
            self.nbias = 6
            self.nstoch = 3
        else:
            self.nbias = 9
            self.nstoch = 4

        # Precompute tracer matrix indices
        self.ntracers_matrix = self._initialize_tracer_matrix()

        # Aggregate all parameters and active parameters
        self._initialize_parameters()

        # Cache cosmological parameter ordering for later slicing
        self.cosmo_names = list(self.cosmo_parameters.keys())
        self.n_cosmo = len(self.cosmo_names)

        # Setup basic attributes
        self.kmin = kmin
        self.kmax = kmax
        self.knorm = 0.1
        self.nbarsample = tracers_shot_noise
        self.nbartotal = np.sum(tracers_shot_noise)
        self.Vbox = Vbox
        self.multipoles = multipoles

        # Compute wavenumber and power spectrum setup
        self.set_ks()

        # Store fiducial cosmological parameters
        self.z = z
        self.omega_cdm_fid = self.cosmo_parameters["omegacdm"]["value"]
        self.h_fid = self.cosmo_parameters["h"]["value"]
        self.As_fid = self.cosmo_parameters["As"]["value"]
        self.f_NL_fid = self.cosmo_parameters["f_NL"]["value"]

        # Additional model parameters
        self.p1loop = p1loop
        self.noise_amplitude = noise_amplitude
        self.eps = eps

        # Compute fiducial cosmological functions
        self.pk_mult_fid, self.knormh_fid, self.fz = self.compute_cosmo_func(
            self.omega_cdm_fid,
            self.h_fid,
            self.As_fid,
            openfile=self.open_precomputed_files,
            savefile=self.save_precomputed_files,
        )
        # print ("pkmult", len(self.pk_mult_fid))
        # exit(1)

        # Initialize model data
        self.Pdata = self.get_data(
            self.pk_mult_fid, self.fz, self.all_parameters_values
        )
        print("Spectrum 0: (AA): ", self.Pdata[0 * self.len_k : 1 * self.len_k])
        print("Spectrum 2: (AA): ", self.Pdata[1 * self.len_k : 2 * self.len_k])
        print("Spectrum 4: (AA): ", self.Pdata[2 * self.len_k : 3 * self.len_k])
        print("Spectrum 0: (AB): ", self.Pdata[3 * self.len_k : 4 * self.len_k])
        print("Spectrum 2: (AB)  ", self.Pdata[4 * self.len_k : 5 * self.len_k])
        print("Spectrum 4: (AB)  ", self.Pdata[5 * self.len_k : 6 * self.len_k])
        print("Spectrum 0: (BB): ", self.Pdata[6 * self.len_k : 7 * self.len_k])
        print("Spectrum 2: (BB): ", self.Pdata[7 * self.len_k : 8 * self.len_k])
        print("Spectrum 4: (BB): ", self.Pdata[8 * self.len_k : 9 * self.len_k])
        print(self.len_k, len(self.Pdata))

        # Initialize analytic deriv
        if not self.RSD:
            self.analytic_deriv = self.get_analytic_deriv()
        else:
            self.analytic_deriv = self.get_analytic_deriv_rsd()

    def _initialize_tracer_matrix(self):
        """
        Create a matrix for pairwise tracer combinations for efficient indexing.

        Returns:
        - ntracers_matrix: 2D array mapping tracer pairs to unique indices.
        """
        ntracers_matrix = np.zeros((self.ntracers, self.ntracers), dtype=int)
        count = 0
        for i in range(self.ntracers):
            for j in range(i, self.ntracers):
                ntracers_matrix[i, j] = count
                ntracers_matrix[j, i] = count
                count += 1
        return ntracers_matrix

    def _initialize_parameters(self):
        """
        Aggregate all model parameters, their values, and active parameters.
        """
        self.all_parameters = [props for props in self.cosmo_parameters.values()]
        self.all_parameters_values = [
            props["value"] for props in self.cosmo_parameters.values()
        ]
        self.active_parameters = [
            props
            for props in self.cosmo_parameters.values()
            if props.get("active", False)
        ]
        self.active_priors = [
            props["priorsigma"]
            for props in self.cosmo_parameters.values()
            if props.get("active", False)
        ]

        # Aggregate bias and stochastic parameters
        for tracer_i in self.bias_parameters:
            temp_param = [props for props in tracer_i.values()]
            temp_value = [props["value"] for props in tracer_i.values()]
            temp_active = [
                props for props in tracer_i.values() if props.get("active", False)
            ]
            temp_prior = [
                props["priorsigma"]
                for props in tracer_i.values()
                if props.get("active", False)
            ]
            self.all_parameters = np.concatenate((self.all_parameters, temp_param))
            self.all_parameters_values = np.concatenate(
                (self.all_parameters_values, temp_value)
            )
            self.active_parameters = np.concatenate(
                (self.active_parameters, temp_active)
            )
            self.active_priors = np.concatenate((self.active_priors, temp_prior))
        for tracer_i in self.stoch_parameters:
            temp_param = [props for props in tracer_i.values()]
            temp_value = [props["value"] for props in tracer_i.values()]
            temp_active = [
                props for props in tracer_i.values() if props.get("active", False)
            ]
            temp_prior = [
                props["priorsigma"]
                for props in tracer_i.values()
                if props.get("active", False)
            ]
            self.all_parameters = np.concatenate((self.all_parameters, temp_param))
            self.all_parameters_values = np.concatenate(
                (self.all_parameters_values, temp_value)
            )
            self.active_parameters = np.concatenate(
                (self.active_parameters, temp_active)
            )
            self.active_priors = np.concatenate((self.active_priors, temp_prior))

        # Parameter counts
        self.len_param = len(self.active_parameters)
        self.len_total = len(self.all_parameters)

    def set_ks(self):
        """
        Compute wavenumber arrays and their corresponding weights.
        """
        k = np.array(
            [
                0.00436332,
                0.00690733,
                0.01191144,
                0.01681231,
                0.02110734,
                0.02586348,
                0.03066663,
                0.0351514,
                0.03980047,
                0.04462564,
                0.04929755,
                0.05408208,
                0.05878054,
                0.06337915,
                0.06796408,
                0.07265316,
                0.0773745,
                0.08206283,
                0.08669998,
                0.09151402,
                0.09614883,
                0.10081397,
                0.10552219,
                0.11013738,
                0.11481222,
                0.11955234,
                0.12421079,
                0.12884013,
                0.13356501,
                0.13828716,
                0.14306583,
                0.14776735,
                0.15239276,
                0.15710744,
                0.16179708,
                0.16642278,
                0.17110649,
                0.17579253,
                0.18052566,
                0.18520921,
                0.18987729,
                0.19453861,
                0.19923207,
                0.20393509,
                0.20863545,
                0.21333136,
                0.21805128,
                0.2227085,
                0.22738634,
                0.23205931,
                0.2367079,
                0.24136284,
                0.24605665,
                0.25077188,
                0.25547436,
                0.26018536,
                0.26486027,
                0.26955128,
                0.2742491,
                0.27894068,
                0.2836418,
                0.2883313,
                0.29297578,
                0.2976653,
            ]
        )

        k = np.array(k, dtype=np.float64)
        deltak = np.diff(k)
        deltak = np.append(deltak, deltak[-1])  # Ensure same length as k
        Nk = k**2 * deltak * self.Vbox / (2.0 * np.pi**2.0)
        mask = (k <= self.kmax) & (self.kmin <= k)
        self.k_loc = k[mask]
        self.k_loc_unmasked = k
        self.Nk_loc = Nk[mask]
        self.len_k = len(self.k_loc)
        self.mask = mask

    # print (mask.shape, k.shape, self.k_loc.shape)
    # exit(1)

    def get_data(self, pk_mult, fz, args):  # both ST and MT
        """
        Get model-specific data
        """

        # get bias and setoch set
        _, h, f_NL, bias_set_full, stoch_set_full = self.get_cosmo_bias_stoch_set(args)

        # Initialize list to store spectra
        spectras = []

        # Loop over tracer pairs
        count = 0
        for tracer_A in range(self.ntracers):
            for tracer_B in range(tracer_A, self.ntracers):
                bias_setA = bias_set_full[tracer_A]
                bias_setB = bias_set_full[tracer_B]
                stoch_setAB = stoch_set_full[count]
                count += 1

                if not self.RSD:
                    spectrum = self.spec_model_nlin(
                        pk_mult,
                        h,
                        fz,
                        bias_setA,
                        bias_setB,
                        stoch_setAB,
                        self.nbarsample[tracer_A],
                        self.nbarsample[tracer_B],
                        f_NL,
                    )
                else:
                    spectrum = self.spec_model_nlin_rsd(
                        pk_mult,
                        h,
                        fz,
                        bias_setA,
                        bias_setB,
                        stoch_setAB,
                        self.nbarsample[tracer_A],
                        self.nbarsample[tracer_B],
                    )

                spectras.append(spectrum)
        if count != len(stoch_set_full):
            print("Something wrong with Stochs", count, len(stoch_set_full))
            exit(1)

        # Concatenate the spectra and return the result
        return np.concatenate(spectras)

    def get_cosmo_bias_stoch_set(self, args, FLAG_NOCROSS_SPEC=False):
        """
        Initialize model-specific data such as the fiducial spectrum and tracer-specific values.
        """

        cosmo_values = args[: self.n_cosmo]
        cosmo_map = dict(zip(self.cosmo_names, cosmo_values))

        omega_cdm = cosmo_map.get("omegacdm", self.omega_cdm_fid)
        h = cosmo_map.get("h", self.h_fid)
        f_NL = cosmo_map.get("f_NL", self.f_NL_fid)

        bias_set = []
        bias_start = self.n_cosmo
        for idx_tracers in range(self.ntracers):
            start = bias_start + idx_tracers * self.nbias
            end = bias_start + (idx_tracers + 1) * self.nbias
            bias_set.append(args[start:end])

        if FLAG_NOCROSS_SPEC:
            idxmax = self.ntracers
        else:
            idxmax = self.ntracerssum

        stochastic_set = []
        stoch_start = bias_start + self.ntracers * self.nbias
        for idx_tracers in range(idxmax):
            start = stoch_start + idx_tracers * self.nstoch
            end = stoch_start + (idx_tracers + 1) * self.nstoch
            stochastic_set.append(args[start:end])

        return omega_cdm, h, f_NL, bias_set, stochastic_set

    def get_derivatives(self, analytic_bias=True, FLAG_NOCROSS_SPEC=False):
        """
        Compute derivatives of the model with respect to all active parameters.

        Returns:
        - derivatives: List of derivatives for active parameters.
        """
        function = self.get_data
        derivatives = []
        active_idx = 0

        for param_idx in range(self.len_total):
            # print ("deriv calc:", param_idx)
            param = self.all_parameters[param_idx]
            if param.get("active", False):
                if param.get("type") == "cosmo":
                    # Numerical derivatives for cosmological parameters
                    numerics = self.numerical_derivative(
                        function, active_idx, param_idx, self.eps
                    )
                    derivatives.append(numerics)
                else:
                    # Analytical derivatives for bias and stochastic parameters
                    if analytic_bias:
                        analytics = self.analytical_derivative(function, param_idx)
                        # numerics = self.numerical_derivative(function, active_idx, param_idx, self.eps)
                        # for ii in range(len(analytics)):
                        #  if ( ((analytics[ii] - numerics[ii]))**2.>1e-6 and numerics[ii] > 1e-6):
                        #    numerics = self.numerical_derivative(function, active_idx, param_idx, self.eps, debug = True)
                        #    print ('something weird!! ', analytics[ii], numerics[ii], param['name'], ii)
                        derivatives.append(analytics)
                    else:
                        numerics = self.numerical_derivative(
                            function, active_idx, param_idx, self.eps
                        )
                        # print(numerics.shape)
                        # print(param_idx, numerics, len(numerics))
                        derivatives.append(numerics)
                active_idx += 1

        # Ensure all active parameters were processed
        if active_idx != self.len_param:
            print("Error: Mismatch in active parameter count!")
            exit(1)

        if FLAG_NOCROSS_SPEC:
            new_derivatives = []
            nmultipoles = sum(self.multipoles)
            for i in range(self.len_param):
                #                temp = np.zeros((self.ntracers * self.len_k * np.sum(self.multipoles)))
                all_tracers = []
                for idx1 in range(self.ntracers):
                    idx12 = self.ntracers_matrix[idx1][idx1]
                    temp = derivatives[i][
                        idx12 * nmultipoles * self.len_k : (idx12 + 1)
                        * nmultipoles
                        * self.len_k
                    ]
                    # for idxk in range(self.len_k):
                    #    temp[idx1*self.len_k+idxk] = derivatives[i][idx12*self.len_k+idxk]
                    all_tracers.append(temp)
                all_tracers = np.array(all_tracers).flatten()
                new_derivatives.append(all_tracers)
            derivatives = new_derivatives

        return derivatives

    def spec_model_nlin(
        self,
        pk_mult_loc,
        h_loc,
        fz,
        bias_setA,
        bias_setB,
        stoch_set,
        nbarA_loc,
        nbarB_loc,
        f_NL,
    ):
        """
        Compute the model spectrum using tracer parameters.

        Parameters:
        - pk_mult_loc: Multipole power spectrum array.
        - h_loc: Hubble constant.
        - fz: Not used, just in rsd
        - bias_setA, bias_setB: Bias parameter sets for tracers A and B.
        - stoch_set: Stochastic parameter set.
        - nbarA_loc, nbarB_loc: Shot noise values for tracers A and B.
        - f_NL: value of f_NL

        Returns:
        - Non-linear model spectrum.

        Notes:
            Code currently computes f_NL contributions only up to linear order!
        """
        # Unpack bias parameters
        b1_A, b2_A, bG2_A, bGamma3_A, bNabla2_A, bphi_A = bias_setA
        b1_B, b2_B, bG2_B, bGamma3_B, bNabla2_B, bphi_B = bias_setB

        # Unpack stochastic parameters
        cshot, c0, c2 = stoch_set

        nbarmix = np.sqrt(nbarA_loc * nbarB_loc)
        knormh_loc = self.knorm * h_loc

        # Compute the non-linear spectrum
        result = (
            b1_A * b1_B * (pk_mult_loc[14] + self.p1loop * pk_mult_loc[0])
            + (b1_A * bphi_B + b1_B * bphi_A) * f_NL * pk_mult_loc[53]
            + self.p1loop
            * (bNabla2_A * b1_B + bNabla2_B * b1_A)
            * pk_mult_loc[10]
            / knormh_loc**2
            + self.p1loop * 0.5 * (b1_A * b2_B + b1_B * b2_A) * pk_mult_loc[2]
            + self.p1loop * 0.25 * b2_A * b2_B * pk_mult_loc[1]
            + self.p1loop * (b1_A * bG2_B + b1_B * bG2_A) * pk_mult_loc[3]
            + self.p1loop
            * (b1_A * (bG2_B + 0.4 * bGamma3_B) + b1_B * (bG2_A + 0.4 * bGamma3_A))
            * pk_mult_loc[6]
            + self.p1loop * bG2_A * bG2_B * pk_mult_loc[5]
            + self.p1loop * 0.5 * (b2_A * bG2_B + b2_B * bG2_A) * pk_mult_loc[4]
        ) * h_loc**3

        # Add stochastic term
        stochastic_term = (cshot + c0 + c2 * self.k_loc**2 / self.knorm**2) / nbarmix

        # Add noise if enabled
        noise_array = np.array(
            [
                1.0
                + (
                    np.random.normal(0, self.noise_amplitude, 1)[0]
                    if self.noise_amplitude
                    else 0.0
                )
                for _ in range(len(stochastic_term))
            ]
        )

        return result * noise_array + stochastic_term

    def spec_model_nlin_rsd(
        self,
        pk_mult_loc,
        h_loc,
        fz,
        bias_setA,
        bias_setB,
        stoch_set,
        nbarA_loc,
        nbarB_loc,
    ):
        """
        Compute the non-linear redshift-space model spectrum using tracer parameters.

        Parameters:
        - pk_mult_loc: Multipole power spectrum array.
        - h_loc: Hubble constant.
        - fz: Growth rate of structure.
        - bias_setA, bias_setB: Bias parameter sets for tracers A and B.
        - stoch_set: Stochastic parameter set.
        - nbarA_loc, nbarB_loc: Shot noise values for tracers A and B.

        Returns:
        - Non-linear model spectrum.
        """
        # Unpack bias parameters for tracers A and B
        (
            b1_A,
            b2_A,
            bG2_A,
            bGamma3_A,
            cct_2_0_A,
            cct_2_2_A,
            cct_2_4_A,
            cct_4_4_A,
            cct_4_6_A,
        ) = bias_setA
        (
            b1_B,
            b2_B,
            bG2_B,
            bGamma3_B,
            cct_2_0_B,
            cct_2_2_B,
            cct_2_4_B,
            cct_4_4_B,
            cct_4_6_B,
        ) = bias_setB
        #        print ("\n\n\n values for A, B:", cct_2_2_B, cct_2_2_A )
        cct_2_6_A, cct_2_6_B = 0, 0

        # Unpack stochastic parameters
        cshot, cst_0_0, cst_0_2, cst_2_2 = stoch_set

        # Precompute common factors
        nbarmix = np.sqrt(nbarA_loc * nbarB_loc)
        knormh_loc = self.knorm * h_loc
        k2ratio = self.k_loc**2 / self.knorm**2
        # print (k2ratio, pk_mult_loc[14])
        pk_mult14_scaled = (k2ratio * pk_mult_loc[14]) * h_loc**3

        # Compute counter-terms for tracers A and B
        Pct_0_AB = self.p1loop * (
            +(
                35.0 * (fz + 3.0 * b1_A) * cct_2_0_B
                + 7.0 * (3.0 * fz + 5.0 * b1_A) * cct_2_2_B
                + 3.0 * (5.0 * fz + 7.0 * b1_A) * (cct_2_4_B + k2ratio * cct_4_4_B)
                + (35.0 / 21.0) * (7.0 * fz + b1_A) * (cct_2_6_B + k2ratio * cct_4_6_B)
            )
            * (1.0 / 105.0)
            * pk_mult14_scaled
        )

        Pct_2_AB = self.p1loop * (
            +(
                7.0 * fz * cct_2_0_B
                + (6.0 * fz + 7.0 * b1_A) * cct_2_2_B
                + (5.0 * fz + 6.0 * b1_A) * (cct_2_4_B + k2ratio * cct_4_4_B)
                + (20.0 / 33.0)
                * (28.0 * fz + 33.0 * b1_A)
                * (cct_2_6_B + k2ratio * cct_4_6_B)
            )
            * (2.0 / 21.0)
            * pk_mult14_scaled
        )

        Pct_4_AB = self.p1loop * (
            +(
                11.0 * fz * cct_2_2_B
                + (15.0 * fz + 11.0 * b1_A) * (cct_2_4_B + k2ratio * cct_4_4_B)
                + (15.0 / 13.0)
                * (14.0 * fz + 13.0 * b1_A)
                * (cct_2_6_B + k2ratio * cct_4_6_B)
            )
            * (8.0 / 385.0)
            * pk_mult14_scaled
        )

        Pct_0_BA = self.p1loop * (
            +(
                35.0 * (fz + 3.0 * b1_B) * cct_2_0_A
                + 7.0 * (3.0 * fz + 5.0 * b1_B) * cct_2_2_A
                + 3.0 * (5.0 * fz + 7.0 * b1_B) * (cct_2_4_A + k2ratio * cct_4_4_A)
                + (35.0 / 21.0) * (7.0 * fz + b1_B) * (cct_2_6_A + k2ratio * cct_4_6_A)
            )
            * (1.0 / 105.0)
            * pk_mult14_scaled
        )

        Pct_2_BA = self.p1loop * (
            +(
                7.0 * fz * cct_2_0_A
                + (6.0 * fz + 7.0 * b1_B) * cct_2_2_A
                + (5.0 * fz + 6.0 * b1_B) * (cct_2_4_A + k2ratio * cct_4_4_A)
                + (20.0 / 33.0)
                * (28.0 * fz + 33.0 * b1_B)
                * (cct_2_6_A + k2ratio * cct_4_6_A)
            )
            * (2.0 / 21.0)
            * pk_mult14_scaled
        )

        Pct_4_BA = self.p1loop * (
            +(
                11.0 * fz * cct_2_2_A
                + (15.0 * fz + 11.0 * b1_B) * (cct_2_4_A + k2ratio * cct_4_4_A)
                + (15.0 / 13.0)
                * (14.0 * fz + 13.0 * b1_B)
                * (cct_2_6_A + k2ratio * cct_4_6_A)
            )
            * (8.0 / 385.0)
            * pk_mult14_scaled
        )

        # Add stochastic terms
        Pst_0 = (
            cshot
            + cst_0_0
            + self.p1loop * cst_0_2 * k2ratio
            + self.p1loop * cst_2_2 * fz * k2ratio / 3.0
        ) / nbarmix
        Pst_2 = self.p1loop * (2.0 * cst_2_2 * fz * k2ratio / 3.0) / nbarmix
        Pst_4 = 0.0

        # Compute multipoles
        monopole = (
            (
                pk_mult_loc[15]
                + self.p1loop * pk_mult_loc[21]
                + (b1_A + b1_B)
                / 2.0
                * (pk_mult_loc[16] + self.p1loop * pk_mult_loc[22])
                + b1_A * b1_B * (pk_mult_loc[17] + self.p1loop * pk_mult_loc[23])
                + self.p1loop * 0.25 * b2_A * b2_B * pk_mult_loc[1]
                + self.p1loop * (b1_A * b2_B + b1_B * b2_A) / 2.0 * pk_mult_loc[30]
                + self.p1loop * (b2_A + b2_B) / 2.0 * pk_mult_loc[31]
                + self.p1loop * (b1_A * bG2_B + b1_B * bG2_A) / 2.0 * pk_mult_loc[32]
                + self.p1loop * (bG2_A + bG2_B) / 2.0 * pk_mult_loc[33]
                + self.p1loop * (b2_A * bG2_B + b2_B * bG2_A) / 2.0 * pk_mult_loc[4]
                + self.p1loop * bG2_A * bG2_B * pk_mult_loc[5]
                + self.p1loop
                * (bG2_A + 0.4 * bGamma3_A)
                * (b1_B * pk_mult_loc[7] + pk_mult_loc[8])
                + self.p1loop
                * (bG2_B + 0.4 * bGamma3_B)
                * (b1_A * pk_mult_loc[7] + pk_mult_loc[8])
            )
            * h_loc**3
            + Pct_0_AB
            + Pct_0_BA
            + Pst_0
        )
        #        print ("15,16,17: ", pk_mult_loc[15] + self.p1loop*pk_mult_loc[21]
        #            , (b1_A + b1_B) / 2. * (pk_mult_loc[16] + self.p1loop*pk_mult_loc[22])
        #            , b1_A * b1_B * (pk_mult_loc[17] + self.p1loop*pk_mult_loc[23]))
        #        exit(1)

        quadrupole = (
            (
                pk_mult_loc[18]
                + self.p1loop * pk_mult_loc[24]
                + (b1_A + b1_B)
                / 2.0
                * (pk_mult_loc[19] + self.p1loop * pk_mult_loc[25])
                + self.p1loop * b1_A * b1_B * pk_mult_loc[26]
                + self.p1loop * (b1_A * b2_B + b1_B * b2_A) / 2.0 * pk_mult_loc[34]
                + self.p1loop * (b2_A + b2_B) / 2.0 * pk_mult_loc[35]
                + self.p1loop * (b1_A * bG2_B + b1_B * bG2_A) / 2.0 * pk_mult_loc[36]
                + self.p1loop * (bG2_A + bG2_B) / 2.0 * pk_mult_loc[37]
                + self.p1loop * (bG2_A + 0.4 * bGamma3_A) * pk_mult_loc[9]
                + self.p1loop * (bG2_B + 0.4 * bGamma3_B) * pk_mult_loc[9]
            )
            * h_loc**3
            + Pct_2_AB
            + Pct_2_BA
            + Pst_2
        )

        hexadecapole = (
            (
                pk_mult_loc[20]
                + self.p1loop * pk_mult_loc[27]
                + self.p1loop * (b1_A + b1_B) / 2.0 * pk_mult_loc[28]
                + self.p1loop * b1_A * b1_B * pk_mult_loc[29]
                + self.p1loop * (b2_A + b2_B) / 2 * pk_mult_loc[38]
                + self.p1loop * (bG2_A + bG2_B) / 2.0 * pk_mult_loc[39]
            )
            * h_loc**3
            + Pct_4_AB
            + Pct_4_BA
            + Pst_4
        )

        if not self.multipoles[0]:
            monopole = []
            # monopole = [0] * self.len_k
        if not self.multipoles[1]:
            quadrupole = []
        if not self.multipoles[2]:
            hexadecapole = []

        # Concatenate results
        # result = np.concatenate([1e-6*monopole, 1e-6*quadrupole, hexadecapole])
        result = np.concatenate([monopole, quadrupole, hexadecapole])
        if np.isnan(result).any():
            print("ERROR!!")
            exit(1)

        # Add noise if enabled
        if self.noise_amplitude:
            noise_array = 1 + np.random.normal(0, self.noise_amplitude, len(result))
            result *= noise_array

        return result

    #    def compute_cosmo_func(self, omega_cdm, h_local, As_local, openfile = False, savefile = True):
    def compute_cosmo_func(
        self, omega_cdm, h_local, As_local, openfile=True, savefile=False
    ):
        """
        Compute the cosmological multipole power spectrum using CLASS.

        Parameters:
        - omega_cdm: Cold dark matter density parameter.
        - h_local: Local Hubble constant.

        Returns:
        - pk_mult_local: Array of multipole power spectra.
        - knormh: Normalized wavenumber.
        """

        def get_filename(omega_cdm, h_local, As_local):
            filename_return = "./savedspec/"
            if As_local == AS_FID:
                filename_return += "As_0"
            elif As_local == AS_FID + 0.04 * AS_FID:
                filename_return += "As_p04"
            elif As_local == AS_FID + 0.08 * AS_FID:
                filename_return += "As_p08"
            elif As_local == AS_FID - 0.04 * AS_FID:
                filename_return += "As_m04"
            elif As_local == AS_FID - 0.08 * AS_FID:
                filename_return += "As_m08"

            elif As_local == AS_FID + 0.004 * AS_FID:
                filename_return += "As_p004"
            elif As_local == AS_FID + 0.008 * AS_FID:
                filename_return += "As_p008"
            elif As_local == AS_FID - 0.004 * AS_FID:
                filename_return += "As_m004"
            elif As_local == AS_FID - 0.008 * AS_FID:
                filename_return += "As_m008"

            else:
                print("Invalid As value: ", As_local)
                exit(1)

            filename_return += "_"

            if h_local == H_FID:
                filename_return += "h_0"
            elif h_local == H_FID + 0.04 * H_FID:
                filename_return += "h_p04"
            elif h_local == H_FID + 0.08 * H_FID:
                filename_return += "h_p08"
            elif h_local == H_FID - 0.04 * H_FID:
                filename_return += "h_m04"
            elif h_local == H_FID - 0.08 * H_FID:
                filename_return += "h_m08"

            elif h_local == H_FID + 0.004 * H_FID:
                filename_return += "h_p004"
            elif h_local == H_FID + 0.008 * H_FID:
                filename_return += "h_p008"
            elif h_local == H_FID - 0.004 * H_FID:
                filename_return += "h_m004"
            elif h_local == H_FID - 0.008 * H_FID:
                filename_return += "h_m008"

            else:
                print("Invalid h value: ", h_local)
                exit(1)

            filename_return += "_"

            if omega_cdm == OMEGA_CDM_FID:
                filename_return += "w_0"
            elif omega_cdm == OMEGA_CDM_FID + 0.04 * OMEGA_CDM_FID:
                filename_return += "w_p04"
            elif omega_cdm == OMEGA_CDM_FID + 0.08 * OMEGA_CDM_FID:
                filename_return += "w_p08"
            elif omega_cdm == OMEGA_CDM_FID - 0.04 * OMEGA_CDM_FID:
                filename_return += "w_m04"
            elif omega_cdm == OMEGA_CDM_FID - 0.08 * OMEGA_CDM_FID:
                filename_return += "w_m08"

            elif omega_cdm == OMEGA_CDM_FID + 0.004 * OMEGA_CDM_FID:
                filename_return += "w_p004"
            elif omega_cdm == OMEGA_CDM_FID + 0.008 * OMEGA_CDM_FID:
                filename_return += "w_p008"
            elif omega_cdm == OMEGA_CDM_FID - 0.004 * OMEGA_CDM_FID:
                filename_return += "w_m004"
            elif omega_cdm == OMEGA_CDM_FID - 0.008 * OMEGA_CDM_FID:
                filename_return += "w_m008"

            else:
                print("Invalid omega value: ", omega_cdm)
                exit(1)

            return filename_return

        if not openfile:
            print("Calculating cosmology")

            # Define cosmological settings
            common_settings = {
                "A_s": As_local * 1e-9,
                #'A_s': 2.100e-9,
                "n_s": 0.966,
                "tau_reio": 0.054,
                "omega_b": 0.02238,
                "omega_cdm": omega_cdm,
                "h": h_local,
                "N_ur": 3.044,
                "YHe": 0.2425,  ### Francesco asked
                "z_pk": self.z,
            }

            # Initialize CLASS cosmology object
            cosmo = classy.Class()
            cosmo.set(common_settings)

            # Set precision and output parameters
            cosmo.set(
                {
                    "output": "mPk",
                    "non linear": "PT",
                    "IR resummation": "Yes",
                    "Bias tracers": "Yes",
                    "cb": "Yes",
                    "RSD": "Yes",
                    "AP": "Yes",
                    "Omfid": "0.31",
                }
            )

            # Compute cosmology
            cosmo.compute()

            #          # Wavenumber in 1/Mpc
            #          kh_local = np.array(self.k_loc * h_local, dtype=np.float64)
            #
            #          # Initialize output
            #          cosmo.initialize_output(kh_local, self.z, len(kh_local))
            #          pk_mult_local = cosmo.get_pk_mult(kh_local, self.z, len(kh_local))

            # Wavenumber in 1/Mpc
            kh_local_unmasked = np.array(
                self.k_loc_unmasked * h_local, dtype=np.float64
            )

            # Initialize output
            cosmo.initialize_output(kh_local_unmasked, self.z, len(kh_local_unmasked))
            pk_mult_local = cosmo.get_pk_mult(
                kh_local_unmasked, self.z, len(kh_local_unmasked)
            )

            f_local = cosmo.scale_independent_growth_factor_f(self.z)

            def saveinfile(savename, variable):
                with open(savename, "wb") as f:
                    pickle.dump(variable, f)

            if savefile:
                filename = get_filename(omega_cdm, h_local, As_local)
                print("Saving cosmology at: ", filename)
                if pk_mult_local.shape != (96, 64):
                    print("Wrong shape for printing! Not the full ks")
                saveinfile(filename + "_pk_mult.dat", pk_mult_local)
                if self.z == 0:
                    saveinfile(filename + "_f0.dat", f_local)
                else:
                    print("Implement another f here!")
                    exit(1)

        else:
            filename = get_filename(omega_cdm, h_local, As_local)
            print("Opening cosmology from: ", filename, omega_cdm)
            filename = get_filename(omega_cdm, h_local, As_local)
            with open(filename + "_pk_mult.dat", "rb") as f:
                pk_mult_local = pickle.load(f)
            if self.z == 0:
                with open(filename + "_f0.dat", "rb") as f:
                    f_local = pickle.load(f)
            else:
                print("Implement another f here!")
                exit(1)

        # Normalize wavenumber
        knormh = np.array(self.knorm * h_local, dtype=np.float64)

        # print (pk_mult_local.shape, self.mask.shape, self.mask.shape)
        # print (pk_mult_local.shape, self.mask.shape, pk_mult_local[:, self.mask].shape)
        return pk_mult_local[:, self.mask], knormh, f_local

    def numerical_derivative(self, FUNC, active_idx, param_idx, eps, debug=False):
        """
        Calculate the numerical derivative of a function with respect to a parameter using a fourth-order central difference.

        Parameters:
        - FUNC: The function to differentiate.
        - active_idx: Index of the active parameter being varied.
        - param_idx: Index of the parameter in the full parameter array.
        - eps: Relative step size for finite difference.

        Returns:
        - derivative_log: Logarithmic derivative of FUNC with respect to the parameter.
        """
        # Convert all parameter values to a list for modification
        args = list(self.all_parameters_values)

        # Save the original value of the parameter
        original_value = args[param_idx]
        # if (debug == True): print(original_value, param_idx)

        # Compute relative step size
        eps_rel = eps * original_value
        if eps_rel == 0.0:  # Handle zero-value parameters (e.g., stochastic terms)
            eps_rel = eps * self.nbartotal

        # Precompute function value at the current parameter set
        f_x = self.Pdata  # Pre-stored fiducial data; avoids recomputation

        # Evaluate FUNC at shifted parameter values
        shifted_values = []
        for shift in [2, 1, -1, -2]:  # Evaluate shifts at ±2ε and ±ε
            args[param_idx] = original_value + shift * eps_rel

            # Handle cosmological vs. non-cosmological parameters
            if self.all_parameters[param_idx].get("type") == "cosmo":
                cosmo_map = dict(zip(self.cosmo_names, args[: self.n_cosmo]))
                pk_mult_loc, _, fz = self.compute_cosmo_func(
                    cosmo_map.get("omegacdm", self.omega_cdm_fid),
                    cosmo_map.get("h", self.h_fid),
                    cosmo_map.get("As", self.As_fid),
                    openfile=self.open_precomputed_files,
                    savefile=self.save_precomputed_files,
                )
            else:
                pk_mult_loc = self.pk_mult_fid
                fz = self.fz

            shifted_values.append(np.log(FUNC(pk_mult_loc, fz, args)))

        # Restore the original parameter value
        args[param_idx] = original_value

        # Calculate fourth-order central difference for the logarithmic derivative
        derivative_log = (
            (
                -shifted_values[0]
                + 8 * shifted_values[1]
                - 8 * shifted_values[2]
                + shifted_values[3]
            )
            / (12 * eps_rel)
            * f_x
        )
        # if (debug == True): print('shifted: ', f_x[32:64])
        # if (debug == True): print('shifted (2): ', shifted_values[0][32:64], shifted_values[1][32:64], shifted_values[2][32:64], shifted_values[3][32:64])

        #        if (np.isnan(derivative_log).any()):
        #          print("ERROR!! nan in deriv", shifted_values[0], len(shifted_values[0]))
        #          exit(1)

        return derivative_log

    def analytical_derivative(self, FUNC, param_idx):
        # get bias and setoch set using fiducial
        _, h, _, bias_set_full, stoch_set_full = self.get_cosmo_bias_stoch_set(
            self.all_parameters_values
        )

        # Initialize list to store spectra
        spectras = []
        if self.RSD:
            len_ell = 3
        else:
            len_ell = 1

        paramtype = self.all_parameters[param_idx].get("type")
        paramname = self.all_parameters[param_idx].get("name")

        # Loop over tracer pairs
        count = 0
        for tracer_A in range(self.ntracers):
            for tracer_B in range(tracer_A, self.ntracers):
                bias_setA = bias_set_full[tracer_A]
                bias_setB = bias_set_full[tracer_B]
                stoch_setAB = stoch_set_full[count]
                count += 1

                if paramtype == "bias":
                    inA = tracer_A in self.all_parameters[param_idx].get("traceridx")
                    inB = tracer_B in self.all_parameters[param_idx].get("traceridx")
                    if not inA and not inB:
                        spectrum = [0] * self.len_k * len_ell
                    elif not inA and inB:
                        spectrum = self.eval_derivative(
                            paramname,
                            1,
                            h,
                            bias_setA,
                            bias_setB,
                            stoch_setAB,
                            self.nbarsample[tracer_A],
                            self.nbarsample[tracer_B],
                        )
                    elif inA and not inB:
                        spectrum = self.eval_derivative(
                            paramname,
                            0,
                            h,
                            bias_setA,
                            bias_setB,
                            stoch_setAB,
                            self.nbarsample[tracer_A],
                            self.nbarsample[tracer_B],
                        )
                    elif inA and inB:
                        prefactor = 2.0 if paramtype == "bias" else 1.0
                        spectrum = prefactor * np.array(
                            self.eval_derivative(
                                paramname,
                                0,
                                h,
                                bias_setA,
                                bias_setB,
                                stoch_setAB,
                                self.nbarsample[tracer_A],
                                self.nbarsample[tracer_B],
                            )
                        )
                if paramtype == "stoch":
                    Ais0 = (
                        tracer_A == self.all_parameters[param_idx].get("traceridx")[0]
                    )
                    Bis1 = (
                        tracer_B == self.all_parameters[param_idx].get("traceridx")[1]
                    )
                    if Ais0 and Bis1:
                        prefactor = 2.0 if paramtype == "bias" else 1.0
                        spectrum = prefactor * np.array(
                            self.eval_derivative(
                                paramname,
                                0,
                                h,
                                bias_setA,
                                bias_setB,
                                stoch_setAB,
                                self.nbarsample[tracer_A],
                                self.nbarsample[tracer_B],
                            )
                        )
                    else:
                        spectrum = [0] * self.len_k * len_ell
                spectras.append(spectrum)

        if count != len(stoch_set_full):
            print("Something wrong with Stochs", count, len(stoch_set_full))
            exit(1)

        # Concatenate the spectra and return the result
        return np.concatenate(spectras)

    def get_analytic_deriv(self):
        """
        Precompute symbolic derivatives of the model with respect to bias and stochastic parameters.

        Returns:
        - derivatives: Dictionary mapping parameters to their symbolic derivatives.
        """
        # Define symbolic variables for bias and stochastic parameters
        b1_A, b1_B, b2_A, b2_B = sp.symbols("b1_A b1_B b2_A b2_B")
        bG2_A, bG2_B = sp.symbols("bG2_A bG2_B")
        bGamma3_A, bGamma3_B = sp.symbols("bGamma3_A bGamma3_B")
        bNabla2_A, bNabla2_B = sp.symbols("bNabla2_A bNabla2_B")
        c0, c2, h_loc, nbarmix, knormh_loc, knorm_loc = sp.symbols(
            "c0 c2 h_loc nbarmix knormh_loc knorm_loc"
        )
        k_loc = sp.symbols("k_loc")

        # Define symbolic multipole power spectrum terms
        pk_mult_symbolic = sp.symbols("pk_mult_symbolic0:15")

        # Construct the symbolic model expression
        expr = (
            b1_A * b1_B * (pk_mult_symbolic[14] + self.p1loop * pk_mult_symbolic[0])
            + (bNabla2_A * b1_B + bNabla2_B * b1_A)
            * pk_mult_symbolic[10]
            / knormh_loc**2
            + 0.5 * (b1_A * b2_B + b1_B * b2_A) * pk_mult_symbolic[2]
            + 0.25 * b2_A * b2_B * pk_mult_symbolic[1]
            + (b1_A * bG2_B + b1_B * bG2_A) * pk_mult_symbolic[3]
            + (b1_A * (bG2_B + 0.4 * bGamma3_B) + b1_B * (bG2_A + 0.4 * bGamma3_A))
            * pk_mult_symbolic[6]
            + bG2_A * bG2_B * pk_mult_symbolic[5]
            + 0.5 * (b2_A * bG2_B + b2_B * bG2_A) * pk_mult_symbolic[4]
        ) * h_loc**3

        # Add stochastic term
        stochastic_term = (c0 + c2 * k_loc**2 / knorm_loc**2) / nbarmix
        full_expr = expr + stochastic_term

        # Compute symbolic derivatives for each parameter
        derivatives = {}
        for param in [
            b1_A,
            b1_B,
            b2_A,
            b2_B,
            bG2_A,
            bG2_B,
            bGamma3_A,
            bGamma3_B,
            bNabla2_A,
            bNabla2_B,
            c0,
            c2,
        ]:
            derivatives[param] = sp.diff(full_expr, param)

        return derivatives

    def get_analytic_deriv_rsd(self):
        """
        Precompute symbolic derivatives of the model with respect to bias and stochastic parameters.

        Returns:
        - derivatives: Dictionary mapping parameters to their symbolic derivatives.
        """
        # Define symbolic variables for bias and stochastic parameters
        (
            b1_A,
            b2_A,
            bG2_A,
            bGamma3_A,
            cct_2_0_A,
            cct_2_2_A,
            cct_2_4_A,
            cct_4_4_A,
            cct_4_6_A,
        ) = sp.symbols(
            "b1_A b2_A bG2_A bGamma3_A cct_2_0_A cct_2_2_A cct_2_4_A cct_4_4_A cct_4_6_A"
        )
        (
            b1_B,
            b2_B,
            bG2_B,
            bGamma3_B,
            cct_2_0_B,
            cct_2_2_B,
            cct_2_4_B,
            cct_4_4_B,
            cct_4_6_B,
        ) = sp.symbols(
            "b1_B b2_B bG2_B bGamma3_B cct_2_0_B cct_2_2_B cct_2_4_B cct_4_4_B cct_4_6_B"
        )
        (
            cshot,
            cst_0_0,
            cst_0_2,
            cst_2_2,
            h_loc,
            nbarmix,
            knormh_loc,
            knorm_loc,
            k_loc,
            fz,
        ) = sp.symbols(
            "cshot cst_0_0 cst_0_2 cst_2_2 h_loc nbarmix knormh_loc knorm_loc k_loc fz"
        )
        #
        #        # Define symbolic multipole power spectrum terms
        pk_mult_symbolic = sp.symbols("pk_mult_symbolic0:40")

        # Unpack bias parameters for tracers A and B
        cct_2_6_A, cct_2_6_B = 0, 0

        # Unpack stochastic parameters

        # Precompute common factors
        # nbarmix = np.sqrt(nbarA_loc * nbarB_loc)
        knormh_loc = knorm_loc * h_loc
        k2ratio = k_loc**2 / knorm_loc**2
        pk_mult14_scaled = (k2ratio**2 * pk_mult_symbolic[14]) * h_loc**3

        # Compute counter-terms for tracers A and B
        Pct_0_AB = (
            +(
                35.0 * (fz + 3.0 * b1_A) * cct_2_0_B
                + 7.0 * (3.0 * fz + 5.0 * b1_A) * cct_2_2_B
                + 3.0 * (5.0 * fz + 7.0 * b1_A) * (cct_2_4_B + k2ratio * cct_4_4_B)
                + (35.0 / 21.0) * (7.0 * fz + b1_A) * (cct_2_6_B + k2ratio * cct_4_6_B)
            )
            * (1.0 / 105.0)
            * pk_mult14_scaled
        )

        Pct_2_AB = (
            +(
                7.0 * fz * cct_2_0_B
                + (6.0 * fz + 7.0 * b1_A) * cct_2_2_B
                + (5.0 * fz + 6.0 * b1_A) * (cct_2_4_B + k2ratio * cct_4_4_B)
                + (20.0 / 33.0)
                * (28.0 * fz + 33.0 * b1_A)
                * (cct_2_6_B + k2ratio * cct_4_6_B)
            )
            * (2.0 / 21.0)
            * pk_mult14_scaled
        )

        Pct_4_AB = (
            +(
                11.0 * fz * cct_2_2_B
                + (15.0 * fz + 11.0 * b1_A) * (cct_2_4_B + k2ratio * cct_4_4_B)
                + (15.0 / 13.0)
                * (14.0 * fz + 13.0 * b1_A)
                * (cct_2_6_B + k2ratio * cct_4_6_B)
            )
            * (8.0 / 385.0)
            * pk_mult14_scaled
        )

        Pct_0_BA = (
            +(
                35.0 * (fz + 3.0 * b1_B) * cct_2_0_A
                + 7.0 * (3.0 * fz + 5.0 * b1_B) * cct_2_2_A
                + 3.0 * (5.0 * fz + 7.0 * b1_B) * (cct_2_4_A + k2ratio * cct_4_4_A)
                + (35.0 / 21.0) * (7.0 * fz + b1_B) * (cct_2_6_A + k2ratio * cct_4_6_A)
            )
            * (1.0 / 105.0)
            * pk_mult14_scaled
        )

        Pct_2_BA = (
            +(
                7.0 * fz * cct_2_0_A
                + (6.0 * fz + 7.0 * b1_B) * cct_2_2_A
                + (5.0 * fz + 6.0 * b1_B) * (cct_2_4_A + k2ratio * cct_4_4_A)
                + (20.0 / 33.0)
                * (28.0 * fz + 33.0 * b1_B)
                * (cct_2_6_A + k2ratio * cct_4_6_A)
            )
            * (2.0 / 21.0)
            * pk_mult14_scaled
        )

        Pct_4_BA = (
            +(
                11.0 * fz * cct_2_2_A
                + (15.0 * fz + 11.0 * b1_B) * (cct_2_4_A + k2ratio * cct_4_4_A)
                + (15.0 / 13.0)
                * (14.0 * fz + 13.0 * b1_B)
                * (cct_2_6_A + k2ratio * cct_4_6_A)
            )
            * (8.0 / 385.0)
            * pk_mult14_scaled
        )

        # Add stochastic terms
        Pst_0 = (
            cshot + cst_0_0 + cst_0_2 * k2ratio + cst_2_2 * fz * k2ratio / 3.0
        ) / nbarmix
        Pst_2 = (2.0 * cst_2_2 * fz * k2ratio / 3.0) / nbarmix
        Pst_4 = 0.0

        # Compute multipoles
        monopole = (
            (
                pk_mult_symbolic[15]
                + pk_mult_symbolic[21]
                + (b1_A + b1_B) / 2.0 * (pk_mult_symbolic[16] + pk_mult_symbolic[22])
                + b1_A * b1_B * (pk_mult_symbolic[17] + pk_mult_symbolic[23])
                + 0.25 * b2_A * b2_B * pk_mult_symbolic[1]
                + (b1_A * b2_B + b1_B * b2_A) / 2.0 * pk_mult_symbolic[30]
                + (b2_A + b2_B) / 2.0 * pk_mult_symbolic[31]
                + (b1_A * bG2_B + b1_B * bG2_A) / 2.0 * pk_mult_symbolic[32]
                + (bG2_A + bG2_B) / 2.0 * pk_mult_symbolic[33]
                + (b2_A * bG2_B + b2_B * bG2_A) / 2.0 * pk_mult_symbolic[4]
                + bG2_A * bG2_B * pk_mult_symbolic[5]
                + (bG2_A + 0.4 * bGamma3_A)
                * (b1_B * pk_mult_symbolic[7] + pk_mult_symbolic[8])
                + (bG2_B + 0.4 * bGamma3_B)
                * (b1_A * pk_mult_symbolic[7] + pk_mult_symbolic[8])
            )
            * h_loc**3
            + Pct_0_AB
            + Pct_0_BA
            + Pst_0
        )

        quadrupole = (
            (
                pk_mult_symbolic[18]
                + pk_mult_symbolic[24]
                + (b1_A + b1_B) / 2.0 * (pk_mult_symbolic[19] + pk_mult_symbolic[25])
                + b1_A * b1_B * pk_mult_symbolic[26]
                + (b1_A * b2_B + b1_B * b2_A) / 2.0 * pk_mult_symbolic[34]
                + (b2_A + b2_B) / 2.0 * pk_mult_symbolic[35]
                + (b1_A * bG2_B + b1_B * bG2_A) / 2.0 * pk_mult_symbolic[36]
                + (bG2_A + bG2_B) / 2.0 * pk_mult_symbolic[37]
                + (bG2_A + 0.4 * bGamma3_A) * pk_mult_symbolic[9]
                + (bG2_B + 0.4 * bGamma3_B) * pk_mult_symbolic[9]
            )
            * h_loc**3
            + Pct_2_AB
            + Pct_2_BA
            + Pst_2
        )

        hexadecapole = (
            (
                pk_mult_symbolic[20]
                + pk_mult_symbolic[27]
                + (b1_A + b1_B) / 2.0 * pk_mult_symbolic[28]
                + b1_A * b1_B * pk_mult_symbolic[29]
                + (b2_A * b2_B) / 2 * pk_mult_symbolic[38]
                + (bG2_A + bG2_B) / 2.0 * pk_mult_symbolic[39]
            )
            * h_loc**3
            + Pct_4_AB
            + Pct_4_BA
            + Pst_4
        )

        derivatives_0 = {}
        derivatives_2 = {}
        derivatives_4 = {}
        for param in [
            b1_A,
            b2_A,
            bG2_A,
            bGamma3_A,
            cct_2_0_A,
            cct_2_2_A,
            cct_2_4_A,
            cct_4_4_A,
            cct_4_6_A,
            b1_B,
            b2_B,
            bG2_B,
            bGamma3_B,
            cct_2_0_B,
            cct_2_2_B,
            cct_2_4_B,
            cct_4_4_B,
            cct_4_6_B,
            cst_0_0,
            cst_0_2,
            cst_2_2,
        ]:
            derivatives_0[param] = sp.diff(monopole, param)
            derivatives_2[param] = sp.diff(quadrupole, param)
            derivatives_4[param] = sp.diff(hexadecapole, param)
        return [derivatives_0, derivatives_2, derivatives_4]

    def eval_derivative(
        self, paramname, AorB, h, bias_setA, bias_setB, stoch_set, nbarA_loc, nbarB_loc
    ):
        """
        Evaluate the precomputed symbolic derivatives numerically for the active parameter.

        Parameters:
        - param_idx: Index of the parameter in the full parameter array.
        - nbarsample: Shot noise sample for normalization.

        Returns:
        - result: Array of evaluated derivatives for each wavenumber.
        """
        # Precompute the fiducial power spectrum
        eval_array = self.pk_mult_fid

        if not self.RSD:
            # Unpack bias parameters
            b1_A_value, b2_A_value, bG2_A_value, bGamma3_A_value, bNabla2_A_value = (
                bias_setA
            )
            b1_B_value, b2_B_value, bG2_B_value, bGamma3_B_value, bNabla2_B_value = (
                bias_setB
            )

            # Unpack stochastic parameters
            cshot, c0_value, c2_value = stoch_set
            nbarmix_value = np.sqrt(nbarA_loc * nbarB_loc)

            # Symbolic variables corresponding to model parameters
            b1_A, b1_B, b2_A, b2_B = sp.symbols("b1_A b1_B b2_A b2_B")
            bG2_A, bG2_B = sp.symbols("bG2_A bG2_B")
            bGamma3_A, bGamma3_B = sp.symbols("bGamma3_A bGamma3_B")
            bNabla2_A, bNabla2_B = sp.symbols("bNabla2_A bNabla2_B")
            c0, c2, h_loc, nbarmix, knormh_loc, knorm_loc = sp.symbols(
                "c0 c2 h_loc nbarmix knormh_loc knorm_loc"
            )
            k_loc = sp.symbols("k_loc")
            pk_mult_symbolic = sp.symbols("pk_mult_symbolic0:15")

            # Map parameters to their symbolic counterparts
            # idx_deriv = [0, 0, b1_A, b2_A, bG2_A, bGamma3_A, bNabla2_A, 1, c0, c2]
            if paramname == "b1_":
                variable = [b1_A, b1_B]
            if paramname == "b2_":
                variable = [b2_A, b2_B]
            if paramname == "bG2_":
                variable = [bG2_A, bG2_B]
            if paramname == "bGamma3_":
                variable = [bGamma3_A, bGamma3_B]
            if paramname == "bNabla2_":
                variable = [bNabla2_A, bNabla2_B]
            if paramname == "c0_":
                variable = [c0, c0]
            if paramname == "c2_":
                variable = [c2, c2]

            derivative_symbolic = self.analytic_deriv[variable[AorB]]

            # Assign fixed values for the substitution
            knormh_value = self.knorm * h
            values = {
                b1_A: b1_A_value,
                b1_B: b1_B_value,
                b2_A: b2_A_value,
                b2_B: b2_B_value,
                bG2_A: bG2_A_value,
                bG2_B: bG2_B_value,
                bGamma3_A: bGamma3_A_value,
                bGamma3_B: bGamma3_B_value,
                bNabla2_A: bNabla2_A_value,
                bNabla2_B: bNabla2_B_value,
                c0: c0_value,
                c2: c2_value,
                h_loc: h,
                nbarmix: nbarmix_value,
                knormh_loc: knormh_value,
                knorm_loc: self.knorm,
            }

            # Compute the evaluated derivative for each wavenumber
            result = []
            for kidx, k_value in enumerate(self.k_loc):
                # Substitute power spectrum terms and wavenumber
                pk_subs = {
                    pk_mult_symbolic[i]: eval_array[i, kidx]
                    for i in range(len(pk_mult_symbolic))
                }
                combined_subs = {**values, **pk_subs, k_loc: k_value}

                # Evaluate the derivative
                evaluated_derivative = derivative_symbolic.subs(combined_subs)
                result.append(evaluated_derivative)

        # RSD part
        else:
            # Unpack bias parameters
            (
                b1_A_value,
                b2_A_value,
                bG2_A_value,
                bGamma3_A_value,
                cct_2_0_A_value,
                cct_2_2_A_value,
                cct_2_4_A_value,
                cct_4_4_A_value,
                cct_4_6_A_value,
            ) = bias_setA
            (
                b1_B_value,
                b2_B_value,
                bG2_B_value,
                bGamma3_B_value,
                cct_2_0_B_value,
                cct_2_2_B_value,
                cct_2_4_B_value,
                cct_4_4_B_value,
                cct_4_6_B_value,
            ) = bias_setB

            # Unpack stochastic parameters
            cshot, cst_0_0_value, cst_0_2_value, cst_2_2_value = stoch_set

            # Precompute common factors
            nbarmix_value = np.sqrt(nbarA_loc * nbarB_loc)

            # Define symbolic variables for bias and stochastic parameters
            (
                b1_A,
                b2_A,
                bG2_A,
                bGamma3_A,
                cct_2_0_A,
                cct_2_2_A,
                cct_2_4_A,
                cct_4_4_A,
                cct_4_6_A,
            ) = sp.symbols(
                "b1_A b2_A bG2_A bGamma3_A cct_2_0_A cct_2_2_A cct_2_4_A cct_4_4_A cct_4_6_A"
            )
            (
                b1_B,
                b2_B,
                bG2_B,
                bGamma3_B,
                cct_2_0_B,
                cct_2_2_B,
                cct_2_4_B,
                cct_4_4_B,
                cct_4_6_B,
            ) = sp.symbols(
                "b1_B b2_B bG2_B bGamma3_B cct_2_0_B cct_2_2_B cct_2_4_B cct_4_4_B cct_4_6_B"
            )
            (
                cshot,
                cst_0_0,
                cst_0_2,
                cst_2_2,
                h_loc,
                nbarmix,
                knormh_loc,
                knorm_loc,
                k_loc,
                fz,
            ) = sp.symbols(
                "cshot cst_0_0 cst_0_2 cst_2_2 h_loc nbarmix knormh_loc knorm_loc k_loc fz"
            )
            pk_mult_symbolic = sp.symbols("pk_mult_symbolic0:40")

            # Map parameters to their symbolic counterparts
            # idx_deriv = [0, 0, b1_A, b2_A, bG2_A, bGamma3_A, bNabla2_A, 1, c0, c2]
            if paramname == "b1_":
                variable = [b1_A, b1_B]
            if paramname == "b2_":
                variable = [b2_A, b2_B]
            if paramname == "bG2_":
                variable = [bG2_A, bG2_B]
            if paramname == "bGamma3_":
                variable = [bGamma3_A, bGamma3_B]
            if paramname == "cct_2_0_":
                variable = [cct_2_0_A, cct_2_0_B]
            if paramname == "cct_2_2_":
                variable = [cct_2_2_A, cct_2_2_B]
            if paramname == "cct_2_4_":
                variable = [cct_2_4_A, cct_2_4_B]
            if paramname == "cct_4_4_":
                variable = [cct_4_4_A, cct_4_4_B]
            if paramname == "cct_4_6_":
                variable = [cct_4_6_A, cct_4_6_B]
            if paramname == "cst_0_0_":
                variable = [cst_0_0, cst_0_0]
            if paramname == "cst_0_2_":
                variable = [cst_0_2, cst_0_2]
            if paramname == "cst_2_2_":
                variable = [cst_2_2, cst_2_2]

            derivative_symbolic_0 = self.analytic_deriv[0][variable[AorB]]
            derivative_symbolic_2 = self.analytic_deriv[1][variable[AorB]]
            derivative_symbolic_4 = self.analytic_deriv[2][variable[AorB]]

            # Assign fixed values for the substitution
            knormh_value = self.knorm * h
            values = {
                b1_A: b1_A_value,
                b1_B: b1_B_value,
                b2_A: b2_A_value,
                b2_B: b2_B_value,
                bG2_A: bG2_A_value,
                bG2_B: bG2_B_value,
                bGamma3_A: bGamma3_A_value,
                bGamma3_B: bGamma3_B_value,
                cct_2_0_A: cct_2_0_A_value,
                cct_2_2_A: cct_2_2_A_value,
                cct_2_4_A: cct_2_4_A_value,
                cct_4_4_A: cct_4_4_A_value,
                cct_4_6_A: cct_4_6_A_value,
                cct_2_0_B: cct_2_0_B_value,
                cct_2_2_B: cct_2_2_B_value,
                cct_2_4_B: cct_2_4_B_value,
                cct_4_4_B: cct_4_4_B_value,
                cct_4_6_B: cct_4_6_B_value,
                cst_0_0: cst_0_0_value,
                cst_0_2: cst_0_2_value,
                cst_2_2: cst_2_2_value,
                h_loc: h,
                fz: self.fz,
                nbarmix: nbarmix_value,
                knormh_loc: knormh_value,
                knorm_loc: self.knorm,
            }

            # Compute the evaluated derivative for each wavenumber
            result_0 = []
            result_2 = []
            result_4 = []
            for kidx, k_value in enumerate(self.k_loc):
                # Substitute power spectrum terms and wavenumber
                pk_subs = {
                    pk_mult_symbolic[i]: eval_array[i, kidx]
                    for i in range(len(pk_mult_symbolic))
                }
                combined_subs = {**values, **pk_subs, k_loc: k_value}

                # Evaluate the derivative
                evaluated_derivative_0 = derivative_symbolic_0.subs(combined_subs)
                evaluated_derivative_2 = derivative_symbolic_2.subs(combined_subs)
                evaluated_derivative_4 = derivative_symbolic_4.subs(combined_subs)
                result_0.append(evaluated_derivative_0)
                result_2.append(evaluated_derivative_2)
                result_4.append(evaluated_derivative_4)
            result = np.concatenate((result_0, result_2, result_4))
        #          print( "result:", result.shape)
        #        print (result_0, result_2, result_4 , result)
        return result
