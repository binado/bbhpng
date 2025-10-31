from pprint import pprint

import numpy as np

import covariance
import fiducial_cosmo
import modellib
from config import Config

# def fisher(ntracers, eps=0.04, noise_amplitude=0.0, kmin_value = 0.001, kmax_value = 0.15, V1440_factor = 1., nbar_value = 4e-4, b1set = [], b2set = [], bG2set = [], bGamma3set = [], bNabla2set = [], cct_2_0_set = [], cct_2_2_set = [], cct_2_4_set = [], cct_4_4_set = [], cct_4_6_set = [], omega_cdm_fid = OMEGA_CDM_FID, h_fid = H_FID, As_fid = AS_FID, includeAs = False, nonlin = True, nocross_stoch = False, RSD = False, multipoles = [True, True, True]):


def fisher(
    model,
    analytic_bias=True,
    nocross_spec=False,
    RSD=False,
    multipoles=[True, True, True],
    assymetric_kmax=[],
):
    """
    Perform Fisher matrix analysis for a single-tracer model.

    Args:
        model (Model): The single-tracer model instance.

    Returns:
        numpy.ndarray: The inverted Fisher matrix.
    """

    #    model = modellib.set_model(ntracers, nbar_value = nbar_value, RSD = RSD, V1440_factor = V1440_factor, includeAs = includeAs)

    len_param = model.len_param
    len_k = model.len_k
    print("len k: ", len_k)
    print("len param: ", len_param)

    if assymetric_kmax != [] and np.any(np.array(assymetric_kmax) > model.kmax):
        print("increase kmax!!!", model.kmax, assymetric_kmax)
        exit(1)

    print("Calculating derivatives ")
    # Calculate parameter derivatives with respect to model predictions
    derivatives = model.get_derivatives(
        analytic_bias=analytic_bias, FLAG_NOCROSS_SPEC=nocross_spec
    )

    #  for i in range(len(derivatives)):
    #    param = model.active_parameters[i]
    #    if(param.get('name') == 'b2_'):
    #      print ("Entered!", i)
    #      derivatives[i] = derivatives[i]/2.

    # derivatives_cross = model.get_derivatives(analytic_bias = analytic_bias, FLAG_NOCROSS_SPEC = False)
    # print (np.array(derivatives).shape, np.array(derivatives_cross).shape)
    # exit(1)
    #  for i in range(len (derivatives)):
    #    print ('deriv ', i, derivatives[i])

    # Compute covariance matrix and its inverse
    print("Calculating covariance ")
    cov = covariance.get_covariance(
        model, FLAG_NOCROSS_SPEC=nocross_spec, RSD=RSD, multipoles=multipoles
    )
    print("deriv, cov", np.array(derivatives).shape, np.array(cov).shape)
    print("Inverting covariance ")
    covm1 = np.linalg.inv(cov)
    try:
        np.linalg.cholesky(cov)
        print(" Covariance is positive definite")
    except np.linalg.LinAlgError:
        print(" Covariance is not positive definite!")
        return False

    # Initialize the Fisher matrix
    print("Calculating Fisher ")
    Fmatrix = np.zeros((len_param, len_param))

    # Compute Fisher matrix (had to go back here to include multipoles)
    idxmax = model.ntracers
    if RSD:
        nmultipoles = sum(multipoles)
    else:
        nmultipoles = 1
    # exit(1)
    for i in range(len_param):
        print("param idx ", i)
        for j in range(len_param):
            # for j in range(i, len_param):
            for idx_Pspec1 in range(idxmax):
                # for idx_Pspec2 in range(idxmax):
                for idx_Pspec2 in range(idx_Pspec1, idxmax):
                    if idx_Pspec1 != idx_Pspec2 and nocross_spec:
                        continue
                    for idx_Pspec3 in range(idxmax):
                        for idx_Pspec4 in range(idx_Pspec3, idxmax):
                            if idx_Pspec3 != idx_Pspec4 and nocross_spec:
                                continue

                            if nocross_spec:
                                idx_ab = model.ntracers_matrix[idx_Pspec1][idx_Pspec2]
                                idx_cd = model.ntracers_matrix[idx_Pspec3][idx_Pspec4]
                            else:
                                idx_ab = idx_Pspec1
                                idx_cd = idx_Pspec3

                            for ell1 in range(nmultipoles):
                                for ell2 in range(nmultipoles):
                                    for idxk in range(len_k):
                                        if assymetric_kmax != []:
                                            if (
                                                model.k_loc[idxk]
                                                > assymetric_kmax[idx_ab]
                                            ):
                                                #                                            print ("Skipping: ", model.k_loc[idxk], assymetric_kmax[idx_ab], idx_ab)
                                                #            exit(1)
                                                continue
                                            if (
                                                model.k_loc[idxk]
                                                > assymetric_kmax[idx_cd]
                                            ):
                                                #                                           print ("Skipping: ", model.k_loc[idxk], assymetric_kmax[idx_cd], idx_cd)
                                                #           exit(1)
                                                continue
                                        idx1 = (
                                            (idx_ab) * len_k * nmultipoles
                                            + ell1 * len_k
                                            + idxk
                                        )
                                        idx2 = (
                                            (idx_cd) * len_k * nmultipoles
                                            + ell2 * len_k
                                            + idxk
                                        )
                                        # print (idx1, idx2)
                                        Fmatrix[i, j] += (
                                            derivatives[i][idx1]
                                            * derivatives[j][idx2]
                                            * covm1[idx1, idx2]
                                        )
                                        if derivatives[i][idx1] != derivatives[i][idx1]:
                                            masking = model.Pdata > 0
                                            print(
                                                "Weird: ",
                                                i,
                                                derivatives[i][masking],
                                                derivatives[i][~masking],
                                            )

    #                                          print ("Weird", i, idx1,  derivatives[i], np.array(derivatives).shape, model.Pdata[len_k:len_k*2][idxk])
    # print ("Weird", i, idx1,  derivatives[i], np.array(derivatives).shape, idx_ab, idx_cd, idxk, model.Pdata[len_k:len_k*2], model.Pdata[len_k:len_k*2][idxk], len_k)

    # for i in range(len_param):
    #   for j in range(len_param):
    #     print ("Fisher :", i, j,  Fmatrix[i, j])
    #  exit(1)

    #    # Compute Fisher matrix
    #    if (nocross_spec): idxmax = model.ntracers
    #    else:              idxmax = model.ntracerssum
    #    for i in range(len_param):
    #        for j in range(len_param):
    #            for idx_Pspec1 in range(idxmax):
    #                for idx_Pspec2 in range(idxmax):
    #                    for idxk in range(len_k):
    #                        print (np.array(derivatives).shape, idx_Pspec1*len_k+idxk, idx_Pspec2*len_k+idxk)
    #                        Fmatrix[i, j] += (derivatives[i][idx_Pspec1*len_k+idxk] *
    #                                            derivatives[j][idx_Pspec2*len_k+idxk] *
    #                                            covm1[idx_Pspec1*len_k+idxk, idx_Pspec2*len_k+idxk])

    # Add prior information to Fisher matrix
    sigma2prior_array = model.active_priors**2
    priors = np.diag(1.0 / sigma2prior_array[sigma2prior_array != 0])
    Fmatrix += priors

    # Invert the Fisher matrix and display eigenvalues
    print("Inverting Fisher ")
    #    for i in range(len_param):
    #      for j in range(len_param):
    #        print(i,j,Fmatrix[i][j])
    try:
        np.linalg.cholesky(Fmatrix)
        print(" Fisher is positive definite")
    except np.linalg.LinAlgError:
        return False
    Fm1 = np.linalg.inv(Fmatrix)
    print("Eigenvalues (params):", np.linalg.eigvals(Fmatrix))

    # Display parameter errors
    for idxtemp in range(len(Fm1)):
        paramname = model.active_parameters[idxtemp].get("name")
        print("Errors ", paramname, np.sqrt(Fm1[idxtemp][idxtemp]))

    return Fm1


### the analysis
if __name__ == "__main__":
    config = Config.from_cli()

    config_args = {
        "ntracers": config["tracers.ntracers"],
        "eps": config["tracers.eps"],
        "noise_amplitude": config["tracers.noise_amplitude"],
        "kmin_value": config["tracers.kmin_value"],
        "kmax_value": config["tracers.kmax_value"],
        "V1440_factor": config["tracers.V1440_factor"],
        "nbar_value": config["tracers.nbar_value"],
        "b1set": config["bias.b1set"],
        "b2set": config["bias.b2set"],
        "bG2set": config["bias.bG2set"],
        "bGamma3set": config["bias.bGamma3set"],
        "bNabla2set": config["bias.bNabla2set"],
        "cct_2_0_set": config["bias.cct_2_0_set"],
        "cct_2_2_set": config["bias.cct_2_2_set"],
        "cct_2_4_set": config["bias.cct_2_4_set"],
        "cct_4_4_set": config["bias.cct_4_4_set"],
        "cct_4_6_set": config["bias.cct_4_6_set"],
        "use_universality_relation": config["bias.use_universality_relation"],
        "universality_relation_p": config["bias.universality_relation_p"],
        "omega_cdm_fid": config["cosmology.omega_cdm_fid"],
        "h_fid": config["cosmology.h_fid"],
        "As_fid": config["cosmology.As_fid"],
        "f_NL_fid": config["cosmology.f_NL_fid"],
        "includeAs": config["cosmology.includeAs"],
        "nonlin": config["power_spectrum.nonlin"],
        "nocross_stoch": config["power_spectrum.nocross_stoch"],
        "RSD": config["power_spectrum.RSD"],
        "multipoles": config["power_spectrum.multipoles"],
        "active_bias": config["power_spectrum.active_bias"],
        "active_stoch": config["power_spectrum.active_stoch"],
        "open_precomputed_files": config["caching.open_precomputed_files"],
        "save_precomputed_files": config["caching.save_precomputed_files"],
        "redshift": config["power_spectrum.redshift"],
    }

    default_args = {
        "omega_cdm_fid": fiducial_cosmo.OMEGA_CDM_FID,
        "h_fid": fiducial_cosmo.H_FID,
        "As_fid": fiducial_cosmo.AS_FID,
        "f_NL_fid": fiducial_cosmo.F_NL_VALUE,
    }

    merged_args = {**default_args, **config_args}
    if config["verbose"]:
        pprint(merged_args)
    model = modellib.set_model(**merged_args)
    fisher(
        model,
        analytic_bias=True,
        RSD=config["power_spectrum.RSD"],
        multipoles=config["power_spectrum.multipoles"],
    )
