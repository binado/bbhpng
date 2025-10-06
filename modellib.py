import modelclass
from fiducial_cosmo import *


def import_and_overwrite_globals(filepath):
    """
    Imports a Python file and overwrites all existing global variables with the ones from the fi     le.

    Parameters:
    filepath (str): Path to the Python file to be imported.
    """
    modelclass.import_and_overwrite_globals(filepath)
    #    print ("Modellib:", CCT_2_2_VALUE)
    global_vars = globals()

    # Read the file contents
    with open(filepath, "r") as f:
        file_contents = f.read()

    # Execute the file contents in the global scope
    exec(file_contents, global_vars)


#    print ("Modellib:", CCT_2_2_VALUE)


# Define stochastic parameters for auto and cross correlations
def create_bias_params(
    traceridx,
    b1_value,
    b2_value,
    bG2_value,
    bGamma3_value,
    b_ct_values,
    activelist=[True, True, True, True, True, True, True, True, True],
    RSD=False,
):
    """Helper to create stochastic parameter dictionaries."""

    if not RSD:
        return {
            "b1": {
                "name": "b1_",
                "value": b1_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[0],
                "type": "bias",
                "traceridx": traceridx,
            },
            "b2": {
                "name": "b2_",
                "value": b2_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[1],
                "type": "bias",
                "traceridx": traceridx,
            },
            "bG2": {
                "name": "bG2_",
                "value": bG2_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[2],
                "type": "bias",
                "traceridx": traceridx,
            },
            "bGamma3": {
                "name": "bGamma3_",
                "value": bGamma3_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[3],
                "type": "bias",
                "traceridx": traceridx,
            },
            "bNabla2": {
                "name": "bNabla2_",
                "value": b_ct_values[0],
                "priorsigma": FIXED_PRIOR,
                "active": activelist[4],
                "type": "bias",
                "traceridx": traceridx,
            },
        }

    else:
        return {
            "b1": {
                "name": "b1_",
                "value": b1_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[0],
                "type": "bias",
                "traceridx": traceridx,
            },
            "b2": {
                "name": "b2_",
                "value": b2_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[1],
                "type": "bias",
                "traceridx": traceridx,
            },
            "bG2": {
                "name": "bG2_",
                "value": bG2_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[2],
                "type": "bias",
                "traceridx": traceridx,
            },
            "bGamma3": {
                "name": "bGamma3_",
                "value": bGamma3_value,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[3],
                "type": "bias",
                "traceridx": traceridx,
            },
            "cct_2_0": {
                "name": "cct_2_0_",
                "value": b_ct_values[0],
                "priorsigma": FIXED_PRIOR,
                "active": activelist[4],
                "type": "bias",
                "traceridx": traceridx,
            },
            "cct_2_2": {
                "name": "cct_2_2_",
                "value": b_ct_values[1],
                "priorsigma": FIXED_PRIOR,
                "active": activelist[5],
                "type": "bias",
                "traceridx": traceridx,
            },
            "cct_2_4": {
                "name": "cct_2_4_",
                "value": b_ct_values[2],
                "priorsigma": FIXED_PRIOR,
                "active": activelist[6],
                "type": "bias",
                "traceridx": traceridx,
            },
            "cct_4_4": {
                "name": "cct_4_4_",
                "value": b_ct_values[3],
                "priorsigma": FIXED_PRIOR,
                "active": activelist[7],
                "type": "bias",
                "traceridx": traceridx,
            },
            "cct_4_6": {
                "name": "cct_4_6_",
                "value": b_ct_values[4],
                "priorsigma": FIXED_PRIOR,
                "active": activelist[8],
                "type": "bias",
                "traceridx": traceridx,
            },
        }


# Define stochastic parameters for auto and cross correlations
def create_stochastic_params(
    cshot, traceridx, activelist=[True, True, True], RSD=False
):
    """Helper to create stochastic parameter dictionaries."""
    if not RSD:
        return {
            "cshot": {
                "name": "cshot_",
                "value": cshot,
                "priorsigma": FIXED_PRIOR,
                "active": False,
                "type": "stoch",
                "traceridx": traceridx,
            },
            "c0": {
                "name": "c0_",
                "value": 0.0,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[0],
                "type": "stoch",
                "traceridx": traceridx,
            },
            "c2": {
                "name": "c2_",
                "value": 0.0,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[1],
                "type": "stoch",
                "traceridx": traceridx,
            },
        }
    else:
        return {
            "cshot": {
                "name": "cshot_",
                "value": cshot,
                "priorsigma": FIXED_PRIOR,
                "active": False,
                "type": "stoch",
                "traceridx": traceridx,
            },
            "cst_0_0": {
                "name": "cst_0_0_",
                "value": 0.0,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[0],
                "type": "stoch",
                "traceridx": traceridx,
            },
            "cst_0_2": {
                "name": "cst_0_2_",
                "value": 0.0,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[1],
                "type": "stoch",
                "traceridx": traceridx,
            },
            "cst_2_2": {
                "name": "cst_2_2_",
                "value": 0.0,
                "priorsigma": FIXED_PRIOR,
                "active": activelist[2],
                "type": "stoch",
                "traceridx": traceridx,
            },
        }


def set_model(
    ntracers,
    eps=0.04,
    noise_amplitude=0.0,
    kmin_value=0.001,
    kmax_value=0.15,
    V1440_factor=1.0,
    nbar_value=4e-4,
    b1set=[],
    b2set=[],
    bG2set=[],
    bGamma3set=[],
    bNabla2set=[],
    cct_2_0_set=[],
    cct_2_2_set=[],
    cct_2_4_set=[],
    cct_4_4_set=[],
    cct_4_6_set=[],
    omega_cdm_fid=OMEGA_CDM_FID,
    h_fid=H_FID,
    As_fid=AS_FID,
    f_NL_fid=F_NL_VALUE,
    includeAs=False,
    nonlin=True,
    nocross_stoch=False,
    RSD=False,
    multipoles=[True, True, True],
    active_bias=[True, True, True, True, True, True, True, True, True],
    active_stoch=[True, True, True],
    open_precomputed_files=True,
    save_precomputed_files=False,
    redshift=0,
):
    """
    Create a tracer (ST or MT) cosmological model.

    Args:
        eps (float): Scaling parameter for the model.
        noise_amplitude (float): Amplitude of additional noise.

    Returns:
        Model: An instance of the Model class configured for MT analysis.
    """
    # Define cosmological parameters
    cosmo = {
        "omegacdm": {
            "name": "omegacdm_",
            "value": omega_cdm_fid,
            "priorsigma": FIXED_PRIOR,
            "active": True,
            "type": "cosmo",
            "traceridx": [-1],
        },
        "h": {
            "name": "h_",
            "value": h_fid,
            "priorsigma": FIXED_PRIOR,
            "active": True,
            "type": "cosmo",
            "traceridx": [-1],
        },
    }
    if includeAs:
        cosmo["As"] = {
            "name": "As_",
            "value": As_fid,
            "priorsigma": FIXED_PRIOR,
            "active": True,
            "type": "cosmo",
            "traceridx": [-1],
        }

    else:
        cosmo["As"] = {
            "name": "As_",
            "value": As_fid,
            "priorsigma": FIXED_PRIOR,
            "active": False,
            "type": "cosmo",
            "traceridx": [-1],
        }

    cosmo["f_NL"] = {
        "name": "f_NL_",
        "value": f_NL_fid,
        "priorsigma": FIXED_PRIOR,
        "active": True,
        "type": "cosmo",
        "traceridx": [-1],
    }

    # active lists for theory
    # active_bias  = [True, True, True, True, True, True, True, True, True]
    # active_stoch = [True, True, True]
    p1loop = 1.0
    if not nonlin:
        active_bias = [True, False, False, False, False, False, False, False, False]
        active_stoch = [True, False, False]
        p1loop = 0.0

    # Define bias parameters for two tracers (A and B)
    bias_set = []
    for i in range(ntracers):
        b1_value, b2_value, bG2_value, bGamma3_value, bNabla2_value = (
            B1_VALUE,
            B2_VALUE,
            BG2_VALUE,
            BGAMMA3_VALUE,
            BNABLA2_VALUE,
        )
        cct_2_0_value, cct_2_2_value, cct_2_4_value, cct_4_4_value, cct_4_6_value = (
            CCT_2_0_VALUE,
            CCT_2_2_VALUE,
            CCT_2_4_VALUE,
            CCT_4_4_VALUE,
            CCT_4_6_VALUE,
        )
        b1_value, b2_value, bG2_value, bGamma3_value, bNabla2_value = (
            B1_VALUE,
            B2_VALUE,
            BG2_VALUE,
            BGAMMA3_VALUE,
            BNABLA2_VALUE,
        )
        if b1set != []:
            b1_value += b1set[i]
        if b2set != []:
            b2_value += b2set[i]
        if bG2set != []:
            bG2_value += bG2set[i]
        if bGamma3set != []:
            bGamma3_value += bGamma3set[i]
        if bNabla2set != []:
            bNabla2_value += bNabla2set[i]
        if cct_2_0_set != []:
            cct_2_0_value += cct_2_0_set[i]
        if cct_2_2_set != []:
            cct_2_2_value += cct_2_2_set[i]
        if cct_2_4_set != []:
            cct_2_4_value += cct_2_4_set[i]
        if cct_4_4_set != []:
            cct_4_4_value += cct_4_4_set[i]
        if cct_4_6_set != []:
            cct_4_6_value += cct_4_6_set[i]

        if not RSD:
            bias_ct_values = [bNabla2_value]
        else:
            bias_ct_values = [
                cct_2_0_value,
                cct_2_2_value,
                cct_2_4_value,
                cct_4_4_value,
                cct_4_6_value,
            ]

        print(b1_value, b2_value, bG2_value, bGamma3_value, bias_ct_values)
        bias_set.append(
            create_bias_params(
                [i],
                b1_value,
                b2_value,
                bG2_value,
                bGamma3_value,
                bias_ct_values,
                activelist=active_bias,
                RSD=RSD,
            )
        )
        # print ("BIAS SET: " , bias_set)

    stoch_set = []
    for i in range(ntracers):
        for j in range(i, ntracers):
            if not nocross_stoch:
                active_stoch_temp = active_stoch
            elif nocross_stoch and i != j:
                active_stoch_temp = [False, False, False]
            else:
                active_stoch_temp = active_stoch

            cshot = 1.0 if i == j else 0.0
            stoch_set.append(
                create_stochastic_params(
                    cshot, [i, j], activelist=active_stoch_temp, RSD=RSD
                )
            )

    # Define other model parameters
    kmin = kmin_value
    kmax = kmax_value
    Vbox = 1440**3 * V1440_factor  # Volume of the simulation box
    z = redshift  # Redshift

    if isinstance(nbar_value, list):
        if len(nbar_value) != ntracers:
            print("Wrong shot noise provided")
            exit(1)

        # Tracer shot noise levels
        tracers_shot_noise = nbar_value  # Noise for tracers
    else:
        nbar_total = nbar_value  # Total shot

        # Tracer shot noise levels
        tracers_shot_noise = []  # Noise for tracers
        for i in range(ntracers):
            tracers_shot_noise.append(nbar_total / ntracers)

    # Instantiate and return the Model
    return modelclass.Model(
        cosmo,
        bias_set,
        stoch_set,
        tracers_shot_noise,
        kmin,
        kmax,
        Vbox,
        z,
        eps=eps,
        noise_amplitude=noise_amplitude,
        p1loop=p1loop,
        RSD=RSD,
        multipoles=multipoles,
        open_precomputed_files=open_precomputed_files,
        save_precomputed_files=save_precomputed_files,
    )
