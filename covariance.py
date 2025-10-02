import numpy as np
from sympy.physics.wigner import wigner_3j

def get_covariance(model, FLAG_NOCROSS_SPEC = False, RSD = False, multipoles = [True, True, True]):
  print ("Getting Covariance \n")
  if (RSD == False): return get_covariance_real(model, FLAG_NOCROSS_SPEC = FLAG_NOCROSS_SPEC)
  else: return get_covariance_rsd(model, FLAG_NOCROSS_SPEC = FLAG_NOCROSS_SPEC, multipoles = multipoles)


def get_covariance_real(model, FLAG_NOCROSS_SPEC = False): ### no-cross not really used here ### I think it it...

  lenk = model.len_k

  # Initialize covariance matrix 
  if (FLAG_NOCROSS_SPEC == 0):
    covmt = np.zeros((model.ntracerssum * lenk, model.ntracerssum * lenk))
  else:
    covmt = np.zeros((model.ntracers * lenk, model.ntracers * lenk))
  print ("Cov shape: ", covmt.shape)

  for idx_cov1 in range(model.ntracers):
      for idx_cov2 in range(idx_cov1, model.ntracers):
          if (FLAG_NOCROSS_SPEC == 1 and  idx_cov2 != idx_cov1): continue ### just skip those if no cross

          for idx_cov3 in range(model.ntracers):
              for idx_cov4 in range(idx_cov3, model.ntracers):
                  if (FLAG_NOCROSS_SPEC == 1 and  idx_cov4 != idx_cov3): continue

                  if (FLAG_NOCROSS_SPEC == 0):
                      idx_ab = model.ntracers_matrix[idx_cov1][idx_cov2]
                      idx_cd = model.ntracers_matrix[idx_cov3][idx_cov4]
                  else:
                      idx_ab = idx_cov1
                      idx_cd = idx_cov3

                  idx_ac = model.ntracers_matrix[idx_cov1][idx_cov3]
                  idx_ad = model.ntracers_matrix[idx_cov1][idx_cov4]
                  idx_bc = model.ntracers_matrix[idx_cov2][idx_cov3]
                  idx_bd = model.ntracers_matrix[idx_cov2][idx_cov4]

                  PAC_real = model.Pdata[idx_ac*lenk:(idx_ac+1)*lenk]
                  PAD_real = model.Pdata[idx_ad*lenk:(idx_ad+1)*lenk]
                  PBC_real = model.Pdata[idx_bc*lenk:(idx_bc+1)*lenk]
                  PBD_real = model.Pdata[idx_bd*lenk:(idx_bd+1)*lenk]

                  for idx_k in range(lenk):
                      Mk = model.Nk_loc[idx_k]
                      if PAC_real[idx_k] < 0 or PBD_real[idx_k] < 0 or PAD_real[idx_k] < 0 or PBC_real[idx_k] < 0:
                          print("Negative Ps")
                          exit(1)
                      CABCD = (PAC_real[idx_k] * PBD_real[idx_k] + PAD_real[idx_k] * PBC_real[idx_k]) / Mk
                      covmt[idx_ab*lenk + idx_k, idx_cd*lenk + idx_k] = CABCD
  return covmt

def cov_rsd_value(ell, ell_prime, Nk, PAC, PBD, PAD, PBC):
#def cov_rsd_value(ell, ell_prime, k, Deltak, Vsurvey, PAC, PBD, PAD, PBC):
    summing = 0
    ell_array = [0,2,4]
    for index1, ell1 in enumerate(ell_array):
        for index2, ell2 in enumerate(ell_array):
            #for index3, ell3 in enumerate(ell_array):
            for index3, ell3 in enumerate([0,2,4,6,8]): ### Important: here it is fixed by triangle ineq. in Wigner!
                summing += (2.*ell3+1.)*wigner_3j(ell1, ell2, ell3, 0, 0, 0)**2.*wigner_3j(ell, ell_prime, ell3, 0, 0, 0)**2.*(PAC[index1]*PBD[index2] + (-1)**(ell_prime+ell2)*PAD[index1]*PBC[index2])
                #summing += (-1.)**ell2*(2.*ell3+1.)*wigner_3j(ell1, ell2, ell3, 0, 0, 0)**2.*wigner_3j(ell, ell_prime, ell3, 0, 0, 0)**2.*(PAC[index1]*PBD[index2] + (-1)**(ell_prime)*PAD[index1]*PBC[index2])
    #return summing*(2.*ell+1.)*(2.*ell_prime+1.)/(2.*Vsurvey*k*k*Deltak)
    #Nk = Vsurvey*k*k*Deltak/(2.*np.pi**2.)
    return summing*(2.*ell+1.)*(2.*ell_prime+1.)/Nk


def get_covariance_rsd(model, FLAG_NOCROSS_SPEC = False, multipoles = [True, True, True]): ## todo nocross
  ell_array = np.array([0,2,4])[multipoles]
  lenk = model.len_k
  lenell = len(ell_array)

  if (FLAG_NOCROSS_SPEC == False):
    lencov = lenell*model.ntracerssum ### (multipoles) x (spectra)
  else:
    lencov = lenell*model.ntracers ### (multipoles) x (spectra)

  covmt = np.zeros((lencov*lenk, lencov*lenk))
  print ("Cov shape: ", covmt.shape)

  for Aidx in range(model.ntracers):
    for Bidx in range(Aidx,model.ntracers):
      if (FLAG_NOCROSS_SPEC == 1 and  Aidx != Bidx): continue ### just skip those if no cross

      for Cidx in range(model.ntracers):
        for Didx in range(Cidx,model.ntracers):
          if (FLAG_NOCROSS_SPEC == 1 and  Cidx != Didx): continue ### just skip those if no cross

          print (Aidx, Bidx, Cidx, Didx)
          if (FLAG_NOCROSS_SPEC == 0):
            ABidx = model.ntracers_matrix[Aidx][Bidx]
            CDidx = model.ntracers_matrix[Cidx][Didx]
          else:
            ABidx = Aidx
            CDidx = Cidx

          ACidx = model.ntracers_matrix[Aidx][Cidx]
          BDidx = model.ntracers_matrix[Bidx][Didx]
          ADidx = model.ntracers_matrix[Aidx][Didx]
          BCidx = model.ntracers_matrix[Bidx][Cidx]

          if (multipoles[0]): 
            PAC_0 = model.Pdata[ACidx*lenk*lenell+0*lenk:(ACidx)*lenk*lenell+0*lenk+lenk]
            PBD_0 = model.Pdata[BDidx*lenk*lenell+0*lenk:(BDidx)*lenk*lenell+0*lenk+lenk]
            PAD_0 = model.Pdata[ADidx*lenk*lenell+0*lenk:(ADidx)*lenk*lenell+0*lenk+lenk]
            PBC_0 = model.Pdata[BCidx*lenk*lenell+0*lenk:(BCidx)*lenk*lenell+0*lenk+lenk]
#            print ("Cov mono:", PAC_0, PBD_0, PAD_0, PBC_0)
          else: 
            PAC_0 = [0] * lenk
            PBD_0 = [0] * lenk
            PAD_0 = [0] * lenk
            PBC_0 = [0] * lenk

          if (multipoles[1]): 
            temp = np.sum(multipoles[0]) ### n spec before
            PAC_2 = model.Pdata[ACidx*lenk*lenell+temp*lenk:(ACidx)*lenk*lenell+temp*lenk+lenk]
            PBD_2 = model.Pdata[BDidx*lenk*lenell+temp*lenk:(BDidx)*lenk*lenell+temp*lenk+lenk]
            PAD_2 = model.Pdata[ADidx*lenk*lenell+temp*lenk:(ADidx)*lenk*lenell+temp*lenk+lenk]
            PBC_2 = model.Pdata[BCidx*lenk*lenell+temp*lenk:(BCidx)*lenk*lenell+temp*lenk+lenk]
          else: 
            PAC_2 = [0] * lenk
            PBD_2 = [0] * lenk
            PAD_2 = [0] * lenk
            PBC_2 = [0] * lenk

          if (multipoles[2]): 
            temp = np.sum(multipoles[0:2]) ### n spec before
            PAC_4 = model.Pdata[ACidx*lenk*lenell+temp*lenk:(ACidx)*lenk*lenell+temp*lenk+lenk]
            PBD_4 = model.Pdata[BDidx*lenk*lenell+temp*lenk:(BDidx)*lenk*lenell+temp*lenk+lenk]
            PAD_4 = model.Pdata[ADidx*lenk*lenell+temp*lenk:(ADidx)*lenk*lenell+temp*lenk+lenk]
            PBC_4 = model.Pdata[BCidx*lenk*lenell+temp*lenk:(BCidx)*lenk*lenell+temp*lenk+lenk]
          else: 
            PAC_4 = [0] * lenk
            PBD_4 = [0] * lenk
            PAD_4 = [0] * lenk
            PBC_4 = [0] * lenk

          for index1, ell1 in enumerate(ell_array):
            for index2, ell2 in enumerate(ell_array):
              for idx in range(lenk):
                PAC = [PAC_0[idx], PAC_2[idx], PAC_4[idx]]
                PBD = [PBD_0[idx], PBD_2[idx], PBD_4[idx]]
                PAD = [PAD_0[idx], PAD_2[idx], PAD_4[idx]]
                PBC = [PBC_0[idx], PBC_2[idx], PBC_4[idx]]
                k   = model.k_loc[idx]

                #if (idx == lenk-1):  Deltak = 0.00462541
                if (idx == lenk-1):  Deltak = model.k_loc[idx] - model.k_loc[idx-1] ### fixing it for last index...
                else:  Deltak = model.k_loc[idx+1] - model.k_loc[idx]

                CABCD = cov_rsd_value(ell1, ell2, model.Nk_loc[idx], PAC, PBD, PAD, PBC)
                #CABCD = cov_rsd_value(ell1, ell2, k, Deltak, model.Vbox, PAC, PBD, PAD, PBC)
                covidx1 = ABidx*len(ell_array)*lenk + index1*lenk + idx
                covidx2 = CDidx*len(ell_array)*lenk + index2*lenk + idx
                covmt[covidx1][covidx2] = CABCD


#  for i in range (lencov*lenk):
#    for j in range (lencov*lenk):
#      print ("Cov: ", i,j, covmt[i][j])
#  exit(1)

  return covmt

