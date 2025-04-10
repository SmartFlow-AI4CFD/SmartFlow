from cales_post import CaNS, Moser
import numpy as np

folders = [
'eval-0/',
'eval-1/',
'eval-2/',
'eval-3/',
'eval-4/',
]

folders_ref = [
'/scratch/maochao/code/CaLES/run/CHA_RETAU1000_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
'/scratch/maochao/code/CaLES/run/CHA_RETAU2000_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
'/scratch/maochao/code/CaLES/run/CHA_RETAU5200_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
'/scratch/maochao/code/CaLES/run/CHA_RETAU4000_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
'/scratch/maochao/code/CaLES/run/CHA_RETAU10000_H0.1_SMAG_AR1_NX128_NY48_NZ32_MINIMAL/',
]

for i in range(len(folders)):
  les = CaNS(folders[i])
  les.read_stats()
  utau = 2.0*les.retau/les.reb

  ref = CaNS(folders_ref[i])
  ref.read_stats()

  err = np.sqrt(np.sum(les.dzf*(les.u-ref.u)**2)) / np.sum(les.dzf*ref.u)

  with open("compare_params.txt", "a") as f:
      f.write(f"{les.reb:12.0f} {ref.retau:12.6f} {les.retau:12.6f} {(les.retau-ref.retau)/ref.retau:12.6f} {err:12.6f}\n")