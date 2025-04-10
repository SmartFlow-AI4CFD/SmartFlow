from cales_post import CaNS, Moser
import numpy as np
import matplotlib.pyplot as plt

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
  utau_ref = 2.0*ref.retau/ref.reb

  err = np.sqrt(np.sum(les.dzf*(les.u-ref.u)**2)) / np.sum(les.dzf*ref.u)

  plt.semilogx(ref.zc, ref.u, label='Reference')
  plt.semilogx(les.zc, les.u, label='LES')
  plt.legend()
  plt.savefig(f'compare_{i}.png')
  plt.close()


  plt.semilogx(ref.zc*ref.retau, ref.u/utau_ref, label='Reference')
  plt.semilogx(les.zc*les.retau, les.u/utau, label='LES')
  plt.legend()
  plt.grid(True, which='both', linestyle='--', linewidth=0.5)
  plt.savefig(f'compare_{i}_plus.png')
  plt.close()