import numpy as np

data = np.loadtxt("train-0/forcing.out")
print(data.shape)


selected = [0, 20, 40, 60, 80, 100, 114]

times = data[:, 0]
groups = []
current_group = [data[0]]
for i in range(1, len(data)):
    if times[i] < times[i - 1]:
        groups.append(np.array(current_group))
        current_group = []
    current_group.append(data[i])
groups.append(np.array(current_group))

print(len(groups))

import matplotlib.pyplot as plt

for idx in selected:
    plt.plot(groups[idx][:, 0], groups[idx][:, 1], label=f'CFD {idx}')
    mean_value = np.mean(groups[idx][:, 1])
    print(f"Mean for group {idx}: {mean_value}")

plt.xlabel('Column 0')
plt.ylabel('Column 1')
plt.legend()
# plt.show()
plt.savefig('split_forcing_0.png')
plt.close()


data = np.loadtxt("train-1/forcing.out")
print(data.shape)

times = data[:, 0]
groups = []
current_group = [data[0]]
for i in range(1, len(data)):
    if times[i] < times[i - 1]:
        groups.append(np.array(current_group))
        current_group = []
    current_group.append(data[i])
groups.append(np.array(current_group))

print(len(groups))

import matplotlib.pyplot as plt

for idx in selected:
    plt.plot(groups[idx][:, 0], groups[idx][:, 1], label=f'CFD {idx}')
    mean_value = np.mean(groups[idx][:, 1])
    print(f"Mean for group {idx}: {mean_value}")

plt.xlabel('Column 0')
plt.ylabel('Column 1')
plt.legend()
# plt.show()
plt.savefig('split_forcing_1.png')
plt.close()


data = np.loadtxt("train-2/forcing.out")
print(data.shape)

times = data[:, 0]
groups = []
current_group = [data[0]]
for i in range(1, len(data)):
    if times[i] < times[i - 1]:
        groups.append(np.array(current_group))
        current_group = []
    current_group.append(data[i])
groups.append(np.array(current_group))

print(len(groups))

import matplotlib.pyplot as plt

for idx in selected:
    plt.plot(groups[idx][:, 0], groups[idx][:, 1], label=f'CFD {idx}')
    mean_value = np.mean(groups[idx][:, 1])
    print(f"Mean for group {idx}: {mean_value}")

plt.xlabel('Column 0')
plt.ylabel('Column 1')
plt.legend()
# plt.show()
plt.savefig('split_forcing_2.png')
plt.close()