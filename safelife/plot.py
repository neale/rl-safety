import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread

sns.set(style="darkgrid")

a = []
with open ('rand_reward_0.txt', 'r') as f:
    for line in f.readlines():
        a.append(float(line))
b= []
with open ('rand_reward_1.txt', 'r') as f:
    for line in f.readlines():
        b.append(float(line))


fig, ax = plt.subplots(3, 2)
plt.xlabel('Epidoes')
plt.ylabel('Episodic Reward')

mean_a = np.ones((len(a))) * np.array(a).mean()
mean_b = np.ones((len(b))) * np.array(b).mean()

sns.lineplot(x=list(range(len(a))), y=a, ax=ax[0, 0], label='reward')
sns.lineplot(x=list(range(len(a))), y=mean_a, linestyle='--', ax=ax[0, 0], label='mean')
sns.lineplot(x=list(range(len(b))), y=b, ax=ax[0, 1], label='reward')
sns.lineplot(x=list(range(len(b))), y=mean_b, linestyle='--', ax=ax[0, 1], label='mean')
plt.legend(loc='best', fancybox=True)
ax[0, 0].set_title('Trial 1')
ax[0, 1].set_title('Trial 2')


ax[1, 0].imshow(imread('z_dim.png'))
ax[1, 0].set_xlabel('Rand Reward (-1, 1)')

ax[1, 1].imshow(imread('rand2.png'))
ax[1, 1].set_xlabel('Rand Reward (-1, 1)')

ax[2, 0].imshow(imread('vae_samples.png.png'))
ax[2, 0].set_xlabel('70x70 VAE samples')

ax[2, 1].imshow(imread('generated2.png'))
ax[2, 1].set_xlabel('70x70 VAE samples')

plt.show()
