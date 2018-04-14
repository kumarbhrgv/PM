import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

"""
basic Gibbs sampler with conditionals
"""
def gibbs_sampler(x_init, means, std_devs, num_samples):
    x1_samples = []
    x2_samples = []
    for i in range(0, num_samples):
        mean_x1_x2 = means[0] + (std_devs[0][-1] / std_devs[-1][-1]) * (x_init - means[-1])
        std_dev_x1_x2 = std_devs[0][0] - (std_devs[0][-1] / std_devs[-1][-1]) * std_devs[-1][0]
        x1 = np.random.normal(mean_x1_x2, std_dev_x1_x2)
        mean_x2_x1 = means[-1] + (std_devs[-1][0] / std_devs[0][0]) * (x1 - means[0])
        std_dev_x2_x1 = std_devs[-1][-1] - (std_devs[-1][0] / std_devs[0][0]) * std_devs[0][-1]
        x2 =  np.random.normal(mean_x2_x1, std_dev_x2_x1)
        x1_samples.append(x1)
        x2_samples.append(x2)
        if i % 5000 == 0:
            pass
            #plot_samples(x1_samples,x2_samples)
    return x1_samples, x2_samples


def plot_samples(dist_x,dist_y):
    ex = np.linspace(-6, 6, 100)
    ex2 = np.linspace(-10, 10, 100)
    plt.subplot(1, 2, 1)
    plt.hist(dist_x, 50, normed=True)
    title = "number of samples = "+ str(len(dist_x))
    plt.plot(ex, norm.pdf(ex, 1, 1), 'r-', label=title)
    plt.title(title)
    plt.subplot(1, 2, 2)

    plt.hist(dist_y, 50, normed=True)
    plt.plot(ex2, norm.pdf(ex2, 0, 3), 'r-', label=title)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    samples = 100000
    x_in = np.random.rand()%2
    mean = [1, 0]
    std_dev = np.array([[1, -0.5], [-0.5, 3]])
    dist_x, dist_y = gibbs_sampler(x_in, mean, std_dev, samples)
    plot_samples(dist_x,dist_y)
    """
    x_in = 0
    dist_x, dist_y = gibbs_sampler(x_in, mean, std_dev, samples)
    plot_samples(dist_x,dist_y)
    """
    grid = (sns.jointplot(np.array(dist_x), np.array(dist_y),kind='kde',color='k'))
    grid.savefig("second.png")
