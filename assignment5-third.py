import numpy as np
import matplotlib.pyplot as plt

"""
Defined probability for F
"""
def probability(sample):
    if sample < 0 or sample > 1:
        return 0
    else:
        return sample**3

"""
Defined probability for G|F
"""
def proposal(probability, sample):
    if sample < 0 or sample > 1:
        return 0
    else:
        return 1 - abs(sample - probability)

"""
Metropolis Hastings
"""
def MH(num_samples):
    accepted_f = {}
    accepted_g = {}
    mean = 0.1
    dev = 0.3
    f_init, g_init = 0.5, 0.5
    accepted_count = 0
    for i in range(0, num_samples):
        f = np.random.normal(mean, dev)
        g = np.random.normal(mean, dev)

        f = f + f_init
        g = g + g_init

        p_1 = probability(f)
        q_1 = proposal(f, g)
        p_2 = probability(f_init)
        q_2 = proposal(f_init, g_init)

        u = np.random.rand()
        if p_2 > 0 and q_2 > 0:
            acceptance = min(1, (p_1*q_1)/(p_2*q_2))
        else:
            acceptance = 1

        if acceptance > u:
            f_init = f
            g_init = g
            accepted_f[accepted_count] = f
            accepted_g[accepted_count] = g
            accepted_count += 1
    return accepted_f, accepted_g


if __name__ == "__main__":
    num = 1000000
    accepted_1, accepted_2 = MH(num)
    #Plot the samples
    fig = plt.figure()
    plt.xlabel('G values')
    plt.ylabel('F values')
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.hist2d(list(accepted_2.values()), list(accepted_1.values()), bins=20,cmap='jet')
    plt.show()
    # Display the traversal
    plt.subplot(1, 2, 1)
    plt.plot(list(accepted_1.keys())[0:1000], list(accepted_1.values())[0:1000])
    plt.title("walk vs iteration, x")
    plt.subplot(1, 2, 2)
    plt.plot(list(accepted_2.keys())[0:1000], list(accepted_2.values())[0:1000],"r")
    plt.title("walk vs iteration, y")
    plt.show()
    #Part b
    print(np.cov(list(accepted_1.values()), list(accepted_2.values())) + np.mean(list(accepted_1.values()))*np.mean(list(accepted_2.values())))