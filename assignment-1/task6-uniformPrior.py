import math
from matplotlib import pyplot as plt
import numpy as np

"""
Make a bar graph of the prior distribution, P(H), for 𝜎1 = 𝜎2 = 6.  
Make a graph of the prior distribution for 𝜎1 = 𝜎2 = 12.
"""


class PriorProb(object):
    def __init__(self,l1,l2,s1,s2,sigma1,sigma2):
        self.l1 = l1
        self.l2 = l2
        self.s1 = s1
        self.s2 = s2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def makePrior(self):
        return 1/(self.s1*self.s2)


class PriorDistribution(object):
    def __init__(self,h,sigma1,sigma2):
        self.h = h
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.prior_distribution = {}

    def makePriorDistribution(self):
        for i in range(1,self.h+1):
            probability = PriorProb(-i,-i,i,i,self.sigma1,self.sigma2)
            self.prior_distribution[i] = probability.makePrior()
        return self.prior_distribution

    def plotPriorDistribution(self,prior_dist):
        label = prior_dist.keys()
        index = np.arange(len(label))
        plt.bar(index, prior_dist.values())
        plt.xlabel('hypothesis - i', fontsize=5)
        plt.ylabel('prior probability', fontsize=5)
        plt.xticks(index, label, fontsize=5, rotation=30)
        title = 'Uninformative Prior distribution with sigma1 = '+str(self.sigma1)+" and sigma2 = "+str(self.sigma2)
        plt.title(title)
        plt.show()

    def normalizePrior(self):
        self.normalized_values = { k: v/sum(self.prior_distribution.values()) for k,v  in self.prior_distribution.items()}
        return self.normalized_values


if __name__ == "__main__":

    priordistribution = PriorDistribution(10, 6, 6)
    prior_dist = priordistribution.makePriorDistribution()
    priordistribution.plotPriorDistribution(prior_dist)
    normalized_prior_dist = priordistribution.normalizePrior()
    priordistribution.plotPriorDistribution(normalized_prior_dist)
    print(normalized_prior_dist,sum(normalized_prior_dist.values()))

