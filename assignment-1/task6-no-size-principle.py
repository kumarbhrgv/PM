from task1 import PriorProb
import numpy as np
import math
from matplotlib import pyplot as plt

"""
Given one observation, X = {(1.5, 0.5)},
compute the posterior P(H|X) with 𝜎 = 12.
You will get one probability for each possible hypothesis.
Display your result either as a bar graph or a list of probabilities.
Use Tenenbaum's Size Principle as the likelihood function.
"""

class LikeLihood(object):
    def __init__(self,observation,sigma1,sigma2):
        self.observation = observation
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def makeLikelihood(self,l1,l2,s1,s2):
        count = 0
        for ob in self.observation:
            x = ob[0]
            y = ob[1]
            if l1 <= x <= s1 and l2<= y <= s2:
                count +=1
        if len(self.observation) == count:
            return 1
        else:
            return 0

class Posterior(object):
    def __init__(self,obs,sigma1,sigma2):
        self.observation = obs
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def generatePosterior(self,l1,l2,s1,s2):
        likelihood = LikeLihood(self.observation, self.sigma1, self.sigma2)
        probability_likelihood = likelihood.makeLikelihood(l1,l2,s1,s2)
        prob = PriorProb(l1, l2, s1, s2, self.sigma1, self.sigma2)
        probability_prior = prob.makePrior()
        return probability_prior*probability_likelihood


class PosteriorDistribution(object):
    def __init__(self,h,sigma1,sigma2,obs):
        self.h= h
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.observation = obs
        self.posterior_probability = {}

    def generatePosterior(self):
        posterior = Posterior(self.observation,self.sigma1,self.sigma2)
        for i in range(1, self.h + 1):
            self.posterior_probability[i] = posterior.generatePosterior(-i,-i,i,i)
        return self.posterior_probability

    def normalize_posterior(self):
        return { k: v/sum(self.posterior_probability.values()) for k,v  in self.posterior_probability.items()}

    def plotDistribution(self,prior_dist):
        label = prior_dist.keys()
        index = np.arange(len(label))
        plt.bar(index, prior_dist.values())
        plt.xlabel('hypothesis - i', fontsize=5)
        plt.ylabel('prior probability', fontsize=5)
        plt.xticks(index, label, fontsize=5, rotation=30)
        title = 'Normalized Posterior distribution without size principle for sigma1 = '+str(self.sigma1)+" and sigma2 = "+str(self.sigma2)
        plt.title(title)
        plt.show()


if __name__ == '__main__':
    obs = [(1.5,0.5)]
    h = 10
    sigma1 = 20
    sigma2 = sigma1
    posteriorDistribution = PosteriorDistribution(h,sigma1,sigma2,obs)
    probs = posteriorDistribution.generatePosterior()
    normalized_probs = posteriorDistribution.normalize_posterior()
    posteriorDistribution.plotDistribution(normalized_probs)
    print(sum(normalized_probs.values()))