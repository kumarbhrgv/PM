from task2 import PosteriorDistribution
import numpy as np
from matplotlib import pyplot as plt
"""
Class to generate Generalization probability for given observation over hypothesis space
"""


class Generalization(object):
    
    def __init__(self, h, sigma1, sigma2, obs):
        self.h = h
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.observation = obs
        self.general_probability = np.zeros((2*self.h+1, 2*self.h+1))
        self.posterior = PosteriorDistribution(h,sigma1,sigma2,obs)
        self.posterior.generatePosterior()
        self.posterior_probability = self.posterior.normalize_posterior()
        self.posterior.plotDistribution(self.posterior_probability)


    def random_point_prediction(self,y):
        y_prediction = {}
        for h_val in range(1,self.h+1):
            if h_val > 0:
                if -h_val <= y[0] <= h_val:
                    if -h_val <= y[-1] <= h_val:
                        y_prediction[h_val] = self.posterior_probability[h_val]
                    else:
                        y_prediction[h_val] = 0
                else:
                    y_prediction[h_val] = 0

        return y_prediction

    def generate_Generalization_prediction(self):
        for row in range(-self.h,self.h+1):
            for col in range(-self.h,self.h+1):
                prediction = self.random_point_prediction((row, col))
                probability = sum(prediction.values())
                self.general_probability[row + 10][col + 10] = probability

        return self.general_probability

    def plot_Generalization(self):
        x = np.arange(-self.h, self.h+1)
        y = np.arange(-self.h, self.h + 1)
        X, Y = np.meshgrid(x, y)
        title = "plot contour with Generalization probability for observation = " + str(self.observation)
        plt.title(title)
        k = plt.contourf(X, Y, self.general_probability,cmap=plt.cm.gray)
        plt.colorbar(k)
        plt.show()


if __name__ == '__main__':
    h = 10
    sigma1 = 10
    sigma2 = 10
    obs = [(1.5,0.5)]
    general = Generalization(h,sigma1,sigma2,obs)
    general.generate_Generalization_prediction()
    general.plot_Generalization()
    
    obs2 = [(4.5, 2.5)]
    general = Generalization(h, sigma1, sigma2, obs2)
    general.generate_Generalization_prediction()
    general.plot_Generalization()

    h = 10
    sigma1 = 30
    sigma2 = 30
    obs3 = [(2.2, -.2)]
    general = Generalization(h, sigma1, sigma2, obs3)
    general.generate_Generalization_prediction()
    general.plot_Generalization()

    obs4 = [(2.2, -.2), (.5, .5)]
    general = Generalization(h, sigma1, sigma2, obs4)
    general.generate_Generalization_prediction()
    general.plot_Generalization()

    obs5 = [(2.2, -.2), (.5, .5), (1.5, 1)]
    general = Generalization(h, sigma1, sigma2, obs5)
    general.generate_Generalization_prediction()
    general.plot_Generalization()

