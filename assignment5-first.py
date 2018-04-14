import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def generate_weights(num_samples):
    samples = {}
    for i in range(num_samples):
        intelligence = np.random.normal(100, 15)
        university = np.random.rand() < 1 / (1 + np.exp(-(intelligence - 100) / 5))
        major = np.random.rand() < 1 / (1 + np.exp(-(intelligence - 110) / 5))
        salary = np.random.gamma(0.1 * intelligence + major + 3 * university, 5)
        pdf = stats.gamma.pdf(salary, 5)
        #pdf is used as weight in posterior for specific salary
        samples[(intelligence, major, university, salary)] = pdf
    return samples


def get_posterior_for(major, university, salary, samples):
    numerator = 0
    denominator = 0
    for sample in samples.keys():
        if round(sample[3]) == salary:
            denominator += samples[sample]
        if sample[1] == major and sample[2] == university and round(sample[3]) == salary:
            numerator += samples[sample]
    return numerator / denominator


def likelihood_weighting(count_samples=10000):
    MAJOR = {1: 'comp', 0: 'bus'}
    UNIVERSITY = {1: 'ucolo', 0: 'metro'}
    samples = generate_weights(count_samples)
    salary = 120
    print("P(major = {},university={} |salary : 120) = {}"
          .format(MAJOR[0],UNIVERSITY[0],get_posterior_for(major=0, university=0, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 120) = {}"
          .format(MAJOR[0], UNIVERSITY[1],get_posterior_for(major=0, university=1, salary=salary,samples=samples)))
    print("P(major = {},university={} |salary : 120) = {}"
          .format(MAJOR[1], UNIVERSITY[0],
                  get_posterior_for(major=1, university=0, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 120) = {}"
          .format(MAJOR[1], UNIVERSITY[1],
                  get_posterior_for(major=1, university=1, salary=salary, samples=samples)))

    prob1 = [get_posterior_for(major=0, university=0, salary=salary, samples=samples),
            get_posterior_for(major=0, university=1, salary=salary, samples=samples),
            get_posterior_for(major=1, university=0, salary=salary, samples=samples),
            get_posterior_for(major=1, university=1, salary=salary, samples=samples)]
    salary = 60

    print("P(major = {},university={} |salary : 60) = {}"
          .format(MAJOR[0], UNIVERSITY[0], get_posterior_for(major=0, university=0, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 60) = {}"
          .format(MAJOR[0], UNIVERSITY[1], get_posterior_for(major=0, university=1, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 60) = {}"
          .format(MAJOR[1], UNIVERSITY[0],
                  get_posterior_for(major=1, university=0, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 60) = {}"
          .format(MAJOR[1], UNIVERSITY[1],
                  get_posterior_for(major=1, university=1, salary=salary, samples=samples)))

    prob2 = [get_posterior_for(major=0, university=0, salary=salary, samples=samples),
            get_posterior_for(major=0, university=1, salary=salary, samples=samples),
            get_posterior_for(major=1, university=0, salary=salary, samples=samples),
            get_posterior_for(major=1, university=1, salary=salary, samples=samples)]

    salary = 20
    print("P(major = {},university={} |salary : 20) = {}"
          .format(MAJOR[0], UNIVERSITY[0], get_posterior_for(major=0, university=0, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 20) = {}"
          .format(MAJOR[0], UNIVERSITY[1], get_posterior_for(major=0, university=1, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 20) = {}"
          .format(MAJOR[1], UNIVERSITY[0],
                  get_posterior_for(major=1, university=0, salary=salary, samples=samples)))
    print("P(major = {},university={} |salary : 20) = {}"
          .format(MAJOR[1], UNIVERSITY[1],
                  get_posterior_for(major=1, university=1, salary=salary, samples=samples)))

    prob3 = [get_posterior_for(major=0, university=0, salary=salary, samples=samples),
            get_posterior_for(major=0, university=1, salary=salary, samples=samples),
            get_posterior_for(major=1, university=0, salary=salary, samples=samples),
            get_posterior_for(major=1, university=1, salary=salary, samples=samples)]
    plt.plot(prob1, 'r--', prob2 , 'b--', prob3, 'g--')
    plt.show()

if __name__ == '__main__':
    likelihood_weighting(100000)