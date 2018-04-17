import matplotlib.pyplot as plt
import numpy as np

class Task1:
    def __init__(self, alpha):
        self.alpha = alpha

    def prob_new_table(self, n):
        if n == 0:
            return 1
        else:
            return self.alpha / (n - 1 + self.alpha)

    def plot(self,n):
        y_axis = np.zeros(n)
        for i in range(n):
            prob = self.prob_new_table(i + 1)
            y_axis[i] = prob
        x_axis = np.arange(n)
        x_axis += 1
        plt.plot(x_axis, y_axis)
        plt.show()


class Task2:
    def __init__(self,alpha,n):
        self.alpha = alpha
        self.n = n
        self.colors = ['green', 'red', 'blue', 'orange', 'pink', 'brown', 'yellow', 'cyan', 'black', 'white', 'pink']

    def prob_occupied_table(self,nth_cus,number_people_table):
        if not number_people_table:
            return [0]
        Z = nth_cus - 1 + self.alpha
        prob = np.array(number_people_table)
        prob = prob / Z
        return prob.tolist()

    def generate_samples(self):
        task1 = Task1(self.alpha)
        theta = []
        number_people_table = []
        current_table = 0
        samples = []
        for nth in range(self.n):
            probability_new = task1.prob_new_table(nth+1)
            probability_occupied = self.prob_occupied_table(nth+1,number_people_table)
            table_p = probability_occupied + [probability_new]
            if nth == 0:
                table_p = [probability_new]
            temp = np.random.multinomial(1, table_p,1)
            chosen_val = np.argmax(temp[0])
            if chosen_val == current_table:
                current_table += 1
                number_people_table.append(1)
                x_pos = np.random.rand()
                y_pos = np.random.rand()
                theta.append((x_pos, y_pos))
            else:
                number_people_table[chosen_val] += 1
            x, y = np.random.multivariate_normal([theta[chosen_val][0], theta[chosen_val][1]], [[0.01, 0], [0, 0.01]])
            samples.append([x, y, chosen_val])
        return samples

    def plot_scatter(self,samples):
        color_list = samples[:, 2].tolist()
        print("number of clusters : ", len(np.unique(color_list)))
        color_val_list = [self.colors[int(i)] for i in color_list]
        plt.scatter(samples[:, 0], samples[:, 1], edgecolors=color_val_list, s=80, facecolors='none')
        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":

    # Task 1
    currentTask = Task1(0.5)
    currentTask.plot(500)

    # Task 2
    task2 = Task2(0.5,500)
    generated_samples = task2.generate_samples()
    task2.plot_scatter(np.array(generated_samples))

    task2_1000 = Task2(0.5, 1000)
    generated_samples_1000 = task2_1000.generate_samples()
    task2_1000.plot_scatter(np.array(generated_samples_1000))