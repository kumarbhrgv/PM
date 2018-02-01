import pandas as pd

'''
author: Kumar Bhargav Srinivasan
'''
class Task0:
    def __init__(self):
        self.dataframe = pd.read_csv("titanic.txt",delimiter=" ")
        self.dataframe.columns = ['Class', 'Age', 'Gender', 'Outcome']
        self.passenger_class = {"1st" : [0,0] , "2nd":[0,1] ,"3rd": [1,0], "crew": [1,1]}
        self.table = {}

    def get_dataframe(self):
        return self.dataframe

    def joint_probability(self):
        for line in self.dataframe.values:
            temp = self.get_class(line[0])
            instance = (temp[0],temp[1],line[1],line[2],line[3])
            if instance in self.table:
                self.table[instance] +=1
            else:
                self.table[instance] =1
        length = sum(self.table.values())
        for i in self.table:
            self.table[i] = self.table[i] / length
        return self.table

    def print_joint_probability(self):
        print()
        for instance in self.table:
            print(instance,self.table[instance])

    def get_class(self,p_class):
        return self.passenger_class[p_class]

    def generate_stats(self):
        print("Stats of passanger class in titanic")
        print(self.dataframe['Class'].value_counts())
        print("Stats of passanger's Age in titanic")
        print(self.dataframe['Age'].value_counts())
        print("Stats of passanger Gender in titanic")
        print(self.dataframe['Gender'].value_counts())
        print("Stats of passanger survival in titanic")
        print(self.dataframe['Outcome'].value_counts())


if __name__ == "__main__":
    current_task = Task0()
    current_task.generate_stats()
    current_task.joint_probability()
    print("############ Joint Probability ############")
    current_task.print_joint_probability()
