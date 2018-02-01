from task0 import Task0
import pandas as pd
import math

'''
author: Kumar Bhargav Srinivasan
'''

if __name__ == "__main__":
    task = Task0()
    df = task.get_dataframe()
    total_probabilty = pd.crosstab(index=df['Class'],
                            columns=[df['Gender'], df['Age']])
    Outcome_no_df = df.loc[df['Outcome'] == 'no']
    probability_table = pd.crosstab(index=Outcome_no_df['Class'],
                           columns=[Outcome_no_df['Gender'], Outcome_no_df['Age']])
    print(probability_table)
    print(total_probabilty)
    print(probability_table /  total_probabilty)
    classification_table = probability_table /  total_probabilty
    print("Gender|    female |     male     |")
    print("----------------------------------")
    print("Age| adult| child |adult | child |")
    print("----------------------------------")
    passanger_class = classification_table.index.values
    row = 0
    for i in classification_table.values:
        print(passanger_class[row],end=' |  ')
        for val in i:
            if math.isnan(val):
                print("undefined |",end=' ' )
                continue
            if val >=0.5:
                print("Death  |",end=' ')
            else:
                print("Survival   |",end=' ')
        row += 1
        print()
