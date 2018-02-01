from task0 import Task0
import itertools
'''
author: Kumar Bhargav Srinivasan
'''

if __name__ == "__main__":
    task = Task0()
    dataframe = task.get_dataframe()
    dataframe_no_outcome = dataframe.loc[dataframe['Outcome'] == 'no']
    dataframe_yes_outcome = dataframe.loc[dataframe['Outcome'] == 'yes']
    p_death = len(dataframe_no_outcome)/dataframe.shape[0]
    p_survival = 1 - p_death
    p_class_by_death = (dataframe_no_outcome.groupby("Class")["Outcome"].value_counts()
                       / dataframe_no_outcome.shape[0])

    p_gender_by_death = (dataframe_no_outcome.groupby("Gender")["Outcome"].value_counts()
                           / dataframe_no_outcome.shape[0])

    p_age_by_death = (dataframe_no_outcome.groupby("Age")["Outcome"].value_counts()
                           / dataframe_no_outcome.shape[0])

    p_class_by_survival = (dataframe_yes_outcome.groupby("Class")["Outcome"].value_counts()
                           / dataframe_yes_outcome.shape[0])

    p_gender_by_survival = (dataframe_yes_outcome.groupby("Gender")["Outcome"].value_counts()
                            / dataframe_yes_outcome.shape[0])

    p_age_by_survival = (dataframe_yes_outcome.groupby("Age")["Outcome"].value_counts()
                         / dataframe_yes_outcome.shape[0])

    print(p_class_by_death)
    print(p_age_by_death)
    print(p_gender_by_death)
    print(p_class_by_survival)
    print(p_age_by_survival)
    print(p_gender_by_survival)
    table = list(itertools.product(['1st', '2nd', '3rd', 'crew'], ['adult', 'child'],
                                   ['male', 'female'], repeat=1))
    probability_death_table = {}
    probability_survival_table = {}
    for i in table:
        d_prob = (p_class_by_death[i[0]].values[0] * p_age_by_death[i[1]].values[0] * p_gender_by_death[i[2]].values[0] * p_death)
        s_prob = (p_class_by_survival[i[0]].values[0] * p_age_by_survival[i[1]].values[0] * p_gender_by_survival[i[2]].values[0] * p_survival)
        probability_death_table[i] = d_prob / (d_prob+s_prob)
        probability_survival_table[i] = s_prob / (d_prob+s_prob)
    classification_table = {}
    for i in probability_death_table:
        print(i,probability_death_table[i])
        if probability_death_table[i] > probability_survival_table[i]:
            classification_table[i] = "death"
        else:
            classification_table[i] = "survival"

    print(classification_table)

