import pandas as pd
import numpy as np

def survived_within_class():
    df = pd.read_csv("titanic_train.csv")
    # Get survivor data
    survivors = df[df["Survived"]==1]
    # Group survivor data per class
    grouped_survivors = survivors.groupby(["Pclass"])
    # Count number of survivors per class
    survivors_per_class = grouped_survivors.count()["Survived"]
    print(survivors_per_class)
    # Count number of people who didn't survive per class
    non_survivors_per_class = df[df["Survived"]==0].groupby("Pclass").count()["Survived"]
    # Count number of people per class
    total_per_class = df.groupby(["Pclass"])["Survived"].count()
    # Compute ratios of people who survived
    classes_num = len(survivors_per_class.index)
    def ratio_func(survivors_num, count):
        try:
            return survivors_num / count
        except ZeroDivisionError:
            return -1
    ratio_func_vector = np.vectorize(ratio_func)
    survivor_ratios = ratio_func_vector(survivors_per_class, total_per_class)
    non_survivor_ratios = ratio_func_vector(non_survivors_per_class, total_per_class)
    return np.flip(np.around(survivor_ratios, 3)), np.flip(np.around(non_survivor_ratios, 3))
                        
print(survived_within_class())