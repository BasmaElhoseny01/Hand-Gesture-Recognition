from utils import *
def performance_analysis(result,expected):
    # Accuracy is the percentage of data that are correctly classified, which ranges from 0 to 1
    accuracy=accuracy_score(result, expected)
    
    # Precision is your go-to evaluation metric when dealing with imbalanced data.

    # Recall
    return accuracy