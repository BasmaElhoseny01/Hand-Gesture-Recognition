from utils import *
from modules.performance import performance_analysis

with open('../results/svm_results.txt') as f:
    result = [int(i) for i in f]

with open('../results/expected.txt') as f:
    expected = [int(i) for i in f]

accuracy_svm = performance_analysis(result, expected)
print("SVM Accuracy: ", accuracy_svm)


with open('../results/rf_results.txt') as f:
    result = [int(i) for i in f]
accuracy_rf = performance_analysis(result, expected)
print("RF Accuracy: ", accuracy_rf)

# Step(6) Write the Accuracy into txt file
f = open('../results/accuracy.txt', 'w')
f.writelines("SVM: "+str(accuracy_svm)+"\nRF: "+str(accuracy_rf))
