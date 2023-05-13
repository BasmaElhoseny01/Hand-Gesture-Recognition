from utils import *
from modules.performance import performance_analysis


def main(argv):
    preprocessing_option = argv[0]
    feature_extractor_option = argv[1]
    model_option = argv[2]

    lines = ['preprocessing_option: ' + str(preprocessing_option)+"\n",
             'feature_extractor_option: '+str(feature_extractor_option)+"\n",
             'model_option: '+str(model_option)+"\n"
             ]
    
    with open('../results/expected.txt') as f:
      expected = [int(i) for i in f]

    if (model_option == "both" or model_option == "rf"):
        with open('../results/rf_results.txt') as f:
          result = [int(i) for i in f]

        accuracy = performance_analysis(result, expected)
        print("RF Accuracy: ", accuracy)
        lines.append("RF: "+str(accuracy)+"\n")

    if(model_option=="both" or model_option=="svm"):
        with open('../results/svm_results.txt') as f:
            result = [int(i) for i in f]

        accuracy = performance_analysis(result, expected)
        print("SVM Accuracy: ", accuracy)
        lines.append("SVM: "+str(accuracy)+"\n")

    if(model_option!="both" and model_option!="rf" and model_option!="svm"):
        raise TypeError("Wrong Model Option")
    
    with open('../results/accuracy.txt', 'w') as fp:
      fp.writelines(lines)

if __name__ == "__main__":
    main(sys.argv[1:])
