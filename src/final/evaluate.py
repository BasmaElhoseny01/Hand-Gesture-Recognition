import sys
import os
# Step(0) Import the required utilities
current_dir=os.path.abspath('')
print("Current Directory: ",current_dir)
src_dir=os.path.join(current_dir,'../')
sys.path.insert(1, src_dir) # back to the src directory

from modules.performance import performance_analysis

def main(argv):
    preprocessing_option = argv[0]
    feature_extractor_option = argv[1]
    model_option = argv[2]

    lines = ['preprocessing_option: ' + str(preprocessing_option)+"\n",
             'feature_extractor_option: '+str(feature_extractor_option)+"\n",
             'model_option: '+str(model_option)+"\n"
             ]
    
    with open('./expected.txt') as f:
      expected = [int(i) for i in f]

    with open('./results.txt') as f:
        result = [int(i) for i in f]

        accuracy = performance_analysis(result, expected)
        print("Accuracy: ", accuracy)
        lines.append("Accuracy: "+str(accuracy)+"\n")

    
    with open('./accuracy.txt', 'w') as fp:
      fp.writelines(lines)

if __name__ == "__main__":
    main(sys.argv[1:])
