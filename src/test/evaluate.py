import sys
import os
# Step(0) Import the required utilities
current_dir=os.path.abspath('')
print("Current Directory: ",current_dir)
src_dir=os.path.join(current_dir,'../')
sys.path.insert(1, src_dir) # back to the src directory

from modules.performance import performance_analysis

def main(argv):


    
    with open('./expected.txt') as f:
      expected = [int(i) for i in f]

    with open('./results.txt') as f:
        result = [int(i) for i in f]

        accuracy = performance_analysis(result, expected)
        print("Accuracy: ", accuracy)
        # lines.append("Accuracy: "+str(accuracy)+"\n")

    
    # with open('./accuracy.txt', 'w') as fp:
    #   fp.writelines(lines)

if __name__ == "__main__":
    main(sys.argv[1:])
