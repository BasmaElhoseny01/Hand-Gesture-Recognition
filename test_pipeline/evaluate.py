from utils import *

with open('results.txt') as f:
  result = [ int(i) for i in f ]

with open('expected.txt') as f:
  expected = [ int(i) for i in f ]

accuracy = performance_analysis(result,expected)
print(accuracy)