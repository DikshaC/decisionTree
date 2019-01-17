This project implements the decision tree algorithm for the datasets where each dataset is divided into three sets: the training set, the validation set and the test set. Data sets are in CSV format. The first line in the file gives the attribute names. Each line after that is a training (or test) example that contains a list of attribute values separated by a comma. The last attribute is the class-variable, assuming that all attributes take values from the domain [0,1].

The decision tree is made by using two heuristics for selecting the next attribute:
1) Information Gain
2) Variance Impurity

After that post-pruning method is applied to it with integer values of L and K.
The output tree is printed as follows:
  wesley = 0 :
  | honor = 0 :
  | | barclay = 0 : 1
  | | barclay = 1 : 0
  | honor = 1 :
  | | tea = 0 : 0
  | | tea = 1 : 1
  wesley = 1 : 0
  
  The accuarcies are also reported for the above.
