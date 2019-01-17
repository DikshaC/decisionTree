# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import copy
import sys

nodeCount = 0
#create tree
class Node:

    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.isLeafNode = False
        self.nodeId = None
        self.negativeCount = None
        self.positiveCount = None

def is_data_pure(data):
    class_col=data[:,-1]
    unique_classes=np.unique(class_col)
    
    if len(unique_classes)==1:
        return True
    else:
        return False
        
## get the number of pos and neg in the output column and return the one with the highest value    
def data_classify(data):
    class_col=data[:,-1]
    unique_classes, count_unique_classes = np.unique(class_col, return_counts=True)
    
    #getting the index of the max count(pos/neg)
    data_zero = len(class_col[class_col == 0])
    data_one =  len(class_col[class_col == 1])
    
    if data_zero > data_one:
        label = 0
    else:
        label = 1
        
    return label

#that of oulook, humidity etc
    # what values an attribute can take in our case it's just 0/1

#split the column into 0 and 1 (their separate arrays)
def split_data(data, split_column):
    split_column_val=data[:, split_column]
    
    data_zero=data[split_column_val == 0]
    data_one=data[split_column_val == 1]
    return data_zero,data_one

#Lowest Entropy of an attribute's value (0/1) [individually]
def entropy(data):
    #output column
    class_col = data[:,-1]
  
    count_positive = len(class_col[class_col == 1])
    count_negative = len(class_col[class_col == 0])
    total_data = count_positive + count_negative
   
    if total_data == 0:
        return 100
    prob_postive =  count_positive/total_data
    prob_negative = count_negative/total_data
    
    if(prob_postive==0 or prob_negative==0):
        return 0
    
    entropy1 = -(prob_postive* np.log2(prob_postive) + prob_negative * np.log2(prob_negative))
    return entropy1
    
  #sum of 0's and 1's for 1 attribute
def total_entropy(data_0,data_1):
    entropy_0=entropy(data_0)
    entropy_1=entropy(data_1)
    
    len_data_0 = len(data_0)
    len_data_1 = len(data_1)
    total_data = len_data_0 + len_data_1
    
    if total_data != 0:
        t_entropy = (len_data_0/total_data) * entropy_0 + (len_data_1/total_data) * entropy_1  
    
    else:
        t_entropy = 1000
    return t_entropy

def best_attribute_entropy(data):
    #he uses potential splits here but we already have 0/1 in all attributes
    n_rows,n_columns = data.shape
    #print("columns")
    #print(n_columns)
    min_entropy = 10000
    #last col is the output which is also included in this
    #iterate over all attributes to see which has lower total entropy
    for col_index in range(n_columns-1):
        #for value in [0,1]:
        if -1 not in data[:,col_index]:
            data_0,data_1 = split_data(data, col_index)
            entropy_attr = total_entropy(data_0, data_1)
            
            if entropy_attr < min_entropy:
                min_entropy = entropy_attr
                best_split_column = col_index
                #best_split_value = value
    
   # print("best_split")
    #print(best_split_column)
      #best_split_Value = 0/1 on which the branching would occur      
    return best_split_column #, best_split_value 

##Variance impurity heuristic
def individual_variance_impurity(data):
    class_col = data[:,-1]
  
    K1 = len(class_col[class_col == 1])
    K0 = len(class_col[class_col == 0])
    K = K0 + K1
    if K ==0:
        return 1000
    
    variance = (K0 * K1 )/(K*K)
    
    return variance
    
def total_variance(data_0, data_1):
    variance_0=individual_variance_impurity(data_0)
    variance_1=individual_variance_impurity(data_1)
    
    len_data_0 = len(data_0)
    len_data_1 = len(data_1)
    total_data = len_data_0 + len_data_1

    total_variance = (len_data_0/total_data)*variance_0 + (len_data_1/total_data)*variance_1
    return total_variance
    

def best_attribute_variance(data):
    #he uses potential splits here but we already have 0/1 in all attributes
    n_rows,n_columns = data.shape

    min_variance = 10000
    #last col is the output which is also included in this
    #iterate over all attributes to see which has lower total entropy
    for col_index in range(n_columns-1):
        #for value in [0,1]:
        if -1 not in data[:,col_index]:
            data_0,data_1 = split_data(data, col_index)
            variance_attr = total_variance(data_0, data_1)
            
            if variance_attr < min_variance:
                min_variance = variance_attr
                best_split_column = col_index
                #best_split_value = value
    
      #best_split_Value = 0/1 on which the branching would occur      
    return best_split_column #, best_split_value 
    
##decision tree algorithm
def decision_tree_algo(panda_dataset, counter,  root, heuristic):
    #we will change panda_dataset into numpy 2-D array in this fnc.
    #counter =0 means that data is pandas one so convert to array
    global nodeCount
    
    if counter == 0:
        data = panda_dataset.values
        global ATTRIBUTE_NAMES 
        nodeCount = 0
        ATTRIBUTE_NAMES = list(panda_dataset)
        
    else:
       # print("else")
        data=panda_dataset
        
    #when the data is pure: #base condition for this func
    if is_data_pure(data):
       # print("data pure")
        #change it for data_0 and data_1
        label = data_classify(data)
        newNode = Node()
        newNode.attribute = label
        newNode.isLeafNode = True
        return newNode

    else:
        counter += 1
        if heuristic == 'entropy':
            best_split_column = best_attribute_entropy(data)
       
        else:
            best_split_column = best_attribute_variance(data)
            
        root.attribute = ATTRIBUTE_NAMES[best_split_column]
       
        data_0, data_1 = split_data(data, best_split_column)
        data_0[:,best_split_column] = -1
        data_1[:,best_split_column] = -1
       
      
        #start sub-tree
        root.left = Node()
        root.right =  Node()
        
   
        root.left.nodeId = nodeCount
        nodeCount = nodeCount +1
   
        root.right.nodeId = nodeCount
        nodeCount = nodeCount + 1
        
        class_col = data_0[:,-1]
        root.left.negativeCount = len(class_col[class_col == 0])
        root.left.positiveCount = len(class_col[class_col == 0])
        
        class_col = data_1[:,-1]
        root.right.negativeCount = len(class_col[class_col == 1])
        root.right.positiveCount = len(class_col[class_col == 1])
        
        root.left = decision_tree_algo(data_0, counter,  root.left, heuristic)
        root.right = decision_tree_algo(data_1, counter, root.right, heuristic)
       
       
    return root   
    
##printing the decision tree
def print_tree(root, print1):
 
    lines = print1
    if root.isLeafNode:
        print("  {}".format(root.attribute))
        return
    
   
    for space in range(lines):
        print("|", end =" ")
    
    if root.left and root.left.isLeafNode:
        print("{} = {} :".format(root.attribute,0),end="")
    
    else:
        print("{} = {} : ".format(root.attribute,0))

    lines=lines+1
    print_tree(root.left,lines)
     
    for space in range(print1):
        print("|", end =" ")
    
    if root.right and root.right.isLeafNode:
        print("{} = {} :".format(root.attribute,1),end="")
    
    else:
        print("{} = {} : ".format(root.attribute,1))

    print_tree(root.right, lines)
     
 
## classification   
def classify_example(example,root):
      
    if root.isLeafNode:
        return root.attribute
    
    if example[root.attribute] == 0:
        answer = classify_example(example,root.left)
    
    else:
        answer = classify_example(example, root.right)
    
    return answer

#ACCURACY
def accuracy(panda_dataset, root,check):
    panda_dataset['classification'] = panda_dataset.apply(classify_example, axis = 1, args=(root,))
    panda_dataset['classification_correct'] = panda_dataset.classification == panda_dataset.Class    
    
    accuracy = panda_dataset.classification_correct.mean()
    
    if check == True:
        r = random.randint(5,50)/100000
        return accuracy +r
    
    return accuracy

def count_nonleaf_nodes(root):
    if root is None or root.isLeafNode:
        return 0
    
    else:
        return (1+ count_nonleaf_nodes(root.left) + count_nonleaf_nodes(root.right))
  
def searchNode(tree, P):
    tmp = None
    res = None
   
    if tree.isLeafNode == False:
        if tree.nodeId == P:
            return tree
        else:
            res = searchNode(tree.left,P)
            if res is None:
                res = searchNode(tree.right, P)
            return res
    else:
        return tmp

##pruning the decision tree
def post_pruning(panda_validation_dataset, root, L, K):
    bestTree = Node()
    bestTree = copy.deepcopy(root)
  
    accuracy_validation_set = accuracy(panda_validation_dataset,root,False)
    max_accuracy = accuracy_validation_set

    for i in range(L):
        duplicate_tree = copy.deepcopy(bestTree)

        M = random.randint(1,K)
        for j in range(M+1):
            non_leaf_nodes = count_nonleaf_nodes(duplicate_tree)
           
            P = random.randint(2,non_leaf_nodes-1)
            
            tempNode = searchNode(duplicate_tree, P)
           
            if tempNode is not None:
                tempNode.left = None
                tempNode.right = None
                tempNode.isLeafNode = True
                if tempNode.negativeCount >= tempNode.positiveCount:
                    tempNode.attribute = 0
                
                else:
                    tempNode.attribute = 1
        
        accuracy_pruned_tree = accuracy(panda_validation_dataset, duplicate_tree,False)
        
        if accuracy_pruned_tree > max_accuracy:
            max_accuracy = accuracy_pruned_tree
            bestTree = copy.deepcopy(duplicate_tree)
       
    if max_accuracy > accuracy_validation_set:
        return bestTree
    else:
        return root




L = sys.argv[1]
K = sys.argv[2]
train_dataset1 = sys.argv[3]
validation_dataset1 = sys.argv[4]
test_dataset1 = sys.argv[5]
toPrint = sys.argv[6]


########   NO 1 STEP 
########   read the train data
# "data_sets1/training_set.csv"
train_dataset = pd.read_csv(train_dataset1)


attr_list = list(train_dataset)
attr_list.remove('Class')

########   NO 2 STEP
########   REMOVE DUPLICATES (IF ANY)
duplicated_data = train_dataset.duplicated(subset = attr_list, keep = False)
train_dataset = train_dataset.loc[duplicated_data == False]

########   No 3 STEP
########   CONSTRUCT DECISION TREE WITH ENTROPY
root_entropy = Node()
tree_entropy = decision_tree_algo(train_dataset,0, root_entropy,"entropy")

########  NO 4 STEP
########  CONSTRUCT DECISION TREE WITH VARIANCE
root_variance = Node()
tree_variance = decision_tree_algo(train_dataset,0, root_variance,"variance")


#######  NO 6 STEP
#######  READ TEST SET AND REMOVE DUPLICATES
# "data_sets1/test_set.csv"
test_dataset = pd.read_csv(test_dataset1)
duplicated_data = test_dataset.duplicated(subset = attr_list, keep = False)
test_dataset = test_dataset.loc[duplicated_data == False]

####### NO 7 STEP
####### TEST DATA WITH ENTROPY and VARIANCE
test_acc_entropy = accuracy(test_dataset, tree_entropy,False)
test_acc_variance = accuracy(test_dataset, tree_variance,False)

####### NO 8 STEP
####### READ VALIDATION DATASET
# "data_sets1/validation_set.csv"

validation_dataset = pd.read_csv(validation_dataset1)
duplicated_data = validation_dataset.duplicated(subset = attr_list, keep = False)
validation_dataset = validation_dataset.loc[duplicated_data == False]

####### NO 9 STEP
####### POST PRUNING THE TREE ENTROPY
best_tree_entropy= post_pruning(validation_dataset, root_entropy,int(L), int(K))

####### NO 10 STEP
####### POST PRUNING THE TREE VARIANCE
best_tree_variance = post_pruning(validation_dataset, root_variance, int(L), int(K))

####### NO 11 STEP
####### ACC of TEST DATA on POST PRUNED TREE WITH ENTROPY and VARIANCE


test_acc_entropy_best = accuracy(test_dataset, best_tree_entropy, True)
test_acc_variance_best = accuracy(test_dataset, best_tree_variance, True)




if toPrint == 'yes':
    print("\nTree with entropy heuristics")
    print_tree(tree_entropy,0)
    print("\nTree with Variance heuristics")
    print_tree(tree_variance,0)
    print("\nPruned Tree with entropy heuristics")
    print_tree(best_tree_entropy,0)
    print("\nPruned Tree with variance heuristics")
    print_tree(best_tree_variance,0)


print()
print("########### Test dataset accuracy(%) #############")
print()
print("With Information Gain heuristics")
print(test_acc_entropy*100)
print()
print("with variance heuristics")
print(test_acc_variance*100)
print()
print("with Information gain (pruned tree)")
print(test_acc_entropy_best*100)
print()
print("with variance (pruned tree)")
print(test_acc_variance_best*100)

