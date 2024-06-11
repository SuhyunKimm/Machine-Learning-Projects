import csv
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

def test(tree, test_set, ig_idx_arr, evaluation) :
    accurate_cnt = 0
    # tp, fp, fn, tn
    for test_data in test_set:
        node = tree[test_data[ig_idx_arr[0]]]
        for i in range(1,len(ig_idx_arr)) :
            node = node[test_data[ig_idx_arr[i]]]
            if node in evaluation : # node is predicted (in decision tree) and evaluation is the actual value
                break
        pred_idx = evaluation.index(node)
        actual_idx = evaluation.index(test_data[-1])
        proportion_arr[actual_idx] += 1
        
        if node == test_data[-1] :
            for j in range(len(confusion_metrics)) :
                if j == pred_idx or j == actual_idx:
                    confusion_metrics[j][0] +=1 # true positive
                else :
                    confusion_metrics[j][3] +=1 # true negative
            accurate_cnt+=1
        else :
            for j in range(len(confusion_metrics)) :
                if j == pred_idx : #accep unacpp
                    confusion_metrics[j][1] += 1 # false positive very good pred, good actual
                elif j == actual_idx :
                    confusion_metrics[j][2] += 1 # false negative
                else :
                    confusion_metrics[j][3] +=1 # true negative
                    
    return accurate_cnt, confusion_metrics
    
def count_attribute(col,att_val) : ## counting the number of rows that has i-th attribute from att_val array in the 'col' column
    arr = [0 for i in range(len(att_val))]
    for i in range(len(att_val)) :
        arr[i] = col.count(att_val[i])
    return arr
    
def logarithmic_calculator(n, arr) : ## calculate lagorithmic value of the subset
    total = 0
    for i in arr :
        if i == 0 :
            continue
        num = i/n
        total = total - (num* math.log(num,2))
    return total

def cal_ig(val_col,attr_col,eva_col) : ## calculate IG value of each subset
    num_arr = []
    log_arr = []
    total = 0
    size = len(val_col)
    for i in range(0,len(attr_col)) :
        temp = []
        for j in range(0,len(eva_col)) :
            temp.append([attr_col[i],eva_col[j]])
        cnt_arr = count_attribute(val_col,temp)
        h = logarithmic_calculator(sum(cnt_arr),cnt_arr)
        log_arr.append(h)
        num_arr.append(sum(cnt_arr))
    for i in range(len(num_arr)) :
        total += (num_arr[i]/size * log_arr[i])
    return total

def get_plurality_value(ex,att) :
    cnt_arr = [0 for i in range(len(att))]
    for j in range(len(ex)) :
        cnt_arr[att.index(ex[j][0])]+= 1
    return cnt_arr.index(max(cnt_arr))
    
def get_importance_value(entropy_s, examples, attributes, evaluation) :
    ig_arr = []
    for i in range(0,len(attributes)) :
        val_col = [[j[i],j[len(attributes)]] for j in examples]
        total = cal_ig(val_col,attributes[i],evaluation)
        ig_arr.append(entropy_s-total)
    return ig_arr

def check_all_same_classification(examples) :
    first = examples[0]
    for i in range(len(examples)) :
        if examples[i] != first :
            return False
    return True

def decision_tree_learning(examples, attributes, evaluation, parent_examples,parent_node,parent_name,ig) :
    if len(examples) == 0 : ## examples is empty
        plurality_v = parent_examples[0][-1]
        return plurality_v
    elif len(examples[0]) ==1 and check_all_same_classification(examples[0]) :## all examples have the same classification -> 한쪽으로 True거나 False로 쏠린 경우
        return examples[0][0] ## return that classification
    elif len(attributes) == 0 : ## attributes is empty
        result = attributes[0][get_plurality_value(examples,attributes[0])]
        return result
    else : 
        ## find the attribute that has maximum importance value
        max_idx = ig.index(max(ig))
        selected_attribute = attributes[max_idx]
        new_attributes = attributes[0:max_idx]+attributes[max_idx+1:]
        new_ig = ig[0:max_idx] + ig[max_idx+1:]
        ## making tree with a root of the attribute with maximum importance value
        new_node = {}
        selected = examples[0][max_idx]
        visited = [selected]
            
        for i in range(-1,len(selected_attribute)) :
            if i > -1 and selected_attribute[i] not in visited:
                selected = selected_attribute[i]
                visited.append(selected)
            new_node[selected] = ""
            exs = []
            for k in range(len(examples)) :
                if examples[k][max_idx] == selected :
                    exs.append(examples[k])
            new_val_col = [[j[max_idx],j[len(new_attributes)+1]] for j in exs]
            new_exs = copy.deepcopy(exs)
            for j in range(len(exs)) :
                new_exs[j].pop(max_idx)
            
            if(len(new_val_col) > 0) :
                node = decision_tree_learning(new_exs,new_attributes,evaluation,exs,new_node,selected,new_ig)
            else :
                node = decision_tree_learning([],[],evaluation,parent_examples,new_node,selected,new_ig)

            new_node[selected] = node
        parent_node[parent_name] = new_node
        return new_node
            

        
        
## read the data
file = open('car.csv','r')
csvreader = csv.reader(file)

## process the data
data =[]
header = next(csvreader)
test_set = []
train_set = []
ig_arr = []
entropy_s = 0
attribute1 = [['vhigh','high','med','low'], ## buying price
             ['vhigh','high','med','low'], ## cost of maintenance
             ['2','3','4','5more'], ## num of doors
             ['2','4','more'], ## capacity of persons to carry
             ['small','med','big'], ## relative size of luggage boot
             ['low','med','high']] ## estimated safety value

attribute2 = [['blonde','dark','red'],
              ['average','tall','short'],
              ['light','average','heavy'],
              ['none','used']]

#evaluation = ['yes','no']
evaluation = ['unacc','acc','good','vgood']
attribute = attribute1
## number of attributes except the evaluation
attr_size = len(attribute)
for row in csvreader : 
    data.append(row)

## split the dataset into test and train set
random.shuffle(data)
train_set = data[:int((len(data)+1)*0.7)]
#train_set = data
test_set = data[int((len(data)+1)*0.7):]

## decision tree learning
accuracy_arr = []
proportion_arr = [0 for i in range(len(evaluation))]
confusion_metrics = [[0,0,0,0] for i in range(len(evaluation))]

for i in range(6) :
    if i < 5 :
        sub_train_set = train_set[:200*(i+1)]
    else :
        sub_train_set = train_set
        ## find the entropy value of all pairs in the training set
    eva_col = [i[attr_size] for i in sub_train_set]

    count_arr = count_attribute(eva_col,evaluation)

    entropy_s = logarithmic_calculator(len(eva_col),count_arr)
    importance_arr = get_importance_value(entropy_s,sub_train_set,attribute,evaluation)

    ig_idx_arr = np.argsort(np.argsort(np.argsort(importance_arr)[::-1]))
    tree = {}
    tree = decision_tree_learning(sub_train_set,attribute,evaluation,[],tree,"root",importance_arr)
    accurate,confusion_metrics = test(tree,test_set,ig_idx_arr,evaluation)
    accuracy_arr.append(accurate / len(test_set) * 100)
    
print("decision tree")
print(tree)

proportion_arr = [0 for i in range(len(evaluation))]
confusion_metrics = [[0,0,0,0] for i in range(len(evaluation))]
accurate, confusion_metrics = test(tree,test_set,ig_idx_arr,evaluation)
cf_attr = ['TP', 'FP', 'FN', 'TN']

## print the result
print()
print("size of training set : ", len(train_set))
print("size of test set : ",len(test_set))
print("accuracy and confusion matrix for each class : ")

total_precision = 0
total_weighted_precision = 0
total_recall = 0
total_weighted_recall = 0
total_f1 = 0
total_weighted_f1 = 0

for i in range(len(evaluation)) :
    print('----- ',evaluation[i],' -----')
    total = 0
    for j in range(len(cf_attr)) :
        print(cf_attr[j], ' = ',confusion_metrics[i][j], end = ', ')
        total += confusion_metrics[i][j]
    accuracy = (confusion_metrics[i][0] + confusion_metrics[i][3]) / total
    precision = confusion_metrics[i][0] / (confusion_metrics[i][0] + confusion_metrics[i][1])
    total_precision += precision
    total_weighted_precision += (precision * proportion_arr[i])
    recall = confusion_metrics[i][0] / (confusion_metrics[i][0] + confusion_metrics[i][2])
    total_recall += recall
    total_weighted_recall += (recall * proportion_arr[i])
    f1_score = (2 * precision * recall) / (precision + recall)
    total_f1 += f1_score
    total_weighted_f1 += (f1_score * proportion_arr[i])
    
    print()
    print('accuracy : ', round(accuracy,4))
    print('precision : ',round(precision,4))
    print('recall : ',round(recall,4))
    print('F1-score : ',round(f1_score,4))
print()

total_of_matrix = sum(proportion_arr)

print('macro average precision : ', round(total_precision / len(evaluation),4))
print('weighted average precision : ', round(total_weighted_precision / total_of_matrix,4))

print()
print('macro average recall : ', round(total_recall / len(evaluation),4))
print('weighted average recall : ', round(total_weighted_recall / total_of_matrix,4))

print()
print('macro average F1-score : ', round(total_f1 / len(evaluation),4))
print('weighted average F1-score : ',  round(total_weighted_f1  / total_of_matrix,4))

print()
print("total accuracy : ", round(accurate / len(test_set) * 100,4), "%")
print("learning curve : ")
print(accuracy_arr)

plt.xlabel('number of trained samples')
plt.ylabel('accuracy')
plt.plot([200,400,600,800,1000,1210],accuracy_arr)
plt.show()
