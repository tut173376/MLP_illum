import csv 
from PIL import Image
import numpy as np
import operator 

data = []

def img_num(x):
    return(x[4:8])

def img_name(x):
    return(x[0:1])

def sort_csvfile(input_filename, output_filename):

    with open(input_filename) as f:
        reader = csv.reader(f)
        data = [r for r in reader]
        data.pop(0) #remove header
        
    for j in range(len(data)):
        del data[j][4:6]

    for i in range(len(data)):
        data[i].append(img_name(data[i][0]))
        data[i].append(int(img_num(data[i][0])))
        # print(data[i])
        # exit()
    data.sort(key = operator.itemgetter(4,5))

    with open(output_filename, mode='w',newline='') as f:
        file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #file_writer.writerow(['file_name', 'R', 'G', 'B'])
        
        for i in range(len(data)):
            file_writer.writerow([data[i][0],data[i][1],data[i][2],data[i][3]])

# sort_csvfile("best_test_output.csv", 'best_test_output_sort.csv')
# sort_csvfile("best_train_output.csv", 'best_train_output_sort.csv')
