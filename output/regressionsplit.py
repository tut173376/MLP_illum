##Calculate the values(mean,std,var...) of RGB channels and write into each csv file
import csv 
import numpy as np

def split(listpath,outputpath):
    with open(listpath) as f:
        reader = csv.reader(f)
        data = [r for r in reader]

    with open(outputpath) as f1:
        reader1 = csv.reader(f1)
        data1 = [r1 for r1 in reader1]
        #data.pop(0)    
    k = 0
    r_data, g_data , b_data = np.empty((0,4), float),  np.empty((0,4), float),  np.empty((0,4), float)
    label =  np.empty((0,3), float)

    for i in range(len(data)):
        # n = 0
        r, g, b= [], [], []
        for j in range(k, len(data1)):
            if str(data[i]).find(str(data1[j][0]))!=-1:
                # n = n + 1
                k = k + 1
                r.append(float(data1[j][1]))
                g.append(float(data1[j][2]))
                b.append(float(data1[j][3]))
                label1 = [float(data[i][1]), float(data[i][2]), float(data[i][3])]
            else:  
                break
        r_np=np.array(r)
        g_np=np.array(g)
        b_np=np.array(b)

        filename = str(data[i])[2:14].replace('png','csv')
        # print(filename)
        r_data = np.vstack((r_data, [r_np.mean(),np.median(r_np),r_np.std(),r_np.var()]))
        g_data = np.vstack((g_data,[g_np.mean(),np.median(g_np),g_np.std(),g_np.var()]))
        b_data = np.vstack((b_data, [b_np.mean(),np.median(b_np),b_np.std(),b_np.var()]))
        label = np.vstack((label, label1))
    return r_data, g_data , b_data, label

#split("../train_list.csv","../train_input.csv","best_train_output.csv",'reg_train/')
# split("output/test_list_label.csv","output/test_output_sort.csv",'output/reg_test/')
# r_np, r_data, g_data , b_data, label = split("output/train_list_label.csv","output/train_output_sort.csv")
# print(r_np)
# print(r_data.shape,g_data.shape,label.shape)