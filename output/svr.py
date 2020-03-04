## Regression using SVR(RBF kernel)
import csv
import os
import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import output.regressionsplit as rg
import math
import output.sort as sort

#Angular error calculation
def error(output,label):  
    upper = (np.dot(output, label))
    lower = math.sqrt(math.fsum(output**2))*math.sqrt(math.fsum(label**2))
    rad = np.clip((upper/lower),-1.0,1.0)
    acc = math.degrees(math.acos(rad))
    return acc

def svr(BestAcc):
    with open("output/train_list_label.csv") as f1:
        reader1 = csv.reader(f1)
        data1 = [r1 for r1 in reader1]

    with open("output/test_list_label.csv") as f:
        reader = csv.reader(f)
        data = [r for r in reader]

    print('Loading data.....')
    sort.sort_csvfile("trainpatch_output.csv", 'output/train_output_sort.csv')
    sort.sort_csvfile("testpatch_output.csv", 'output/test_output_sort.csv')

    #train
    r_data, g_data , b_data, label = rg.split("output/train_list_label.csv","output/train_output_sort.csv")
    #test
    rt_data, gt_data , bt_data, tlabel = rg.split("output/test_list_label.csv","output/test_output_sort.csv")

    Y_train = label
    Y_test = tlabel

    print('Illuminant regression using SVR.....')
    C = 0.1
    gamma = 1
    svr_rbf_r = SVR(kernel='rbf', C=C, gamma=gamma)
    svr_rbf_g = SVR(kernel='rbf', C=C, gamma=gamma)
    svr_rbf_b = SVR(kernel='rbf', C=C, gamma=gamma)

    y_r = svr_rbf_r.fit(r_data, Y_train[:,0]).predict(r_data)
    y_g = svr_rbf_g.fit(g_data, Y_train[:,1]).predict(g_data)
    y_b = svr_rbf_b.fit(b_data, Y_train[:,2]).predict(b_data)

    yt_r = svr_rbf_r.predict(rt_data)
    yt_g = svr_rbf_g.predict(gt_data)
    yt_b = svr_rbf_b.predict(bt_data)

    y_train_pred = np.array([y_r, y_g, y_b])
    y_test_pred = np.array([yt_r, yt_g, yt_b])
    y_train_pred  = y_train_pred.transpose()
    y_test_pred  = y_test_pred.transpose()

    correct_train, correct_test = 0.0, 0.0
    for j in range(455):
        correct_train += error(y_train_pred[j],Y_train[j])

    for i in range(114):
        correct_test += error(y_test_pred[i],Y_test[i])
   
    trainAcc = correct_train/455
    testAcc = correct_test/114

    if((testAcc<BestAcc)==True):   
        if os.path.isfile('output/finaltrain_output.csv'):
            os.remove('output/finaltrain_output.csv')
        for j in range(455):
            with open('output/finaltrain_output.csv', mode='a',newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(data1[j])[2:14],y_train_pred[j][0],y_train_pred[j][1],y_train_pred[j][2],
                                Y_train[j][0], Y_train[j][1], Y_train[j][2],
                                error(y_train_pred[j],Y_train[j])])
        if os.path.isfile('output/finaltest_output.csv'):
            os.remove('output/finaltest_output.csv')
        for i in range(114):
            with open('output/finaltest_output.csv', mode='a',newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(data[i])[2:14],y_test_pred[i][0],y_test_pred[i][1],y_test_pred[i][2],
                                Y_test[i][0], Y_test[i][1], Y_test[i][2],
                                error(y_test_pred[i],Y_test[i])])

    print('After regression')
    print('Train angular error: {:.6f}'.format(trainAcc))
    print('Test angular error: {:.6f}'.format(testAcc))

    return trainAcc, testAcc

