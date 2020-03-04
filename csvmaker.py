##Write best results to csv files
import csv
import cv2
import numpy as np
import os

def outputs_file(best_train_output, best_test_output, train_path, test_path, train_batchsize, test_batchsize, train_error, test_error, epoch, path):
    with open(path + 'trainpatch_output.csv', mode='w',newline='') as color_file:
        color_writer = csv.writer(color_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_writer.writerow(['file_name', 'R', 'G', 'B', 'error','epoch', epoch])
        for i in range(len(best_train_output)):
            if train_batchsize > 1:
                for j in range(train_batchsize):
                    color_writer.writerow([train_path[i][j], best_train_output[i][j][0], best_train_output[i][j][1], best_train_output[i][j][2], train_error[i*train_batchsize + j]])
            else:
                color_writer.writerow([train_path[i], best_train_output[i][0], best_train_output[i][1], best_train_output[i][2], train_error[i*train_batchsize]])

    with open(path + 'testpatch_output.csv', mode='w',newline='') as color_file:
        color_writer = csv.writer(color_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        color_writer.writerow(['file_name', 'R', 'G', 'B', 'error','epoch', epoch])
        for i in range(len(best_test_output)):
            if test_batchsize > 1:
                for j in range(test_batchsize):
                    color_writer.writerow([test_path[i][j], best_test_output[i][j][0], best_test_output[i][j][1], best_test_output[i][j][2], test_error[i*test_batchsize + j]])
            else:
                color_writer.writerow([str(test_path[i])[2:-3], best_test_output[i][0],best_test_output[i][1],best_test_output[i][2], test_error[i]])
