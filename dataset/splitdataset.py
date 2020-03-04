## Comebine all the data and split data according to the list 
import csv 
import numpy as np
import csv
import os

def combine(csvfile):
    fout = open(csvfile,"a")

    for i in range(524,621,1):
        imname = '8D5U5'+ str(i)
        img_file = 'csvdata/'+ imname + '.csv'
        if os.path.exists(img_file) == True:
            f = open(img_file)
            for line in f:
                fout.write(line)
            f.close() # not really needed

    for j in range(281,904,1):
        imname = 'IMG_0'+ str(j)
        img_file = 'csvdata/'+ imname + '.csv'
        if os.path.exists(img_file) == True:
            f = open(img_file)
            for line in f:
                fout.write(line)
            f.close() # not really needed
    fout.close()

def split(csvfile, data, data1):
    with open(csvfile, mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(data1)):
            for j in range(len(data)):
                if str(data1[i]).find(str(data[j][0]))!=-1:
                    writer.writerow([data[j][0], data[j][1], data[j][2], data[j][3], 
                                    data[j][4], data[j][5], data[j][6],
                                    data[j][7],data[j][8],data[j][9]])

#combine("output(all).csv")

with open("train_list.csv") as f1:
    reader1 = csv.reader(f1)
    data1 = [r1 for r1 in reader1]

with open("test_list.csv") as f2:
    reader2 = csv.reader(f2)
    data2 = [r2 for r2 in reader2]

with open("output(all).csv") as f:
    reader = csv.reader(f)
    data = [r for r in reader]

split("train_input.csv", data, data1)
split("test_input.csv", data, data2)