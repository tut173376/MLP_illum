from os import walk
import csv
import random

def writecsv_file(filename,data):
    with open(filename, mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(data)):
            writer.writerow(data[i]) 

def remake(cl_num, shuf=False):
    mypath = 'csvdata/'
    f = []
    cl_array = []
    cl_tmp = []
    k = 0

    # To read all the files in the folder
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    
    # Open each csv file and read the data
    for i in range(len(f)):
        data = []
        l = 0
        with open('csvdata/' + str(f[i])) as csvfile:
            reader = csv.reader(csvfile)
            data = [r for r in reader]
        
        # Randomly shuffle the data
        if shuf == True: random.shuffle(data)

        num_remove = len(data) % cl_num
        for j in range(num_remove-1):
            sel = random.randint(0,len(data)-1)
            data.remove(data[sel])

        for k in range(len(data)):
            if(len(cl_tmp) == 6*cl_num): 
                cl_tmp.extend(data[k][7:10])
                cl_tmp.insert(0, data[k][0])
                cl_array.append(cl_tmp)
                cl_tmp = []
                cl_tmp.extend(data[k][1:7]) 
            else:
                cl_tmp.extend(data[k][1:7])
    
    writecsv_file("output1(all).csv", cl_array)           


def split(csvfile, data, data1):
    with open(csvfile, mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(data1)):
            for j in range(len(data)):
                if str(data1[i]).find(str(data[j][0]))!=-1:
                    writer.writerow(data[j])        


remake(20)

with open("train_list.csv") as f1:
    reader1 = csv.reader(f1)
    data1 = [r1 for r1 in reader1]

with open("test_list.csv") as f2:
    reader2 = csv.reader(f2)
    data2 = [r2 for r2 in reader2]

with open("output1(all).csv") as f:
    reader = csv.reader(f)
    data = [r for r in reader]
        
split("train_input1.csv", data, data1)
split("test_input1.csv", data, data2)

        