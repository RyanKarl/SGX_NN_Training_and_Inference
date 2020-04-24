import csv
import sys
import os
import subprocess

counter = 0
filenum = 0

with open(sys.argv[1], newline='') as csvfile:
    filestream = csv.reader(csvfile, delimiter=',')
    for row in filestream:

        if counter % int(sys.argv[2]) != 0:
            with open(sys.argv[1][:-4] + "_" + sys.argv[2] + "_" + str(filenum) + ".csv", 'a', newline='') as newcsvfile:
                filewriter = csv.writer(newcsvfile, delimiter=',')
                filewriter.writerow(row)
        else:
            filenum+=1
            with open(sys.argv[1][:-4] + "_" + sys.argv[2] + "_" + str(filenum) + ".csv", 'a', newline='') as newcsvfile:
                newfilewriter = csv.writer(newcsvfile, delimiter=',')
                newfilewriter.writerow(row)

        counter+=1

dirname = "batch" + str(sys.argv[1])[:-4] + str(sys.argv[2])
subprocess.call(["mkdir", dirname])
dirname = dirname + "/"
for i in range(filenum):
    filename = str(sys.argv[1])[:-4] + "_" + sys.argv[2] + "_" + str(i+1) + ".csv"
    subprocess.call(["mv", filename, dirname])
