# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:30:32 2018

@author: tdlerue
"""

import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import os
import pandas as pd

# list containing all filenames
filenames = []
meetData = np.zeros(1)
count = 0

# create list with filesnames
for root, dirs, files in os.walk("."):
    for filename in files:
        # detect specific files in folder
        if 'index' not in filename and 'Conversion' not in filename and 'info' not in filename:
            # add file
            if count == 0:
                # reading the very first file will to initialize the filenames list
                count = count + 1
                filenames = [filename]
            else:
                filenames.append(filename)

filenames = sorted(filenames)
# remember the first date of the list
dagName = filenames[0][:11]
numberOfFiles = [1]
countDays = 0

# check the number of files for a unique day
for filename in filenames:
    if filename[:11] != dagName:
        # add a new day
        numberOfFiles.append(1)
        countDays = countDays + 1
        dagName = filename[:11]
    else:
        # count number of files within the same day
        numberOfFiles[countDays] = numberOfFiles[countDays] + 1

print(numberOfFiles)
print()
print(countDays)

# extract a couple of days
dag = filenames[0]
for filename in filenames[:247]:
    tdms_file = TdmsFile(filename)
    if len(meetData) == 1:
        channel = tdms_file.object(str(filename[11:13]) + 'h', 'S [VA]')
        meetData = channel.data
    else:
        channel = tdms_file.object(str(filename[11:13]) + 'h', 'S [VA]')
        meetData = np.append(meetData, channel.data)

print(len(meetData))
plt.figure()
plt.plot(meetData)

df_meetData = pd.DataFrame(meetData)
df_meetData.to_csv('testsample.csv', sep=',', index=False)

'''
filenameS = "2018_07_08_12.tdms"
tdms_file = TdmsFile(filenameS)
root_object = tdms_file.object()


channel = tdms_file.object('12h', 'S [VA]')
data = channel.data
print(data)'''
