import numpy as np
from collections import OrderedDict
from numpy import array

IMG_SIZE_PX = 50
SLICE_COUNT = 20

file_name = 'muchdata-' + str(IMG_SIZE_PX) + '-' + str(IMG_SIZE_PX) + '-' + str(SLICE_COUNT) + '.npy'
much_data = np.load(file_name)
train_data = much_data

onedim = [] # Added by Yaqin Huang 4/22/2017
for ind in range(len(train_data[:, 1])):
   if (train_data[:, 1][ind][0] == 0) and (train_data[:, 1][ind][1] == 1):
       onedim.append(1)
   else:
       onedim.append(0)

one_arr = array(onedim)
changed_train_data = np.hstack((train_data, np.atleast_2d(one_arr).T))
sorted_train_data = changed_train_data[changed_train_data[:, 2].argsort()[::-1]]  # Added by Yaqin Huang 4/22/2017

num_of_cancer_count = 0
num_of_normal_count = 0

for data in sorted_train_data:
    X = data[0]
    Y = data[1]
    Z = data[2]

    if (Z == 1):  # Added by Yaqin Huang 4/22/2017
        num_of_cancer_count += 1
    else:
        num_of_normal_count += 1

                 
#print('num_of_cancer_count:' + str(num_of_cancer_count) + ' num_of_normal_count:' + str(num_of_normal_count) + ' percentage: ' + str(num_of_cancer_count / (num_of_cancer_count + num_of_normal_count)))

percentage = 0.7
num_kept = int(num_of_cancer_count / 0.5) - num_of_cancer_count
sel_train_data = sorted_train_data[:(num_of_cancer_count + num_kept), :]

np.random.shuffle(sel_train_data)

num_of_cancer_count = 0
num_of_normal_count = 0

for data in sel_train_data:
    X = data[0]
    Y = data[1]
    Z = data[2]

    if (Z == 1):  # Added by Yaqin Huang 4/22/2017
        num_of_cancer_count += 1
    else:
        num_of_normal_count += 1

#print('')
#print('After Random: ')                
#print('num_of_cancer_count:' + str(num_of_cancer_count) + ' num_of_normal_count:' + str(num_of_normal_count) + ' percentage: ' + str(num_of_cancer_count / (num_of_cancer_count + num_of_normal_count)))
np.save('muchdataper-{}-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT, percentage), sel_train_data)