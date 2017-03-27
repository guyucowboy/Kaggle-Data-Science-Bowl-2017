import dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import matplotlib.pyplot as plt
import cv2
import numpy as np


IMG_SIZE_PX = 150
SLICE_COUNT = 50

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


def process_data(patient,labels_df,img_px_size=50, hm_slices=20, visualize=False):
    
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
    
    # Assuming len(slices) > hm_slices
    
    sliceCount = 0
    groupCount = 0
    while sliceCount < len(slices):
        groupedSlices = []
        groupingFactor = (len(slices) / hm_slices) * (groupCount + 1)
        while sliceCount < groupingFactor and sliceCount < len(slices):
            groupedSlices.append(slices[sliceCount])
            sliceCount+=1
        groupCount+=1
        slice_chunk = list(map(mean, zip(*groupedSlices)))
        new_slices.append(slice_chunk)
    

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])
        
    return np.array(new_slices),label

#                                               stage 1 for real.
data_dir = './input/'
patients = os.listdir(data_dir)
labels = pd.read_csv('./stage1_labels.csv', index_col=0)

much_data = []
for num,patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:
        img_data,label = process_data(patient,labels,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        much_data.append([img_data,label])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)
