import os  # for doing directory operations
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

import SimpleITK as itk
import numpy as np
import csv

IMG_SIZE_PX = 50
SLICE_COUNT = 20


def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    if n == 0: #changed by Yaqin Huang
        yield l
    else:
        for i in range(0, len(l), n):
            yield l[i:i + n]


def mean(a):
    return sum(a) / len(a)


def process_data(patient, labels_df, img_px_size=50, hm_slices=20, visualize=False):
    label = labels_df.get_value(patient, 'class')
    path = data_dir + patient + '.mhd'
    slices = itk.GetArrayFromImage(itk.ReadImage(path)) #changed by Yaqin Huang

    new_slices = []
    slices = [cv2.resize(np.array(slices), (img_px_size, img_px_size))]

    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices + 2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if len(new_slices) == hm_slices + 1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices], ])))
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if visualize:
        fig = plt.figure()
        for num, each_slice in enumerate(new_slices):
            y = fig.add_subplot(4, 5, num + 1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    if label.any() == 1:
        label = np.array([0, 1])
    elif label.all() == 0:
        label = np.array([1, 0])

    return np.array(new_slices), label


#stage 1 for real.
data_dir = './input/'
patients = os.listdir(data_dir)
labels = pd.read_csv('./csv/candidates.csv', index_col=0)

much_data = []
for num, patient in enumerate(patients):
    if patient.endswith('.mhd'):
        if num % 100 == 0:
            print(num)
        try:
            img_data, label = process_data(patient[:-4], labels, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT) #changed by Yaqin Huang
            much_data.append([img_data, label])
        except KeyError as e:
            print('This is unlabeled data!')
            pass

np.save('luna-muchdata-{}-{}-{}.npy'.format(IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT), much_data)
