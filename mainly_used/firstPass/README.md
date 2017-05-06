# Kaggle-Data-Science-Bowl-2017
Boston University EC500 C1 Team YAY

#This repo includes manual for how to use SCC, the result of our project, and the algorithm we used in our project
1. Latest Update: 5/3/2017

2. currently_used_algo folder includes all algorithm we used recently. THe CNN and Image Processing algorithms are used. change_percentage.py is used for generating datset with specific percentage of cancer patients. score_calculator.py is used for calculating the all accuracy score based on the generated csv file

3. currently_job_submission_file folder includes all job files that are used on SCC. 

4. previously_feature_change_algo folder includes the CNN algorithm that includes different number of features

5. previously_image_rezise_algo folder includes all image processing algorithm for generating different size of image dataset

6. result folder includes all the result we got so far

7. Python packages required for Image Processing part: dicom (for reading dicom files) ,os (for doing directory operations), pandas as (for some simple data analysis),
matplotlib, cv2, numpy, math

8. Python packages required for Image Processing part: tensorflow, numpy, csv, collections, numpy, pandas

9. General process:
	0. Preparetion: Download all the stage1_data ,stage1_labels.csv	and stage1_solutions.csv. (Optional extra dataset link could be found in our presentation slides, but you need to manually add it to the stage1 folder and the label csv file) 1. First, use firstpassProcessing.py or other with different images size to generate .npy files 2. Second, the optional step is to use change_percentage.py to change the structure of the dataset. Change the IMG_SIZE_PX and SLICE_COUNT to the corresponding value in the previous step. 3. Change the percentage to the percenatge you want, it will generate the new .npy file. 4. Third, use firstpassCNN.py to train the CNN and it will output the submission.csv file as the prediction result. It will calculate the accuracy and print the result in terminal



