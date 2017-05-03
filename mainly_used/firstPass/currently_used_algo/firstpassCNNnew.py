import tensorflow as tf
import numpy as np
import csv
from collections import OrderedDict
from numpy import array
import pandas as pd


IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def maxpool3d(x):
    #                          size of window     movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    #       5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


much_data = np.load('muchdatanew-50-50-20.npy')
unlabeled_data = np.load('unlabeleddata-50-50-20.npy')

train_data = much_data

#validation_data = much_data[-100:]


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]

                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    pass

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            pred = tf.argmax(prediction, 1)
            #pred2 = tf.argmax(y, 1)

            # print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        # print('Done. Finishing accuracy:')
        # print('Accuracy:', accuracy.eval({x: [i[0] for i in validation_data], y: [i[1] for i in validation_data]}))

        print('fitment percent:', successful_runs / total_runs)
        print()

        d = {}
        stage1 = sess.run(pred, feed_dict={x: [i[0] for i in unlabeled_data]})
        for patient, class_label in zip(unlabeled_data, stage1):
            #print(patient[1], class_label)
            d[patient[1]] = class_label
        od = OrderedDict(sorted(d.items()))

        # Write to CSV
        f = open('stage1_submission'+ str(IMG_SIZE_PX) + '_' + str(SLICE_COUNT) + 'new' + '.csv', 'wt')
        writer = csv.writer(f)
        writer.writerow(('id', 'cancer'))
        for patient in od:
            writer.writerow((patient, od[patient]))
        f.close()


def score_cal():
	solution = pd.read_table('stage1_solution.csv', sep=',', index_col = 0)
	result = pd.read_table('stage1_submission'+ str(IMG_SIZE_PX) + '_' + str(SLICE_COUNT) + 'new' + '.csv', sep=',', index_col = 0)

	solu = solution.cancer.tolist()
	res = result.cancer.tolist()

	total_right = 0
	total_wrong = 0
	false_positive = 0
	false_negetive = 0

	for ind in range(len(solu)):
		if solu[ind] != res[ind]:
			total_wrong += 1
		else:
			total_right += 1

		if (solu[ind] == 1) and (res[ind] == 0):
			false_negetive += 1
		elif (solu[ind] == 0) and (res[ind] == 1):
			false_positive += 1

	print("")
	print("Result: ")
	print("Total number of test dataset is: " + str(len(solu)))
	print("Number of right prediction is: " + str(total_right))
	print("Number of wrong prediction is: " + str(total_wrong))
	print("Number of false negetive prediction is: " + str(false_negetive))
	print("Number of false postive prediction is: " + str(false_positive))
	print("Overall accuracy: " + str(total_right / len(solu)))
	print("")


# Run this locally:
train_neural_network(x)
score_cal()
