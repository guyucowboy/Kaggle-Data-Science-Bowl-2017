import tensorflow as tf
import numpy as np
import csv
from collections import OrderedDict

IMG_SIZE_PX = 50
SLICE_COUNT = 20

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                          size of window     movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x):
               #       3 x 3 x 3 patches, 1 channel, 64 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,64])),
               #       3 x 3 x 3 patches, 64 channels, 128 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,64,128])),
               'W_fc':tf.Variable(tf.random_normal([108160,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([64])),
               'b_conv2':tf.Variable(tf.random_normal([128])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 108160])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

much_data = np.load('muchdata-50-50-20.npy')
unlabeled_data = np.load('unlabeleddata-50-50-20.npy')

train_data = much_data[:-100]
validation_data = much_data[-100:]

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=prediction,labels=y) )
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
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    #print(str(e))
            
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            pred = tf.argmax(prediction, 1)
            #pred2 = tf.argmax(y, 1)

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)
        print()

        #print(sess.run(correct, feed_dict={x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        #print()
        #print(sess.run(pred2, feed_dict={x:[i for i in unlabeled_data]}))

        d = {}
        stage1 = sess.run(pred, feed_dict={x:[i[0] for i in unlabeled_data]})
        for patient,class_label in zip(unlabeled_data,stage1):
            print(patient[1], class_label)
            d[patient[1]] = class_label
        od = OrderedDict(sorted(d.items()))

        # Write to CSV
        f = open('stage1_submission.csv', 'wt')
        writer = csv.writer(f)
        writer.writerow(('id', 'cancer'))
        for patient in od:
            writer.writerow((patient, od[patient]))
        f.close()

# Run this locally:
train_neural_network(x)