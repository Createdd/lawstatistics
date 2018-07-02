import tensorflow as tf
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate evidence numbers between 10 and 20
np.random.seed(42)
sampleSize = 200
numEvid = np.random.randint(low=10, high=50, size=sampleSize)

# Generate number of convictions from the evidence with a random noise added
numConvict = numEvid + np.random.randint(low=3, high=10, size=sampleSize)

# Plot the numbers
plt.title('Number of convictions based on evidence')
plt.plot(numEvid, numConvict, "bx")  # bx = blue x
plt.xlabel("Number of Evidence")
plt.ylabel("Number of Convictions")
plt.show()

# normalize values


def normalize(array):
    return (array - array.mean()) / array.std()


# use 70% of the data for training (the remaining 30% shall be used for testing)
numTrain = math.floor(sampleSize * 0.7)

# convert list to an array and normalize arrays
trainEvid = np.asanyarray(numEvid[:numTrain])
trainConvict = np.asanyarray(numConvict[:numTrain])

trainEvidNorm = normalize(trainEvid)
trainConvictdNorm = normalize(trainConvict)

# do the same for the test data
testEvid = np.asanyarray(numEvid[numTrain:])
testConvict = np.asanyarray(numConvict[numTrain:])

testEvidNorm = normalize(testEvid)
testConvictdNorm = normalize(testConvict)


# ------- Start of using TensorFlow

# define placeholders which will be updated
tfEvid = tf.placeholder("float", name="Evid")
tfConvict = tf.placeholder("float", name="Convict")

# define variables for evidence and conviction during training
tfEvidFactor = tf.Variable(np.random.randn(), name="EvidFactor")
tfConvictOffset = tf.Variable(np.random.randn(), name="ConvictOffset")

# define the operation for predicting the conviction based on evidence by adding both values
tfConvictPredict = tf.add(tfEvidFactor, tfConvictOffset)

# define a loss function (mean squared error)
tfCost = tf.reduce_sum(tf.pow(tfConvictPredict-tfConvict, 2))/(2*numTrain)

# set a learning rate and a gradient descent optimizer
learningRate = 0.1
gradDesc = tf.train.GradientDescentOptimizer(learningRate).minimize(tfCost)

# initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    displayEvery = 2
    numTrainingSteps = 50

    # iterate through the training data
    for i in range(numTrainingSteps):
        # load the training data
        for (x, y) in zip(trainEvidNorm, trainConvictdNorm):
            sess.run(gradDesc, feed_dict={tfEvid: x, tfConvict: y})

        # Print status of learning
        if (i + 1) % displayEvery == 0:
            cost = sess.run(tfCost, feed_dict={
                tfEvid: trainEvidNorm, tfConvict: trainConvictdNorm})
            print("iteration #:", '%04d' % (i + 1), "cost=", "{:.9f}".format(cost),
                  "evidFactor=", sess.run(
                      tfEvidFactor), "convictOffset=", sess.run(tfConvictOffset),
                  "prediction: ", sess.run(tfConvictPredict))

    print("Optimized!")
    trainingCost = sess.run(
        tfCost, feed_dict={tfEvid: trainEvidNorm, tfConvict: trainConvictdNorm})
    print('Trained cost=', trainingCost, 'evidFactor=', sess.run(
        tfEvidFactor), 'convictOffset=', sess.run(tfConvictOffset), '\n')

    # Plot of the training and test data, and learned regression

    # Get values sued to normalized data so we can denormalize data back to its original scale
    trainEvidMean = trainEvid.mean()
    trainEvidStd = trainEvid.std()

    trainConvictMean = trainConvict.mean()
    trainConvictStd = trainConvict.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.xlabel("Number of Evidence")
    plt.ylabel("Number of Convictions")
    plt.plot(trainEvid, trainConvict, 'go', label='Training data')
    plt.plot(testEvid, testConvict, 'mo', label='Testing data')
    plt.plot(trainEvidNorm * trainEvidStd + trainEvidMean,
             (sess.run(tfEvidFactor) * trainEvidNorm +
              sess.run(tfConvictOffset)) * trainConvictStd + trainConvictMean,
             label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()
