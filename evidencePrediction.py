import tensorflow as tf
import numpy as np
import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate evidence numbers between 10 and 20
np.random.seed(42)
sampleSize = 200
numEvid = np.random.randint(low=10, high=50, size=sampleSize)

# Generate number of convictions from the evidence with a random noise added
numConvict = numEvid * 10 + np.random.randint(low=200, high=400, size=sampleSize)

# Plot the numbers
plt.title("Number of convictions based on evidence")
plt.plot(numEvid, numConvict, "bx")  # bx = blue x
plt.xlabel("Number of Evidence")
plt.ylabel("Number of Convictions")
plt.show(block=False)  # Use the keyword 'block' to override the blocking behavior

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


# define placeholders which will be updated
tfEvid = tf.placeholder(tf.float32, name="Evid")
tfConvict = tf.placeholder(tf.float32, name="Convict")

# define variables for evidence and conviction during training
tfEvidFactor = tf.Variable(np.random.randn(), name="EvidFactor")
tfConvictOffset = tf.Variable(np.random.randn(), name="ConvictOffset")

# define the operation for predicting the conviction based on evidence by adding both values
tfPredict = tf.add(tf.multiply(tfEvidFactor, tfEvid), tfConvictOffset)

# define a loss function (mean squared error)
tfCost = tf.reduce_sum(tf.pow(tfPredict - tfConvict, 2)) / (2 * numTrain)

# set a learning rate and a gradient descent optimizer
learningRate = 0.1
gradDesc = tf.train.GradientDescentOptimizer(learningRate).minimize(tfCost)

# initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    displayEvery = 2
    numTrainingSteps = 50

    # Calculate the number of lines to animation
    numPlotsAnim = math.floor(numTrainingSteps / displayEvery)
    # Add storage of factor and offset values from each epoch
    evidFactorAnim = np.zeros(numPlotsAnim)
    convictOffsetAnim = np.zeros(numPlotsAnim)
    plotIndex = 0

    # iterate through the training data
    for i in range(numTrainingSteps):
        # load the training data
        for (x, y) in zip(trainEvidNorm, trainConvictdNorm):
            sess.run(gradDesc, feed_dict={tfEvid: x, tfConvict: y})

        # Print status of learning
        if (i + 1) % displayEvery == 0:
            cost = sess.run(
                tfCost, feed_dict={tfEvid: trainEvidNorm, tfConvict: trainConvictdNorm}
            )
            print(
                "iteration #:",
                "%04d" % (i + 1),
                "cost=",
                "{:.9f}".format(cost),
                "evidFactor=",
                sess.run(tfEvidFactor),
                "convictOffset=",
                sess.run(tfConvictOffset),
            )

            # Save the fit size_factor and price_offfset to allow animation of learning process
            evidFactorAnim[plotIndex] = sess.run(tfEvidFactor)
            convictOffsetAnim[plotIndex] = sess.run(tfConvictOffset)
            plotIndex += 1

    print("Optimized!")
    trainingCost = sess.run(
        tfCost, feed_dict={tfEvid: trainEvidNorm, tfConvict: trainConvictdNorm}
    )
    print(
        "Trained cost=",
        trainingCost,
        "evidFactor=",
        sess.run(tfEvidFactor),
        "convictOffset=",
        sess.run(tfConvictOffset),
        "\n",
    )

    # Plot of the training and test data, and learned regression

    # Get values sued to normalized data so we can denormalize data back to its original scale
    trainEvidMean = trainEvid.mean()
    trainEvidStd = trainEvid.std()

    trainConvictMean = trainConvict.mean()
    trainConvictStd = trainConvict.std()

    xNorm = trainEvidNorm * trainEvidStd + trainEvidMean
    yNorm = (
        sess.run(tfEvidFactor) * trainEvidNorm + sess.run(tfConvictOffset)
    ) * trainConvictStd + trainConvictMean

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.xlabel("Number of Evidence")
    plt.ylabel("Number of Convictions")
    plt.plot(trainEvid, trainConvict, "go", label="Training data")
    plt.plot(testEvid, testConvict, "mo", label="Testing data")
    plt.plot(numEvid, numConvict, "bx", label="original")
    plt.plot(xNorm, yNorm, label="Learned Regression")

    plt.legend(loc="upper left")
    plt.show()

    # Plot another graph that animation of how Gradient Descent sequentially adjusted size_factor and price_offset to
    # find the values that returned the "best" fit line
    fig, ax = plt.subplots()
    line, = ax.plot(numEvid, numConvict)

    plt.rcParams["figure.figsize"] = (10, 8)
    plt.title("Gradient Descent Fitting Regression Line")
    plt.xlabel("Number of Evidence")
    plt.ylabel("Number of Convictions")
    plt.plot(trainEvid, trainConvict, "go", label="Training data")
    plt.plot(testEvid, testConvict, "mo", label="Testing data")

    def animate(i):
        line.set_xdata(xNorm)
        line.set_ydata(
            (evidFactorAnim[i] * trainEvidNorm + convictOffsetAnim[i]) * trainConvictStd
            + trainConvictMean
        )
        return (line,)

    # Init only required for blitting to give a clean slate
    def initAnim():
        line.set_ydata(np.zeros(shape=numConvict.shape[0]))  # set y's to 0
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, plotIndex),
        init_func=initAnim,
        interval=200,
        blit=True,
    )

    plt.show()
