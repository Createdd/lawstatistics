import tensorflow as tf
import numpy as np
import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ======== Generate and plot data

# Generate evidence numbers between 10 and 20
# Generate a number of convictions from the evidence with a random noise added
np.random.seed(42)
sampleSize = 200
numEvid = np.random.randint(low=10, high=50, size=sampleSize)
numConvict = numEvid * 10 + np.random.randint(low=200, high=400, size=sampleSize)

# Plot the data to get a feeling
plt.title("Number of convictions based on evidence")
plt.plot(numEvid, numConvict, "bx")  # bx = blue x
plt.xlabel("Number of Evidence")
plt.ylabel("Number of Convictions")
plt.show(block=False)  # Use the keyword 'block' to override the blocking behavior

# ======== Prepare data

# create a fumnction for normalizing values
# use 70% of the data for training (the remaining 30% shall be used for testing)
def normalize(array):
    return (array - array.mean()) / array.std()


numTrain = math.floor(sampleSize * 0.7)

# convert list to an array and normalize arrays
trainEvid = np.asanyarray(numEvid[:numTrain])
trainConvict = np.asanyarray(numConvict[:numTrain])
trainEvidNorm = normalize(trainEvid)
trainConvictdNorm = normalize(trainConvict)

testEvid = np.asanyarray(numEvid[numTrain:])
testConvict = np.asanyarray(numConvict[numTrain:])
testEvidNorm = normalize(testEvid)
testConvictdNorm = normalize(testConvict)

# ======== Set up variables and operations for TensorFlow

# define placeholders  and variables
tfEvid = tf.placeholder(tf.float32, name="Evid")
tfConvict = tf.placeholder(tf.float32, name="Convict")
tfEvidFactor = tf.Variable(np.random.randn(), name="EvidFactor")
tfConvictOffset = tf.Variable(np.random.randn(), name="ConvictOffset")

# define the operation for predicting the conviction based on evidence by adding both values
# define a loss function (mean squared error)
tfPredict = tf.add(tf.multiply(tfEvidFactor, tfEvid), tfConvictOffset)
tfCost = tf.reduce_sum(tf.pow(tfPredict - tfConvict, 2)) / (2 * numTrain)

# set a learning rate and a gradient descent optimizer
learningRate = 0.1
gradDesc = tf.train.GradientDescentOptimizer(learningRate).minimize(tfCost)

# ======== Start calculations with a session

# initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # set up iteration parameters
    displayEvery = 2
    numTrainingSteps = 50

    # Calculate the number of lines to animation
    # define variables for updating during animation
    numPlotsAnim = math.floor(numTrainingSteps / displayEvery)
    evidFactorAnim = np.zeros(numPlotsAnim)
    convictOffsetAnim = np.zeros(numPlotsAnim)
    plotIndex = 0

    # iterate through the training data
    for i in range(numTrainingSteps):

        # ======== Start training by running the session and feeding the gradDesc
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

            # store the result of each step in the animation variables
            evidFactorAnim[plotIndex] = sess.run(tfEvidFactor)
            convictOffsetAnim[plotIndex] = sess.run(tfConvictOffset)
            plotIndex += 1

    # log the optimized result
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

    # ======== Visualize the result and the training process

    # denormalize variables to be plottable again
    trainEvidMean = trainEvid.mean()
    trainEvidStd = trainEvid.std()
    trainConvictMean = trainConvict.mean()
    trainConvictStd = trainConvict.std()
    xNorm = trainEvidNorm * trainEvidStd + trainEvidMean
    yNorm = (
        sess.run(tfEvidFactor) * trainEvidNorm + sess.run(tfConvictOffset)
    ) * trainConvictStd + trainConvictMean

    # Plot the result graph
    plt.figure()

    plt.xlabel("Number of Evidence")
    plt.ylabel("Number of Convictions")

    plt.plot(trainEvid, trainConvict, "go", label="Training data")
    plt.plot(testEvid, testConvict, "mo", label="Testing data")
    plt.plot(xNorm, yNorm, label="Learned Regression")
    plt.legend(loc="upper left")

    plt.show()

    # Plot an animated graph that shows the process of optimization
    fig, ax = plt.subplots()
    line, = ax.plot(numEvid, numConvict)

    plt.rcParams["figure.figsize"] = (10, 8) # adding fixed size parameters to keep animation in scale
    plt.title("Gradient Descent Fitting Regression Line")
    plt.xlabel("Number of Evidence")
    plt.ylabel("Number of Convictions")
    plt.plot(trainEvid, trainConvict, "go", label="Training data")
    plt.plot(testEvid, testConvict, "mo", label="Testing data")

    # define an animation functon that changes the ydata
    def animate(i):
        line.set_xdata(xNorm)
        line.set_ydata(
            (evidFactorAnim[i] * trainEvidNorm + convictOffsetAnim[i]) * trainConvictStd
            + trainConvictMean
        )
        return (line,)

    # Initialize the animation with zeros for y
    def initAnim():
        line.set_ydata(np.zeros(shape=numConvict.shape[0]))
        return (line,)

    # call the animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, plotIndex),
        init_func=initAnim,
        interval=200,
        blit=True,
    )

    plt.show()
