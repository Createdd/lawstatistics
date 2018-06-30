import tensorflow as tf
import numpy as np
import math
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
import matplotlib.animation as animation


## Generate some evidence numbers between 10 and 20
np.random.seed(42)
num_evid = np.random.randint(low=10, high=50, size=100)


# Generate number of convictions from the evidence with a random noise added
np.random.seed(42)
num_convict = num_evid * 100.0 + np.random.randint(low=20000, high=70000, size=100)


# Plot generated hours and size
plt.plot(num_evid, num_convict, "bx") # bx = blue x
plt.ylabel("Number of Convictions")
plt.xlabel("Number of Evidence")
plt.show()