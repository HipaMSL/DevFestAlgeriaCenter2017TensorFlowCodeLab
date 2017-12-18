#The MNIST data is hosted on Yann LeCun's website.
#If you are copying and pasting in the code from this tutorial, 
#start here with these two lines of code which will download and read in the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#To use TensorFlow, first we need to import it.
import tensorflow as tf

#We describe these interacting operations by manipulating symbolic variables. Let's create one:
x = tf.placeholder(tf.float32, [None, 784])

#We also need the weights and biases for our model.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#First, we multiply x by W with the expression tf.matmul(x, W). 
#This is flipped from when we multiplied them in our equation, where we had Wx, 
#as a small trick to deal with x being a 2D tensor with multiple inputs. We then 
#add b, and finally apply tf.nn.softmax.
y = tf.nn.softmax(tf.matmul(x, W) + b)

#To implement cross-entropy we need to first add a new placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])

#Then we can implement the cross-entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Now that we know what we want our model to do, it's very easy to have TensorFlow train it to do so. 
#Because TensorFlow knows the entire graph of your computations, it can automatically use 
#the backpropagation algorithm to efficiently determine how your variables affect the loss you ask 
#it to minimize. Then it can apply your choice of optimization algorithm to modify the variables 
#and reduce the loss.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#We can now launch the model in an InteractiveSession:
sess = tf.InteractiveSession()

#We first have to create an operation to initialize the variables we created:
tf.global_variables_initializer().run()

#Let's train -- we'll run the training step 1000 times!
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Finally, we ask for our accuracy on our test data.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))