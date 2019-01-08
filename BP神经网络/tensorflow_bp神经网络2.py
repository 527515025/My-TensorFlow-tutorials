from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
# Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 100
display_step = 100
 
# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
n_hidden_3 = 10 # 3rd layer number of neurons
n_hidden_4 = 10 # 4th layer number of neurons
n_hidden_5 = 10 # 5th layer number of neurons
num_input = 2 # data input (img shape: 2)
num_classes = 2 # total classes (2 digits)
 
# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
 
# Store layers weight & bias
with tf.name_scope('parameters'):
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
        'out': tf.Variable(tf.random_normal([n_hidden_5, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
 
# Create model
def neural_net(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['h5']), biases['b5']))
    
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
    return out_layer
 
def classifier(x,y):
#    input the expression you want 
    if x**2+y**2<0.1 or 5*(x-1.1)**4<y:
        return True
    return False
 
# Construct model
logits = neural_net(X)
prediction = tf.nn.sigmoid(logits)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
 
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
 
loss_data=[]
acc_data=[]
x_data=range(1,num_steps+1)
 
# Start training
with tf.Session() as sess:
 
    # Run the initializer
    sess.run(init)
    
    for step in range(1, num_steps+1):
        batch_x=np.random.random((batch_size,2))
        batch_y=np.zeros((batch_size,2))
        for i in range(batch_size):
            if classifier(batch_x[i,0],batch_x[i,1]):
                batch_y[i,0]=1
            else:
                batch_y[i,1]=1
        learning_rate=np.exp(-0.1*step)*0.1
        
        # Run optimization op (backprop)
        train,loss,acc=sess.run([train_op,loss_op,accuracy], feed_dict={X: batch_x, Y: batch_y})
        loss_data.append(loss)
        acc_data.append(acc)
 
    print("Optimization Finished!")
    plt.plot(x_data,loss_data)
    plt.show()
    plt.plot(x_data,acc_data)
    plt.show()
 
    xx = np.linspace(0,1,100)  
    yy = np.linspace(0,1,100)  
    ZZ = np.zeros((100,100))
    XX,YY = np.meshgrid(xx, yy)  
    rightsum = 0
    for i in range(100):
        for j in range(100):
            inp=np.zeros((1,2))
            inp[0,0]=XX[i,j]
            inp[0,1]=YY[i,j]
            poss=sess.run(prediction,feed_dict={X: inp})
            if poss[0,0]>poss[0,1]:
                ZZ[i,j]=1
                if classifier(inp[0,0],inp[0,1]):
                    rightsum+=1
            else:
                ZZ[i,j]=0
                if classifier(inp[0,0],inp[0,1])==False:
                    rightsum+=1
    print('Testing Accuracy',rightsum/100,'%')
    plt.contourf(XX,YY,ZZ,2,colors=('r','w','b')) 


# https://blog.csdn.net/Cfather/article/details/79254717
