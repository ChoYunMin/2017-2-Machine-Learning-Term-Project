
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[2]:


xy = np.loadtxt('winequality-white-2.csv', delimiter=';', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

wine_quality = 3


# In[3]:


print(x_data.shape, x_data)
print(y_data.shape, y_data)


# In[4]:


testxy = np.loadtxt('winequality-red-2.csv', delimiter=';', dtype=np.float32)
testx_data = testxy[:, 0:-1]
testy_data = testxy[:, [-1]]


# In[5]:


print(testx_data.shape, testx_data)
print(testy_data.shape, testy_data)


# In[6]:


x_standardization = x_data
for num in range(11):
    x_standardization[:, num] = (x_data[:, num] - min(x_data[:, num])) / (max(x_data[:, num]) - min(x_data[:, num]))
    
print(x_standardization.shape, x_standardization)


# In[7]:


testx_standardization = testx_data
for num in range(11):
    testx_standardization[:, num] = (testx_data[:, num] - min(testx_data[:, num])) / (max(testx_data[:, num]) - min(testx_data[:, num]))
    
print(testx_standardization.shape, testx_standardization)


# In[8]:


y_group = y_data

#0: bad(0~5), 1: normal(6), 2: good(7~10)
for num2 in range(len(y_data)):
    if y_data[num2][0] == 0 :
        y_group[num2][0] = 0
    elif y_data[num2][0] == 1 :
        y_group[num2][0] = 0
    elif y_data[num2][0] == 2 :
        y_group[num2][0] = 0
    elif y_data[num2][0] == 3 :
        y_group[num2][0] = 0
    elif y_data[num2][0] == 4 :
        y_group[num2][0] = 0
    elif y_data[num2][0] == 5 :
        y_group[num2][0] = 0
    elif y_data[num2][0] == 6 :
        y_group[num2][0] = 1
    elif y_data[num2][0] == 7 :
        y_group[num2][0] = 2
    elif y_data[num2][0] == 8 :
        y_group[num2][0] = 2
    elif y_data[num2][0] == 9 :
        y_group[num2][0] = 2
    elif y_data[num2][0] == 10 :
        y_group[num2][0] = 2

print(y_group.shape, y_group)


# In[9]:


testy_group = testy_data

#0: bad(0~5), 1: normal(6), 2: good(7~10)
for num2 in range(len(testy_data)):
    if testy_data[num2][0] == 0 :
        testy_group[num2][0] = 0
    elif testy_data[num2][0] == 1 :
        testy_group[num2][0] = 0
    elif testy_data[num2][0] == 2 :
        testy_group[num2][0] = 0
    elif testy_data[num2][0] == 3 :
        testy_group[num2][0] = 0
    elif testy_data[num2][0] == 4 :
        testy_group[num2][0] = 0
    elif testy_data[num2][0] == 5 :
        testy_group[num2][0] = 0
    elif testy_data[num2][0] == 6 :
        testy_group[num2][0] = 1
    elif testy_data[num2][0] == 7 :
        testy_group[num2][0] = 2
    elif testy_data[num2][0] == 8 :
        testy_group[num2][0] = 2
    elif testy_data[num2][0] == 9 :
        testy_group[num2][0] = 2
    elif testy_data[num2][0] == 10 :
        testy_group[num2][0] = 2

print(testy_group.shape, testy_group)


# In[10]:


X = tf.placeholder(tf.float32, shape=[None, 11], name='X-input')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y-input')


# In[11]:


with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([11, 20]), name = 'weight1')
    b1 = tf.Variable(tf.random_normal([20]), name = 'bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)


# In[12]:


with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([20, 30]), name = 'weight2')
    b2 = tf.Variable(tf.random_normal([30]), name = 'bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)


# In[13]:


with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.random_normal([30, 20]), name = 'weight3')
    b3 = tf.Variable(tf.random_normal([20]), name = 'bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3) 

    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    layer3_hist = tf.summary.histogram("layer3", layer3)


# In[14]:


with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.random_normal([20, 20]), name = 'weight4')
    b4 = tf.Variable(tf.random_normal([20]), name = 'bias4')
    layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4) 

    w4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("biases4", b4)
    layer4_hist = tf.summary.histogram("layer4", layer4)


# In[15]:


with tf.name_scope("layer5") as scope:
    W5 = tf.Variable(tf.random_normal([20, wine_quality]), name = 'weight5')
    b5 = tf.Variable(tf.random_normal([wine_quality]), name = 'bias5')
    hypothesis = tf.nn.softmax(tf.matmul(layer4, W5) + b5)

    w5_hist = tf.summary.histogram("weights5", W5)
    b5_hist = tf.summary.histogram("biases5", b5)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)


# In[16]:


with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
    cost_summ = tf.summary.scalar("cost", cost)


# In[32]:


with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


# In[33]:


prediction = tf.argmax(hypothesis, 1)
#Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
#Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)


# In[43]:


with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("C:\logs\predicting_wine_logs_r001_16")
    writer.add_graph(sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    feed = {X: x_standardization, Y: y_group}
    for step in range(71):
        
        summary, _ = sess.run([merged_summary, optimizer], feed_dict = feed)
        writer.add_summary(summary, global_step=step)
        
        if step % 10 == 0:
            print(step, sess.run(cost, feed_dict = feed))
            
    h, p, a = sess.run([hypothesis, prediction, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nPrediction (Y): ", p, "\nAccuracy: ", a)
    
    print('---------------------------------')
    
    sampleH, sampleP, sampleA = sess.run([hypothesis, prediction, accuracy], feed_dict={X: testx_standardization, Y: testy_group})
    print("sample: ", sampleH, sampleP, sampleA)

