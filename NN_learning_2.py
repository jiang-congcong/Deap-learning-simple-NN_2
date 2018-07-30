#coding:utf-8
import tensorflow as tf
import numpy as np

#损失函数

BATCH_SIZE=8
seed = 23455
COST = 1
PROFIT = 9

#基于seed产生随机数
rng = np.random.RandomState(seed)

#随机数返回32行2列的矩阵，作为输入数据集
X = rng.rand(32,2)

#从X矩阵中取出一行，求和后判断结果，小于1给Y赋值1，不大于1给Y赋值0
#Y作为输入数据集的标签（正确结果）
Y = [[(x0 + x1+rng.rand()/10.0-0.05)] for (x0,x1) in X]
print("X:\n",X)
print("Y:\n",Y)

#定义神经网络的输入、参数和输出，定义向前传播过程
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))

y = tf.matmul(x,w1)


#定义损失函数及反向传播方法
#loss = tf.reduce_mean(tf.square(y-y_))
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


#生成会话，训练STEPS轮
with tf.Session() as sess :
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    #训练模型
    STEPS = 60000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 500 ==0:
            print("After %d steps the w1 is: "%i)
            print(sess.run(w1),"\n")

    print("the last result w1 is :")
    print(sess.run(w1),"\n")



