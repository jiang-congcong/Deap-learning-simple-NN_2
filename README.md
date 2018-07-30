# Deap-learning-simple-NN_2
神经网络优化--自定义损失函数
神经网络优化中，自定义损失函数更灵活

例如酸奶销售，生产多了损失成本，生产少了，损失利润，假设生产成本COST为1元，利润PROFIT为9 元，实际生产数量为y,市场需求量为y_

则损失为分段函数，损失为 loss = ( y - y_ ) *COST  ( 条件为：y > y- )，loss = ( y_ - y ) * PROFIT   (条件为： y_ > y )

 python 里可以通过这行代码来完成这个计算 ： 

loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
