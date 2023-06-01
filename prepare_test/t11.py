import tensorflow as tf

with tf.name_scope('graph') as scope:
    matrix1 = tf.constant([[3., 3.]], name='matrix1')  # 1 row by 2 column
    matrix2 = tf.constant([[2.], [2.]], name='matrix2')  # 2 row by 1 column
    product = tf.matmul(matrix1, matrix2, name='product')

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)  # 第一个参数指定生成文件的目录。

init = tf.global_variables_initializer()

sess.run(init)