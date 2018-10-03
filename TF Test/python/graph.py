import tensorflow as tf
import numpy as np
(X_train,Y_train),(X_test,Y_test)=tf.keras.datasets.mnist.load_data()
X_train = np.expand_dims(X_train,len(X_train))
Y_train = tf.keras.utils.to_categorical(np.expand_dims(Y_train,len(Y_train)))
X_train = X_train.astype("float32")/255.0
Y_train = Y_train.astype("float32")
def gen_weights(shape):
    return tf.Variable(tf.truncated_normal(shape))
def gen_bias(shape):
    return tf.Variable(tf.truncated_normal([shape]))
def gen_cnn(input_,filter_size,in_channel,out_channel,isinput=False):
    filter_=gen_weights([filter_size,filter_size,in_channel,out_channel])
    conv_layer=tf.nn.conv2d(input_,filter_,strides=[1,1,1,1],padding="SAME")
    bias=gen_bias(out_channel)
    conv_layer+=bias
    conv_layer=tf.nn.max_pool(conv_layer,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME")
    return tf.nn.relu(conv_layer),out_channel
def flatten(input_):
    return tf.reshape(input_,[-1,input_.shape[1:].num_elements()])
def dense(input_,out):
    w=gen_weights([int(input_.shape[-1]),out])
    return tf.matmul(input_,w)+gen_bias(out)
X = tf.placeholder(tf.float32,[None]+list(X_train.shape[1:]),name="input")
Y = tf.placeholder(tf.float32,[None,10])
model,n= gen_cnn(X,5,1,54,isinput=True)
model,n=gen_cnn(model,5,n,40)
model=flatten(model)
model=dense(model,30)
model=dense(model,10)
print(X)
print(model)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
pred = tf.nn.softmax(model)
pred2=tf.argmax(pred,axis=1,name="output")
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred2,tf.argmax(Y,axis=1)),tf.float32))
batch_size = 32
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,30000,batch_size):
    x_batch,y_batch = X_train[i:i+batch_size],Y_train[i:i+batch_size]
    l,a,_ = sess.run([loss,accuracy,optimizer],feed_dict = {X:x_batch,Y:y_batch})
    if not i%100:
        print("Step #{}:, Loss: {}, Accuracy: {}".format(*map(str,[i,l,a])))
g = sess.graph.as_graph_def()
tf.train.write_graph(g,"tmp/model","tf_graph.pb",False)
tf.train.Saver().save(sess,"tmp/model/weights.ckpt")
#:sess.run(pred2,feed_dict={X:X_test[1].reshape(1,28,28,1)})
