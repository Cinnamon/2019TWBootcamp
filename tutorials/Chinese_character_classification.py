import tensorflow as tf
'''自己搭一個Densenet'''

'''定義Densenet 中的Dense Stage'''
def Dense_Stage(inputs_, depth=64, repeat=1):
    for _ in range(repeat):
        X_input=inputs_
        X=tf.layers.conv2d(inputs_,depth,(1,1),strides=(1,1),activation=tf.nn.leaky_relu,**kwargs)
        X=tf.layers.batch_normalization(X)
        X=tf.layers.separable_conv2d(X,depth,(3,3),padding='SAME')
        X=tf.nn.leaky_relu(X)
        X=tf.layers.batch_normalization(X)
        X=tf.concat([X_input,X],3)
        inputs_=X
    return X

'''定義Densenet 中的 Transition_Layers'''

def Transition_Layers(inputs_, size=[1,2,2,1], stride=[1,2,2,1], depth=128):
    X=tf.layers.conv2d(inputs_,depth,(1,1),strides=(1,1),activation=tf.nn.leaky_relu,**kwargs)
    X=tf.nn.max_pool(X,size,stride,padding='SAME')
    X=tf.layers.batch_normalization(X)
    return X
'''計算參數量'''
def get_num_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters


'''開始搭建模型'''
tf.reset_default_graph()
kwargs = {'padding':'same', 'kernel_regularizer':tf.contrib.layers.l2_regularizer(0.003),}

'''定義輸入層'''
with tf.name_scope('input'):
    '''調整input size'''
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 3])
    '''Label-Triplet 是index格式'''
    y_tru_for_trip = tf.placeholder(tf.float32, [None])
    '''Softmax-Label 是One-Hot '''
    y_true_soft=tf.placeholder(tf.float64, [None,13925])


'''神經網路'''
with tf.name_scope('stem'):
    X=tf.layers.conv2d(inputs,64,(3,3),strides=(1,1),activation=tf.nn.leaky_relu,name='stem_1',**kwargs)
    X=tf.layers.batch_normalization(X,name='stem_1_batch_normal')
    #X=tf.nn.leaky_relu(X,name='stem_1_relu')
    print(X)

          
    X=tf.layers.conv2d(X,64,(3,3),strides=(1,1),activation=tf.nn.leaky_relu,name='stem_2',**kwargs)
    X=tf.layers.batch_normalization(X,name='stem_2_batch_normal')
    #X=tf.nn.leaky_relu(X,name='stem_2_relu')
    print(X)
    
    X=tf.layers.separable_conv2d(X,64,(3,3),padding='SAME')
    X=tf.nn.leaky_relu(X)
    X=tf.layers.batch_normalization(X)
    print(X)

    X=tf.layers.conv2d(X,128,(3,3),strides=(1,1),activation=tf.nn.leaky_relu,name='stem_3',**kwargs)
    X=tf.layers.batch_normalization(X,name='stem_3_batch_normal')
    #X=tf.nn.leaky_relu(X,name='stem_3_relu')
    #X=tf.layers.dropout(X,rate=0.2)
    print(X)
    
    X=tf.layers.separable_conv2d(X,128,(3,3),padding='SAME')
    X=tf.nn.leaky_relu(X)
    X=tf.layers.batch_normalization(X)
    print(X)
    
    X=tf.nn.max_pool(X,[1,2,2,1],[1,2,2,1],padding='SAME')
    X=tf.layers.dropout(X,rate=0.2)
with tf.name_scope('Dense_Stage_0'):
    Share=Dense_Stage(X,32,6)
    #print(X)
with tf.name_scope('Transition_0'):
    X=Transition_Layers(Share)
    #print(X)
        
with tf.name_scope('Dense_Stage_1'):
    X=Dense_Stage(X,48,6)
    #print(X)
    
with tf.name_scope('Transition_1'):
    X=Transition_Layers(X)
    #print(X)
    
with tf.name_scope('Dense_Stage_2'):
    X=Dense_Stage(X,64,8)
    #print(X)
    
with tf.name_scope('Transition_2'):
    X=Transition_Layers(X,[1,1,1,1],[1,1,1,1],256)
    #print(X)
    
with tf.name_scope('Dense_Stage_3'):
    X=Dense_Stage(X,80,8)
    #print(X)

with tf.name_scope('Transition_3'):
    X=Transition_Layers(X,[1,2,2,1],[1,2,2,1],256)
    #print(X)
with tf.name_scope('Dense_Stage_4'):
    X=Dense_Stage(X,80,2)
    #print(X)
    
with tf.name_scope('FCL'):
    X=tf.keras.layers.GlobalAveragePooling2D()(X)
    model=tf.layers.dense(X,512,name='out_put',activation=tf.nn.leaky_relu)
    model=tf.layers.batch_normalization(model)
    #print(model)
    
'''透過Softmax訓練'''

with tf.name_scope('Softmax'):
    #Softmax
    Dense_share=Transition_Layers(Share,depth=256)
    #print(Dense_share)
    Dense_share=Dense_Stage(Dense_share,48,4)
    #print(Dense_share)
    Flatten_soft =tf.layers.Flatten()(Dense_share)
    
    #print(Flatten_soft)
    output=tf.layers.dense(Flatten_soft,13925)
    prediction=tf.nn.softmax(output)
    loss_soft=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true_soft,logits=output,name='loss_'))
    

    optimizer_soft=tf.train.AdamOptimizer(0.0001,name='Adam_Soft').minimize(loss_soft)
    
'''透過Triplet訓練'''

with tf.name_scope('Triplet'):
    ##Semi_hard_Triplet by tensorflow
    output_semi=tf.layers.dense(model,256)
    print(output_semi)
    prediction_semi=tf.nn.l2_normalize(output_semi,axis=1)
    
    loss_semi=tf.contrib.losses.metric_learning.triplet_semihard_loss(y_tru_for_trip,prediction_semi)
    
    opt_func_trip = tf.train.AdamOptimizer(0.0001,name='Adam_Triplet')

    tvars_trip = tf.trainable_variables()

    grads_trip, trip_ = tf.clip_by_global_norm(tf.gradients(loss_semi, tvars_trip), 1)

    optimizer_trim_trip= opt_func_trip.apply_gradients(zip(grads_trip, tvars_trip))

'''合併訓練'''

    
with tf.name_scope('Total_Loss'):
    Total_loss=loss_soft+5*loss_semi
    
    opt_func = tf.train.AdamOptimizer(0.0001,name='Adam_Both')

    tvars = tf.trainable_variables()

    grads, _ = tf.clip_by_global_norm(tf.gradients(Total_loss, tvars), 2)

    optimizer_trim= opt_func.apply_gradients(zip(grads, tvars))

init=tf.global_variables_initializer()

print('Model....Preparing..Done')
