import os
import tensorflow as tf

version = 1
export_path = './Bootcamp_model/serve_models/{}'.format(version)

""" load trained model (.ckpt) """
tf.reset_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph(
    './Bootcamp_model/Weight/model_predict_0.61734104_0.954_.ckpt.meta')
saver.restore(
    sess, './Bootcamp_model/Weight/model_predict_0.61734104_0.954_.ckpt')

"""check operations """
[n.name for n in tf.get_default_graph().as_graph_def().node]


""" get input and softmax output tensor """
inputs = tf.get_default_graph().get_tensor_by_name('input/Placeholder:0')
predictions = tf.get_default_graph().get_tensor_by_name('Softmax/Softmax:0')

""" get tensor_info """
model_input = tf.saved_model.utils.build_tensor_info(inputs)
model_output = tf.saved_model.utils.build_tensor_info(predictions)

""" build signature definition """
signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'inputs': model_input},
    outputs={'outputs': model_output},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)


""" get builder """
builder = tf.saved_model.builder.SavedModelBuilder(export_path)

""" build meta graph and variables """
builder.add_meta_graph_and_variables(
    sess,
    [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
    }
)

""" save model (.pb) """
builder.save()


load_graph = tf.Graph()
sess = tf.Session(graph=load_graph)
tf.saved_model.loader.load(
    sess, [tf.saved_model.tag_constants.SERVING], export_path)


sess.close()
