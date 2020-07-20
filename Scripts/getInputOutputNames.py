import tensorflow as tf
gf = tf.GraphDef()   
m_file = open('preTrainedModelx/frozen_inference_graph.pb','rb')
gf.ParseFromString(m_file.read())

with open('somefilex.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')