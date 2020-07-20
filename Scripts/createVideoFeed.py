import numpy as np

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

import cv2
import skvideo
import skvideo.io
from skvideo.io import vread
from skvideo.io import vwrite

PATH_TO_CKPT = './preTrainedModelx/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'label_map.pbtxt')

NUM_CLASSES = 1
sys.path.append("..")

frames2 = []

def detect_in_video():
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            videogen = skvideo.io.vreader('myVideo.mkv')
            frames = list(videogen)
                
            for i,frame in enumerate(frames[:500]):
                
                if i%100 ==0:
                    print(f'{i}/{len(frames)}')
                
                image_np_expanded = np.expand_dims(frame, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2,
                    min_score_thresh=.20)

                frames2.append(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            skvideo.io.vwrite("outputVideoX.mkv", frames2)
            print('Complete!')

detect_in_video()