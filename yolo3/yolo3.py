"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.utils import letterbox_image
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
gpu_num=1

class YOLO(object):
    def __init__(self, log,data, ss):
        self.model_path = os.path.join(log,'trained_weights_final.h5') # model path or trained weights path
        self.anchors_path = os.path.join(data,'yolo_anchors.txt')
        self.classes_path = os.path.join(data,'player_classes.txt')
        self.score = 0.3
        self.iou = 0.45
        self.class_names = 'o'
        #self.anchors = self._get_anchors()
        YOLO_ANCHORS = np.array(
            ((0.7389  , 0.2602), (0.3653, 0.2258), (0.1903  , 0.1437),
             (0.1903 , 0.1352), (0.2278, 0.1078),(0.3097   ,0.0750),
            (0.1514 , 0.1383),(0.1546 ,  0.0836),(0.1083, 0.0766)))
        anchors = YOLO_ANCHORS*ss
        anchors=anchors[[8,7,6,5,4,3,2,1,0],:]
        self.anchors=anchors
        self.sess = K.get_session()
        self.max_boxes = 4
        self.model_image_size = (ss, ss) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            print(model_path)
            self.yolo_model = load_model(model_path, compile=False)
        except:
            print('Exception...',model_path, is_tiny_version,num_classes,num_anchors)
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)

            
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,max_boxes=self.max_boxes,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            #boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        #else:
            #new_image_size = (image.width - (image.width % 32),
                              #image.height - (image.height % 32))
            #boxed_image = letterbox_image(image, new_image_size)
        image_data = image #np.array(boxed_image, dtype='float32')
        image=Image.fromarray(image, 'RGB') # for numpy image 
        
        #print(image_data.shape)
        image_data = image_data/255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(1e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]            
            label = '{} {:.2f}'.format(predicted_class[0], score)             
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            #draw.rectangle(
                #[tuple(text_origin), tuple(text_origin + label_size)],
                #fill=self.colors[c])
            #draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('Time: ' ,end - start)
        return np.array(image), out_boxes, out_scores

    def close_session(self):
        self.sess.close()
