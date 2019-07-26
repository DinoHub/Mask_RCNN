import os
import sys
import random
import math
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

COCO_PERSON_INDEX = 1

coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNN(object):
    def __init__(self, confidence_threshold=0.7):
        config = InferenceConfig()
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.conf_thresh = confidence_threshold

    def get_chips_and_masks(self, img, label_index=COCO_PERSON_INDEX):
        '''
        Params
        ------
        img : nd array like, RGB
        label_index : int, index of label wanted

        Returns
        -------
        list of tuple (chip, mask)
        - chip is a ndarray: bb crop of the image
        - mask is a ndarray: same 2d shape as chip, whose 'pixel' value is either 0 or 1, indicating if that pixel belongs to that class or not. 
        '''

        preds = self.model.detect([img])[0]

        labels = preds['class_ids']
        person_bool_mask = (labels==label_index).astype(bool)

        masks = np.transpose(preds['masks'], (2,0,1))[person_bool_mask]
        ## BB IS IN TLBR
        bboxes = preds['rois'][person_bool_mask]
        scores = preds['scores'][person_bool_mask]

        results = []

        for mask, box, score in zip( masks, bboxes, scores ):
            print(score)
            if score < self.conf_thresh:
                print('skipped')
                continue
            t,l,b,r = box
            if b - t <= 0 or r - l <= 0:
                continue
            content = img[ t:(b+1), l:(r+1), : ]
            minimask = mask[ t:(b+1), l:(r+1) ]
            results.append( (content, np.expand_dims(minimask, axis=-1)) )

        return results                

if __name__ == '__main__':
    import cv2
    # import numpy as np

    maskrcnn = MaskRCNN()
    # img_path = '/home/dh/Pictures/studio8-30Nov18/DSC03887.JPG'
    img_path = '/home/dh/Pictures/crowd5.jpg'
    # img_path = '/home/dh/Pictures/frisbee.jpg'
    img = cv2.imread(img_path)
    # masks, bboxes = 
    chipsandmasks = maskrcnn.get_chips_and_masks(img)
    print(len(chipsandmasks))

    for chip, mask in chipsandmasks:
        masked = chip * mask
        cv2.imshow( '', masked )
        cv2.waitKey(0)

    # input()