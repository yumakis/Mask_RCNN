# example of inference with a pre-trained coco model
import os
import sys
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances, write_logs
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#draw an image with detected objects
# def draw_image_with_boxes(filename, boxes_list):
#     plt.clf()
#     # load the image
#     data = plt.imread(DATADIR + filename)
#     # plot the image
#     plt.imshow(data)
#     # get the context for drawing boxes
#     ax = plt.gca()
#     # plot each box
#     for box in boxes_list:
#          # get coordinates
#          y1, x1, y2, x2 = box
#          # calculate width and height of the box
#          width, height = x2 - x1, y2 - y1
#          # create the shape
#          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#          # draw the box
#          ax.add_patch(rect)
#     # show the plot
#     # plt.show()
#     plt.axis('off')
#     plt.savefig(RESULTDIR + filename, bbox_inches='tight')
#     plt.close()

#Names of classes of our model

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']


class_names = ['BG', 'ball', 'cube', 'cup', 'bottle', 'bowl', 'cylinder']

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 6

# Get the folder the Video Frames we want to analyze
if len(sys.argv) < 2:
    print("Folder Frames required")
    sys.exit()
else:
    data_frames = sys.argv[1]

video_name = data_frames.split('/')[-2].split('_Frames')[0]

DATADIR = data_frames
RESULTDIR = "./test_results/" + video_name + "_results/"

#Create the result folder if needed
try:
    if not os.path.exists(RESULTDIR):
        os.makedirs(RESULTDIR)
except OSERROR:
    print('Error: Creating directory of data')

#load the model
rcnn = MaskRCNN(mode='inference', model_dir = './', config=TestConfig())
rcnn.load_weights('./mask_rcnn_mica.h5', by_name=True)

#Analyze the Frames and write logs with boxes informations
for img_name in os.listdir(DATADIR):
    data = DATADIR + img_name
    img = load_img(data)
    img = img_to_array(img)
    results = rcnn.detect([img], verbose=0)
    r = results[0]
    # draw_image_with_boxes(data, results[0]['rois'])
    display_instances(img_name, RESULTDIR, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    # write_logs(img_name, RESULTDIR, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


# # define the model
# rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# # load coco model weights
# rcnn.load_weights('./mask_rcnn_coco.h5', by_name=True)
# # load photograph
# img = load_img('./Images_samples/Laptop.png')
# img = img_to_array(img)
# # make prediction
# results = rcnn.detect([img], verbose=0)
# r = results[0]
# # visualize the results
# display_instances("Laptop.png" , "./", img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
