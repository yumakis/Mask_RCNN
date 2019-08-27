import os
import sys
import cv2
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


mica_class_names = ['BG', 'ball', 'cube', 'cup', 'bottle', 'bowl', 'cylinder']

# define the test configuration
class TestMicaConfig(Config):
     NAME = "MicaTest"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 5

class TestCocoConfig(Config):
     NAME = "CocoTest"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80


def Get_Frames(video_path):

    video_name = video_path.split('/')[-1].split('.')[0]
    print(video_name)

    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = int(cap.get(5))                     #return the number of frame per second of the video
    time_interval = 5
    time_step = time_interval * fps           #We just keep 1 fram every time_interval

    FRAMEDIR = "./Frames/"+video_name+"_Frames/"


    """Create directory for the extraction and extracts"""

    try:
        if not os.path.exists(FRAMEDIR):
            os.makedirs(FRAMEDIR)
    except OSERROR:
        print('Error: Creating directory of data')

    while True:
        ret, frame = cap.read()
        count += 1
        if ret == True:
            if count % time_step == 0:                        #We just want to keep 1 frame every 5 seconds
                name = FRAMEDIR + video_name + "_frame_" + str(int(count/time_step)).zfill(4) + '.jpg'
                print(name)
                cv2.imwrite(name,frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("End of Video Frames extraction")
    return FRAMEDIR



def TestModel(model, image_path=None, video_path=None, mask=None):

    if video_path:
        data_frames = Get_Frames(video_path)
    else:
        if image_path == None:
            print("Folder Frames required")
            sys.exit()
        else:
            data_frames = image_path

    video_name = data_frames.split('/')[-2].split('_Frames')[0]

    DATADIR = data_frames
    RESULTDIR = "./test_results/" + video_name + "_results/"
    LOGDIR = RESULTDIR + "logs/"

    #Create the result folder if needed
    try:
        if not os.path.exists(RESULTDIR):
            os.makedirs(RESULTDIR)
    except OSERROR:
        print('Error: Creating directory of data')

    try:
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
    except OSERROR:
        print('Error: Creating directory of data')

    #Analyze the Frames and write logs with boxes informations
    for img_name in os.listdir(DATADIR):
        data = DATADIR + img_name
        img = load_img(data)
        img = img_to_array(img)
        results = rcnn.detect([img], verbose=0)
        r = results[0]
        # draw_image_with_boxes(data, results[0]['rois'])
        if mask:
            display_instances(img_name, RESULTDIR, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        write_logs(img_name, LOGDIR, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


if __name__ == '__main__':
    import argparse

    #Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the given model on given datas.')
    parser.add_argument("--mask", required=False,
                        metavar="mask",
                        help="Mask or No Mask")
    parser.add_argument('--data', required=False,
                        metavar="/path/to/data",
                        help="Directory to Data to analyze")
    parser.add_argument('--video', required=False,
                        metavar="/path/to/video",
                        help="Directory to Video to Analyze")
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="'coco' or 'mica'")
    args = parser.parse_args()

    #Validate ArgumentParser
    if args.data == None and args.video == None:
        print("Data Folder or Video Folder required")

    print("Weights:" , args.weights)

    #Configurations

    #Create model
    if args.weights == "coco":
        weights_path = "./mask_rcnn_coco.h5"
        class_names = coco_class_names
        config = TestCocoConfig()
    elif args.weights == "mica":
        weights_path = "./mask_rcnn_mica.h5"
        class_names = mica_class_names
        config = TestMicaConfig()
    else:
        weights_path = args.weights
        class_names = mica_class_names
        config = TestMicaConfig()

    rcnn = MaskRCNN(mode="inference", model_dir="./", config=config)

    #Load wieghts
    print("Loading weights ", weights_path)
    rcnn.load_weights(weights_path, by_name=True)

    TestModel(rcnn, mask=args.mask, image_path=args.data, video_path=args.video)
