import cv2
import os 
import time as tm 
import numpy as np
import random 
import pandas as pd
#from matplotlib import pyplot as plt 
#import functions
from functions import * 
#from yolo import postprocess

master_path_to_dataset = "./TTBB-durham-02-10-17-sub10";
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed


skip_forward_file_pattern = ""; # set to timestamp to skip forward to

#crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);


left_file_list = sorted(os.listdir(full_path_directory_left));

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

count = 0
#YOLO ATTRIBUTES

windowName = 'YOLOv3 object detection: ' + "yolov3.weights"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
trackbarName = 'reporting confidence > (x 0.01)'
cv2.createTrackbar(trackbarName, windowName , 0, 100, on_trackbar)

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 150      # Width of network's input image used to be 416 EXPERIMENTING WITH LOWER VALUES FOR FASTER DETECTION
inpHeight = 150      # Height of network's input image used to be 416

# Load names of classes from file

classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
#END OF YOLO ATTRIBUTES
time = []
minDis = []
avgDis = []
objects = []

for filename_left in left_file_list:

    #can introduce timestamp here from stereo disparity
    # if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
    #     continue;
    # elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
    #         skip_forward_file_pattern = "";

    #from left image get the corresponding right one
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    

    #check that its png and that the corresponding right image exists)
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        #cv2.imshow('left image',imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 15) 
        imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 15) 
        # no need to show the right image
        #cv2.imshow('right image',imgR)

        #convert to greyscale for disparity
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
        

        #preprocessing for better results (CHANGE THAT AND TEST OTHER PREPROCESSING STEPS SUCH AS DENOISING)
        #convert to grey in oder to calculate disparity between left and right images
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');
        #use denoising techniques to improve disparity detection
       

        disparity = stereoProcessor.compute(grayL,grayR);

        #filter out noise and speckles
        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);
        tensor = cv2.dnn.blobFromImage(imgL, 1/255, (imgL.shape[1], imgL.shape[0]), [0,0,0], 1, crop=False)

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);
        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        confThreshold = cv2.getTrackbarPos(trackbarName,windowName) / 100
        classIDs, confidences, boxes, centers = postprocess(imgL, results, confThreshold, nmsThreshold)
        
# draw resulting detections on image
        distances = []
        #following is the code for the detected objects
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            center = centers[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            #calculate the coordinates
            coords = [ left, top, left + width, top + height]
            #for each image calculate the distances from detected objects
            distances.append(project_disparity_to_3d(disparity_scaled, max_disparity, coords,center, imgL))
            drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50),distances[-1])
        print(full_path_filename_left);
        try:
            print("{}: nearest detected object on the scene is: {}m \n".format(full_path_filename_right, min(distances)));
            minDis.append(min(distances))

        except:
            print("{}: nearest detected object on the scene is: {} \n".format(full_path_filename_right,"inf"));
            minDis.append(0)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        time.append(t * 1000.0 / cv2.getTickFrequency())
        #minDis.append(min(distances))
        avgDis.append(sum(filter(lambda x: isinstance(x,float),distances)))
        objects.append(len(boxes))
        # display image
        cv2.imshow(windowName,imgL)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN )
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        # display image
        cv2.imshow(windowName,imgL)
        cv2.imshow("Disparity",disparity_scaled)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN )

        #function for wait save and exit keys, important for images to show

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')) or count==30:
            data = {
                "time":time,
                "minDis":minDis,
                "avgDis":avgDis,
                "objects":objects
            }
            df = pd.DataFrame(data)
            df.to_csv("150Smooth1.csv")
            break; # exit
        elif (key == ord('s')):     # save
            #cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            #cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();
    count +=1