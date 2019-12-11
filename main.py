import cv2
import os 
import time as tm 
import numpy as np
import random 
import pandas as pd
from scipy import misc
from sklearn.preprocessing import normalize
#from matplotlib import pyplot as plt 
#import functions
from functions import * 
#from yolo import postprocess

master_path_to_dataset = "./TTBB-durham-02-10-17-sub10";
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed


skip_forward_file_pattern = ""; # set to timestamp to skip forward to

#crop_disparity = False; # display full or cropL disparity image
pause_playback = False; # pause until key press after each image

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

left_file_list = sorted(os.listdir(full_path_directory_left));

#SGBM PROCESSOR
max_disparity = 160
#stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 11,P1=64,P2=256);
lineThickness = 2 #how the large the thickness defining the crop is

# SGBM Parameters -----------------
window_size = 2            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=max_disparity,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=11,#works well with 11. Higher makes highly innacurate predictions
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=1,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters used to filter disparity with Weighted Least Squares
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

count = 0
#YOLO ATTRIBUTES

windowName = 'YOLOv3 object detection: ' + "yolov3.weights"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
# trackbarName = 'reporting confidence > (x 0.01)'
# cv2.createTrackbar(trackbarName, windowName , 0, 100, on_trackbar)

#disparity variables
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
disparities = 128   # num of disparities to consider
block = 31          # block size to match
units = 0.001       # depth units

#sbm = cv2.StereoBM_create(numDisparities=disparities,blockSize=block)

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 169     # Width of network's input image used to be 416 EXPERIMENTING WITH LOWER VALUES FOR FASTER DETECTION
inpHeight = 169      # Height of network's input image used to be 416

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
    

    #from left image get the corresponding right one
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    

    #check that its png and that the corresponding right image exists)
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR) #read left image
        imgR = cv2.imread(full_path_filename_right, cv2.COLOR_BGR2GRAY) #read right image in grayscale 

        


        # nphorizontal = np.hstack((imgL,cropL))
        # cv2.namedWindow("Horizontal", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Horizontal", cv2.WND_PROP_FULLSCREEN,
        #                         cv2.WINDOW_FULLSCREEN )
        


        #imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 15) 
        #imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 15) 

        #crop the 2 images to exclude the own car (from 544 reducde to 400). This has no impact on the placement of polygons 
        cropL = imgL[:416,304:720]
        cropR = imgR[:416,304:720]
        
        cropL = cv2.bilateralFilter(cropL,d=0,sigmaColor=30,sigmaSpace=20)
        cropR = cv2.bilateralFilter(cropR,d=0,sigmaColor=30,sigmaSpace=20)

        # no need to show the right image
        #cv2.imshow('right image',imgR)
        
        
        #convert to greyscale for disparity, no need for right image as its already in grayscale
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
        

        #preprocessing for better results (CHANGE THAT AND TEST OTHER PREPROCESSING STEPS SUCH AS DENOISING)
        #convert to grey in oder to calculate disparity between left and right images
        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');
        #use denoising techniques to improve disparity detection
        # cv2.imshow("test",cropR)
        # cv2.waitKey(0)
        # break
       


            #we calculate both the disparities in order to apply WLS filtering. Computed based on creation of left and right matcher at the start
        displ = left_matcher.compute(grayL, grayR).astype(np.float32)
        dispr = right_matcher.compute(grayR, grayL).astype(np.float32)
            #convert to int16, don't really need to, check for improvement in computation
            # displ = np.int16(displ)
            # dispr = np.int16(dispr)
            #filter the image by passing the disparities and the original image
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
            #normalize the result of the filtering
        cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            #convert to unit8 and assign it to disparity scaled to use in the functions below
        disparity_scaled = np.uint8(filteredImg)
        disparity_scaled = disparity_scaled[:416,304:720]
       
        #disparity = left_matcher.compute(grayL,grayR);
        
        
        #Old code to filter out noise and speckles and to convert disparity
        # dispNoiseFilter = 5; # increase for more agressive filtering
        # cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        # disparity_scaled = (disparity / 16.).astype(np.uint8);
        #cropL = imgL[:416,304:720]
        tensor = cv2.dnn.blobFromImage(cropL, 1/255, (cropL.shape[1], cropL.shape[0]), [0,0,0], 1, crop=False)

        
        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence
        #confThreshold = cv2.getTrackbarPos(trackbarName,windowName) / 100
        confThreshold=50/100
        classIDs, confidences, boxes, centers = postprocess(cropL, results, confThreshold, nmsThreshold)
        
# draw resulting detections on image
        distances = []
        #following is the code for the detected objects
        for detected_object in range(0, len(boxes)):
            box = boxes[detected_object]
            center = centers[detected_object]
            left = box[0]+304 #add the 304 because of the cropped section being detected 
            top = box[1]
            width = box[2]
            height = box[3]

            #calculate the coordinates
            coords = [ left, top, left + width, top + height]
            #for each image calculate the distances from detected objects
            distances.append(project_disparity_to_3d(disparity_scaled, max_disparity, coords,center, cropL))
            drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50),distances[-1])
        print(full_path_filename_left);
        try:
            print("{}: nearest detected object on the scene is: {}m \n".format(full_path_filename_right, min(distances)));
            minDis.append(min(distances))

        except:
            print("{}: nearest detected object on the scene is: {} \n".format(full_path_filename_right,"inf"));
            minDis.append(0)

        #draw lines on image to show cropped section used for detection
        cv2.rectangle(imgL, (304,0),(720, 543), (0, 255, 0),lineThickness)
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