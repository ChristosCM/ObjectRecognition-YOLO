import numpy as np
import cv2
import math
# dummy on trackbar callback function
# def on_trackbar(val):
#     return


# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

def drawPred(image, class_name, confidence, left, top, right, bottom, colour,dis):
    if class_name=="person":
        colour = (0,0,255)
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2f' % (class_name, confidence)
    label += " {0:.2f}m".format(dis)
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),(left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    #
    centers = []
    own_car = (frameWidth//2,410)
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                #limit to cars and people classes and check if its detecting the camera's car. If it is then don't add it to the lists.
                if (classId==0 or classId==2) and not (left<own_car[0]<left+width and 400<own_car[1]<544):
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    centers.append((center_x,center_y))
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []
    centers_nms = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])
        centers_nms.append(centers[i])
    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms,centers_nms)


camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 208#262.0;
image_centre_w = 208#474.5;

#following functinons are for calculating the distance
def distance(coords):
    return math.sqrt(coords[0]**2+coords[1]**2+coords[2]**2)

def project_disparity_to_3d(disparity, max_disparity, coords,center, rgb=[]):

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);
    distances = []
    #MAYBE WE CAN ONLY DEFINE THE POINTS OF THE DETECTED OBJECTS FROM YOLO AND CUT DOWN ON COMPUTATION COST
    points=  []

    got = False
    # for y in range(coords[1],min(544,coords[3])): # 0 - height is the y axis index
    #     for x in range(coords[0],min(1024,coords[2])): # 0 - width is the x axis index

    #             # if we have a valid non-zero disparity
            
    #         if (disparity[y,x] > 0):
    #             got = True
    #                 # calculate corresponding 3D point [X, Y, Z]

    #                 # stereo lecture - slide 22 + 25
    #             Z = (f * B) / disparity[y,x];

    #             X = ((x - image_centre_w) * Z) / f;
    #             Y = ((y - image_centre_h) * Z) / f;

    #                 # add to points

    #             if(rgb.size > 0):
    #                 points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
    #             else:
    #                 points.append([X,Y,Z]);    
    
    #Only for getting the center
    Z = (f * B) / disparity[center[1],center[0]];

    X = ((center[0] - image_centre_w) * Z) / f;
    Y = ((center[1] - image_centre_h) * Z) / f;

                    # add to points

    if(rgb.size > 0)and got==True:
        #points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
        #only for calculating with the center as a single point of reference 
        points.append([X,Y,Z,rgb[center[1],center[0],2], rgb[center[1],center[0],1],rgb[center[1],center[0],0]]);
    else:
        points.append([X,Y,Z]);    
    if points==[]:
        return 0
    else:   
        return distance(min(points));
