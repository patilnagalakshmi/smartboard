from HandTrackingModule import HandDetector
import cv2
import os
import numpy as np
import mediapipe as mp 
#COLOR
color = (0, 255, 255) 




# feature flags
saveflag = False
circleFlag = False
delay = 0
maxDelay = 65


screenshots_folder = "screenshots"
#screenShots Setup
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)
    
saved_folder = "saved"
#screenShots Setup
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

# Variable to keep track of the image number
image_number = 1

#saved_imageNo
saved_number = 1



# myconfig = r"--psm 10 --oem 3"


#circle
center,rad = None,None

#helper function
def get_min_max_coordinates(coords):
    if not coords:
        return None

    min_x = min(coords, key=lambda x: x[0])[0]
    min_y = min(coords, key=lambda x: x[1])[1]
    max_x = max(coords, key=lambda x: x[0])[0]
    max_y = max(coords, key=lambda x: x[1])[1]

    return min_x, min_y, max_x, max_y


# helper functions
def quit(lmList, lmList2):
    safe = 60
    dist, info = detectorHand.findDistance(lmList[5][0:2], lmList2[9][0:2])
    print("DISTANCE IS : ", dist)
    if dist < safe:
        return True
    return False
# def straightLine():
    # global annotationStart
    # # global annotationNumber
    # global indexFinger
    # if (fingers==[1,1,1,1,1] or fingers2== [1,1,1,1,1]) and (fingers == [0,1,0,0,0] and fingers == [0,1,0,0,0]):
    #     if annotationStart is False:
    #         annotationStart = True
    #         # annotationNumber += 1
    #         annotations.append([])
    #     # indexFinger = (indexFinger[0],600)
    #     if len(annotations[-1])>1:
    #         indexFinger = (annotations[-1][0][0],indexFinger[1])
    #     annotations[-1].append(indexFinger)
    #     cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
        

def drawcircle():
    global indexFinger
    global imgCurrent
    global h1
    global w1
    global allHands,center,rad
    if len(allHands)==2:
        hand1 = allHands[0]
        hand2 = allHands[1]
        cx = (hand1['center'][0]+ hand2['center'][0])//2
        cy = (hand1['center'][1]+ hand2['center'][1])//2
        cx = int(np.interp(cx, [0,500], [0, width]))
        cy = int(np.interp(cy, [100, height - 150], [0, height]))
        length, info, imgCurrent = detectorHand.findDistance(hand1['center'], indexFinger, imgCurrent, color=(255, 0, 0), scale=2)
        print('Cirle Radius', length)
        
        cv2.circle(imgCurrent, (cx,cy), int(length), (255, 0, 0), 4)
        center = (cx,cy)
        rad = length
def remcircle():
    global center,rad
    center,rad=None,None
def drawsquare():
    global heigh,width
    global imgCurrent
    hand1=allHands[0]
    hand2=allHands[1]
    index = lmList[8][0:2]
    thumb = lmList2[1][0:2]
    x1 = int(np.interp(index[0], [0,400], [0, 2*width]))
    y1 = int(np.interp(index[1], [100, height - 250], [0, 2*height]))
    
    x2 = int(np.interp(thumb[0], [0,400], [0, 2*width]))
    y2 = int(np.interp(thumb[1], [100, height - 250], [0, 2*height]))
    imgCurrent=cv2.rectangle(imgCurrent,(x1,y1),(x2,y2),(255,0,0),4)
    

# Parameters

gestureThreshold = 300

folderPath = "Presentation"
screenshotPath = "screenshots"

# Camera Setup
width, height = 1280, 1080
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cv2.setUseOptimized(True)  # Enable OpenCV optimizations

# Hand Detector
detectorHand = HandDetector(detectionCon=0.7, maxHands=2)

# Variables
imgList = []
delay = 10
buttonPressed = False
counter = 0
drawMode = False
imgNumber = 0
delayCounter = 0
annotations = [[]]
# annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Zooming feature
initialDistance = 0
image_scale = 0

# Get list of presentation1 images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    success, img = cap.read()
    # inorder to avoid the mirror effect , flip the image
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    h1, w1, _ = imgCurrent.shape 

    winName = 'slide'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

    # Find the hand and its landmarks
    allHands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    
    if center and rad:
        cv2.circle(imgCurrent, (3*center[0],2*center[1]), int(rad), (255, 0, 0), 4)

    if allHands and buttonPressed is False:  # If hand is detected

        hand = allHands[0]

        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [0,650], [0, width]))
        yVal = int(np.interp(lmList[8][1], [100, height - 150], [0, height]))
        # xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        # yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = xVal, yVal
        if len(allHands) == 2:
            # print(allHands)
            hand2 = allHands[1]
            lmList2 = hand2['lmList']
            fingers2 = detectorHand.fingersUp(hand2)
            if fingers2 == [1, 1, 0, 0, 0] and fingers == [1, 1, 0, 0, 0]:
                if (initialDistance is None):
                    fin_distance, info, img = detectorHand.findDistance(lmList[8][0:2], lmList2[8][0:2], img)
                    initialDistance = fin_distance
                length, info, img = detectorHand.findDistance(lmList[8][0:2], lmList2[8][0:2], img)
                print('initaial distance , length : ', initialDistance, length)
                image_scale = int(1.8 * (length - initialDistance))
                print('IMAGE SCALED BY ', image_scale)
                cx, cy = info[4:]
                #Horizontal Line
            # straightLine()
            if fingers == [0,1,1,0,0] and 10<xVal<w//6 and 2*h//6<yVal<3*h//6:
                delay += 1
                if delay>maxDelay:
                    if circleFlag:
                        circleFlag = False
                    else:
                        circleFlag = True 
                    
            if circleFlag:
                print("Circle Flag is print" )
                drawcircle()
            if(fingers2==[1,1,1,0,0] and fingers==[1,1,1,0,0]):
                print("remove")
                remcircle()
            if((fingers2==[0,1,1,1,1] or fingers==[0,1,1,1,1]) and (fingers==[0,1,0,0,0]) or fingers==[0,1,0,0,0] ):
                print("draw")
                drawsquare()

        else:
            initialDistance = None
        try:
            h1, w1, _ = imgCurrent.shape
            # print('h1,w1', h1,w1)
            # print('image_scale is : ',image_scale)
            new_height, new_width = ((300 + 3 * image_scale) // 2) * 2, ((400 + 3 * image_scale) // 2) * 2
            imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))
            # keeping the image in the center of the width
            windowCenter = (0, 0)
            cv2.moveWindow(winName, windowCenter[0], windowCenter[1])
            cv2.resizeWindow(winName, new_width, new_height)
        except:
            pass

        if cy <= gestureThreshold:  # If hand is at the height of the face
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    # annotationNumber = -1
                    annotationStart = False
                else:
                    cv2.putText(imgCurrent, 'FIRST SLIDE', (20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=3, color=(256, 0, 0), lineType=cv2.LINE_AA, thickness=2)
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    # annotationNumber = -1
                    annotationStart = False
                else:
                    cv2.putText(imgCurrent, 'LAST SLIDE', (20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=3, color=(256, 0, 0), lineType=cv2.LINE_AA, thickness=2)
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        if ((fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]) and len(allHands) == 1):
            if not annotationStart:
                annotationStart = True
                annotations = [[]]
            if annotations:
                annotations[-1].append(indexFinger)
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        else:
            annotationStart = False

        if fingers == [0, 1, 1, 1, 0]:
            if annotations and annotations[-1]:
                buttonPressed = True
        if fingers == [0,1,1,0,0] and 10<xVal<w//6 and h//6<yVal<2*h//6:
            saved_path = os.path.join(saved_folder, f"saved_{saved_number}.png")
            cv2.imwrite(saved_path, imgCurrent)
            saved_number+=1
        if fingers == [0,1,1,0,0] and 10<xVal<w//6 and 5*(h//6)<yVal<h:
            delay+=1
            if delay>maxDelay:
                delay = 0
                color = (0,255,0)
            
        # if fingers == [0, 1, 1, 1, 0]:
        #     if annotations:
        #         annotations.pop(-1)
        #         # annotationNumber -= 1
        #         buttonPressed = True
    else:
        annotationStart = False
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False
    
        
    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                # print(annotation)
                cv2.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)
        if annotation:
            xmin,ymin,xmax,ymax = get_min_max_coordinates(annotation)
            margin = 40
            xmin -=margin
            ymin -= margin
            xmax += margin
            ymax += margin
            cv2.rectangle(imgCurrent, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(imgCurrent.shape[1], xmax)
            ymax = min(imgCurrent.shape[0], ymax)
            delay+=2
            # Check if the region specified by the bounding box is valid
            if xmin < xmax and ymin < ymax and annotationStart and delay>maxDelay:
                delay = 0
                # Capture screenshot of the region defined by the bounding box
                screenshot = imgCurrent[ymin:ymax, xmin:xmax].copy()
                
                # Check if the screenshot contains valid image data
                if not screenshot.size == 0:
                    # Define the path to save the screenshot
                    screenshot_path = os.path.join(screenshots_folder, f"image_{image_number}.png")
                    
                    # Save the screenshot
                    cv2.imwrite(screenshot_path, screenshot)
                    
                    # Increment the image number for the next screenshot
                    image_number += 1
        
    h, w, _ = imgCurrent.shape
    imgSmall = cv2.resize(img, (w // 6, h // 6))
    imgCurrent[0:int(h // 6), 0:int(w // 6)] = np.ones((h // 6, w // 6, 3))*96
    # text 
    text = 'Clear'
    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (10,h//12) 
    
    # fontScale 
    fontScale = 2
    
    # Red color in BGR 
    
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.putText() method 
    imGCurrent = cv2.putText(imgCurrent, text, org, font, fontScale,  
                    color, thickness, cv2.LINE_AA, False) 
    imgCurrent[int(h // 6):2 * int(h // 6), 0:int(w // 6)] = np.ones((h // 6, w // 6, 3)) * 151
    imgCurrent = cv2.circle(imgCurrent, (w//12,h//4), 100, (255, 0, 255), 5)

    # imgCurrent = cv2.putText(imgCurrent, 'Save', (10,3*h//12), font, fontScale,  
    #                 color, thickness, cv2.LINE_AA, False)

    imgCurrent[2 * int(h // 6):3 * int(h // 6), 0:int(w // 6)] = np.ones((h // 6, w // 6, 3)) * 96
    imgCurrent = cv2.rectangle(imgCurrent,(60,2*(h//6)+25),(w//6-60,3*(h//6)-25),(255,0,255),5)
    # imgCurrent = cv2.putText(imgCurrent, "Square", (10,5*h//12), font, fontScale,  
    #                 color, thickness, cv2.LINE_AA, False)

    imgCurrent[3 * int(h // 6):4 * int(h // 6), 0:int(w // 6)] = np.ones((h // 6, w // 6, 3)) * 151
    cv2.putText(imgCurrent, 'Calculator', (10,9*h//12), font, fontScale,  
                    color, thickness, cv2.LINE_AA, False)
    blue_segment = np.zeros((h//6, w//6, 3), dtype=np.uint8)
    blue_segment[:, :, 0] = 255 
    imgCurrent[4 * int(h // 6):5 * int(h // 6), 0:int(w // 6)] = blue_segment
    cv2.putText(imgCurrent, 'Saved', (10,7*h//12), font, fontScale,  
                    color, thickness, cv2.LINE_AA, False)
    green_segment = np.zeros((h//6, w//6, 3), dtype=np.uint8)
    green_segment[:, :, 1] = 255 

    imgCurrent[5 * int(h // 6):6 * int(h // 6), 0:int(w // 6)] = green_segment
    
    try:
        imgCurrent[0:h // 6, w - w // 6: w] = imgSmall
    except:
        pass
    # imgCurrent = cv2.resize(imgCurrent,(new_width,new_height,))
    cv2.imshow(winName, imgCurrent)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

