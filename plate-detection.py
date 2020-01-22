import cv2
import os 
import numpy as np


kernel = np.ones((3,3),np.uint8)
kernel7 = np.ones((7,7),np.uint8)

def recognize(gray):
    width = 800
    height = 200
    dim = (width,height)
    img_resized = cv2.resize(gray,dim)
    thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    y = 0
    h = thresh.shape[0]
    x = 410
    w = thresh.shape[1]
    img_cropped = thresh[y:y+h, x:x+w]
    characters = []
    y = 0 
    h = img_cropped.shape[0]
    x = 0
    w = img_cropped.shape[1]
    z = 300
    cv2.imshow('all', img_cropped)
    for i in range(6):
        im = img_cropped[y:y+h, x:w-z]
        characters.append(im)
        cv2.imshow('img_cropped', im)
        x = x+60
        z = z-60

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def threshold2zero(image, threshold, value=0):
    image = image.astype(np.float64)
    image[image<threshold] = value
    image = image.astype(np.uint8)
    return image

def threshold2one(image, threshold, value=255):
    image = image.astype(np.float64)
    image[image>threshold] = value
    image = image.astype(np.uint8)
    return image
      
def char_rec(frames):
    for frame in frames:
        gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        recognize(gframe)
            
                         
def detect_plate(image):
    if max(image.shape) > 720:
        image = cv2.resize(image, (720,480))
        
    #convert image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #blackout all pixels under 80
    threshold = threshold2zero(gray,80)
    
    #remove noise
    denoised = cv2.bilateralFilter(gray,9,130,255)
    
    #blur the image
    blur = cv2.blur(denoised,(3,3)) 
    
    #morphological transformations
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow('ope', opening)
    rectshapes = cv2.morphologyEx(opening, cv2.MORPH_RECT, kernel,iterations=1)
    whiteing = threshold2zero(opening,100)
    # cv2.imshow('whiteing', whiteing)

    

    closing = cv2.morphologyEx(whiteing, cv2.MORPH_CLOSE, kernel,iterations=1)
    
    #canny edge detector
    edge = cv2.Canny(closing,70,150)
    
    #extract contours
    cnts, new = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    #sort contours due to contour area
    cnts = sorted(cnts, key= cv2.contourArea, reverse=True)[:30]
    image1 = image.copy()
    cc = []
    potentials = []
    for c in cnts:
        #check if contour is closed 
        if cv2.contourArea(c) >= cv2.arcLength(c,True):
            cc.append(c)
            
            #create bounding box
            x,y,w,h = cv2.boundingRect(c)
            
            #check if a bounding box is rectangle 
            if w/h > 1 :
                
                #draw the bounding box 
                cv2.rectangle(image1,(x,y),(x+w, y+h), (0,0,255),2)
                new_img1=image[y:y+h,x:x+w]
                
                if h*w > 500 and max(h,w)/min(h,w) < 10:
                    potentials.append(new_img1)
    # cv2.imshow('all', image1)
    return image1, potentials


def detect_video(video_path, frame_rate):
    i = 0
    cap = cv2.VideoCapture(video_path)
    
    while(cap.isOpened()):
            if i % frame_rate:
                i+=1
                continue
            ret, frame = cap.read()
            frameo = detect_plate(frame)
            sub_frames = frameo[1]
            


            try:
                cv2.imshow('frame',frameo[0])
            except:
                pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            i+=1

    cap.release()
    cv2.destroyAllWindows()
    

def detect_pictures(directory):
    pics = [directory+item for item in os.listdir(directory)]
    for pic in pics:
        try:
            img = cv2.imread(pic)
            plate = detect_plate(img)
            sub_frames = plate[1]
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e :
            print(e)


if __name__ == "__main__":
    detect_video('./video/1.mov', 25)
    # detect_pictures('./plates imgs/')