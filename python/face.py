import cv2
#First we need to load the required XML classifiers. Then load our input image (or video) in grayscale mode
face_cascade=cv2.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("../data/haarcascades/haarcascade_eye.xml")

ds_factor = 0.5
cap=cv2.imread('../data/cruise3.jpg')
#cv2.imshow("Ori cap",cap)
if cap is None:
    raise IOError("Cannot open the webcam!")
#open camera
while True:  
    frame = cap
    #get frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #灰度轉換
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    #畫框
    #Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). Once we get these locations, we can create a ROI (感興趣的區域)for the face and apply eye detection on this ROI (since eyes are always on the face !!! ).
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)  
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)    
    cv2.imshow("face",frame)
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()