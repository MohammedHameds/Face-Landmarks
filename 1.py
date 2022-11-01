import cv2
import dlib


detetcor = dlib.get_frontal_face_detector()
predector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while cap.isOpened() :
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detetcor(gray)
    
    print(faces)
    for face in faces :
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
    
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        
        landmarks = predector(gray,face)
        for i in range (68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame,(x,y),3,(255,0,0),-1)
    
    cv2.imshow('Frame',frame)
    
    if cv2.waitKey(27) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()    