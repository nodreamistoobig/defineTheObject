import cv2
import numpy as np

flimit = 250
slimit = 255

star_template = cv2.imread("star_template.png")
owl_template = cv2.imread("owl_template.png")

def fupdate(value):
    global flimit
    flimit = value
    
def supdate(value):
    global slimit
    slimit = value
    
def pointsOrder(box_x, box_y):
    sorted_x = sorted(box_x)
    sorted_y = sorted(box_y)
    
    left_up = (0,0); left_down = (0,0); right_up = (0,0); right_down = (0,0);
            
    for i in range(4):
        if box_x[i] == sorted_x[0] or box_x[i] == sorted_x[1]:
            if box_y[i] == sorted_y[0] or box_y[i] == sorted_y[1]:
                left_up = (box_x[i], box_y[i])
            else:
                left_down = (box_x[i], box_y[i])
        else:
            if box_y[i] == sorted_y[0] or box_y[i] == sorted_y[1]:
                right_up = (box_x[i], box_y[i])
            else:
                right_down = (box_x[i], box_y[i])
                
    return left_up, left_down, right_up, right_down

cam = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

cv2.namedWindow('Camera', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar("F", "Mask", flimit, 255, fupdate)
cv2.createTrackbar("S", "Mask", slimit, 255, supdate)
cv2.createTrackbar("S", "Paper", slimit, 255, supdate)

kernel = np.ones((7,7))

mask = 0

while cam.isOpened():
    ret, frame = cam.read()
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(converted, np.array([0, flimit, 0]), np.array([180, slimit, 170]))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (11,11), 0)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours)>0:
        paper = max(contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(paper)
        box = cv2.boxPoints(rect)
        box_x = []
        box_y = []
        
        for p in box:
            cv2.circle(frame, tuple(p), 6, (0, 255, 0), 2)
            box_x.append(int(p[0]))
            box_y.append(int(p[1]))
        if box_x: 
            paper_image = frame[min(box_y):max(box_y), min(box_x):max(box_x)]

            if paper_image.size > 0:
                left_up, left_down, right_up, right_down = pointsOrder(box_x, box_y)
                points1 = np.float32([left_up, left_down, right_up, right_down])
                points2 = np.float32([[0,0],[0,210],[297,0],[297,210]])
                M = cv2.getPerspectiveTransform(points1, points2)
                
                aff_img = cv2.warpPerspective(frame, M, (297,210))   
                
                star_match = cv2.matchTemplate(aff_img, star_template, cv2.TM_CCOEFF_NORMED)
                _, star_max_val, _, _ = cv2.minMaxLoc(star_match)
                owl_match = cv2.matchTemplate(aff_img, owl_template, cv2.TM_CCOEFF_NORMED)
                _, owl_max_val, _, _ = cv2.minMaxLoc(owl_match)
                
                if (star_max_val > 0.5):
                     cv2.putText(frame, "STAR", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,167))
                else:
                    if (owl_max_val > 0.5):
                        cv2.putText(frame, "OWL", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,167))
                    
                cv2.imshow("Paper", aff_img) 
                         
        cv2.drawContours(frame, [paper], -1, (0,255,0), 3)
        out.write(frame)
    
    
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.imwrite("screen.png", aff_img)

cam.release()
out.release()
cv2.destroyAllWindows()

#match-template
