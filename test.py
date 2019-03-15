import cv2
import numpy as np
import imutils
from datetime import datetime

# HSV values for different color balls
# Range is 180,255,255

lower_yellow = np.array([17,200,5]) #1
upper_yellow = np.array([25,255,255])
yellow = (255,255,0)
lower_orange = np.array([6,200,5]) #2
upper_orange = np.array([15,255,255])
orange = (255,140,0)
lower_blue = np.array([106,150,5]) #3
upper_blue = np.array([118,255,255])
blue = (0,0,255)
lower_purple = np.array([104,50,0]) #4, hard for darker
upper_purple = np.array([162,200,128])
purple = (138,43,226)
lower_red = np.array([0,100,200]) #5
upper_red = np.array([4,255,255])
red = (255,0,0)
lower_green = np.array([87,235,76]) #6
upper_green = np.array([94,255,128])
green = (0,128,0)
lower_brown = np.array([175,50,0]) #7, hard
upper_brown = np.array([180,255,200])
brown = (165,42,42)
lower_white = np.array([18,0,0])
upper_white = np.array([36,52,255])
white = (255,255,255)
black = (0,0,0)
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,35])

# Class to store information on each ball
class BallInfo:
    def __init__(self, lower_hsv, upper_hsv, rgb, str_rep):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.bgr = (rgb[2], rgb[1], rgb[0])
        self.str_rep = str_rep

def init_ballinfo():
    balls = []
    white_ball = BallInfo(lower_white, upper_white, white, 'white')
    balls.append(white_ball)
    yellow_ball = BallInfo(lower_yellow, upper_yellow, yellow, 'yellow')
    balls.append(yellow_ball)
    orange_ball = BallInfo(lower_orange, upper_orange, orange, 'orange')
    balls.append(orange_ball)
    blue_ball = BallInfo(lower_blue, upper_blue, blue, 'blue')
    balls.append(blue_ball)
    purple_ball = BallInfo(lower_purple, upper_purple, purple, 'purple')
    balls.append(purple_ball)
    red_ball = BallInfo(lower_red, upper_red, red, 'red')
    balls.append(red_ball)
    green_ball = BallInfo(lower_green, upper_green, green, 'green')
    balls.append(green_ball)
    brown_ball = BallInfo(lower_brown, upper_brown, brown, 'brown')
    balls.append(brown_ball)
    black_ball = BallInfo(lower_black, upper_black, black, 'black')
    balls.append(black_ball)
    return balls

def add_ball(ball, hsv, frame):
    # Threshold the HSV image to get only ball colors
    mask = cv2.inRange(hsv, ball.lower_hsv, ball.upper_hsv)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
 
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid

        # circles = [c for c in cnts if cv2.contourArea(c) >= 200]
        circles = cnts
        for c in circles:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
     
            # only proceed if the radius meets a minimum size
            if radius > 13 and radius < 17:
                print("rad: " + str(radius))
                #print(ball.str_rep + " rad: " + str(radius) + " contourArea: " +
                #str(cv2.contourArea(c)))
                # M = cv2.moments(c)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), ball.bgr, 2)
                # cv2.circle(frame, center, 2, (0, 0, 255), -1)
                # Seems like using x,y from contour area is better
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)


def add_cuestick(hsv, frame):
    cue_lower = np.array([14,10,200])
    cue_upper = np.array([18,128,255])


    mask = cv2.inRange(hsv, cue_lower, cue_upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:
        contourArea = cv2.contourArea(cnt)
        if (1000 < contourArea):
            rows,cols = hsv.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            cv2.line(frame,(cols-1,righty),(0,lefty),(0,255,0),2)

    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

def run_simul():
    balls = init_ballinfo()
    # cap = cv2.VideoCapture(1)
    running = True
    while running:
       # ret, frame = cap.read()

         filename = "pool2.jpg"
         # Take each frame
         frame = cv2.imread(filename)

        # Convert BGR to HSV
         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
         balls = init_ballinfo()
         # for ball in balls:
         #     add_ball(ball, hsv, frame)
         add_cuestick(hsv, frame)
         cv2.imshow('frame', frame)
        # input

         # if cv2.waitKey(1) & 0xFF == ord('q'):
         #     break

         while(1):
             k = cv2.waitKey(5) & 0xFF
             if k == 27:
                running = False
                break

    # When everything done, release the capture
    # cap.release()
    cv2.destroyAllWindows()

run_simul()
