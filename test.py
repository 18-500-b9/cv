import cv2
import numpy as np
import imutils

lower_yellow = np.array([18,50,5]) #1
upper_yellow = np.array([25,255,255])
yellow = (255,255,0)
lower_orange = np.array([5,50,5]) #2
upper_orange = np.array([15,255,255])
orange = (255,140,0)
lower_blue = np.array([110,50,5]) #3
upper_blue = np.array([118,255,255])
blue = (0,0,255)
lower_purple = np.array([120,0,0]) #4, hard for darker
upper_purple = np.array([140,255,255])
purple = (138,43,226)
lower_red = np.array([175,50,5]) #5
upper_red = np.array([180,255,255])
red = (255,0,0)
lower_green = np.array([62,50,5]) #6
upper_green = np.array([82,255,255])
green = (0,128,0)
lower_brown = np.array([155,0,0]) #7, hard
upper_brown = np.array([174,255,255])
brown = (165,42,42)
lower_white = np.array([25,0,0])
upper_white = np.array([61,255,255])
white = (255,255,255)

class BallInfo:
    def __init__(self, lower_hsv, upper_hsv, rgb):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.bgr = (rgb[2], rgb[1], rgb[0])

def init_ballinfo():
    balls = []
    white_ball = BallInfo(lower_white, upper_white, white)
    balls.append(white_ball)
    yellow_ball = BallInfo(lower_yellow, upper_yellow, yellow)
    balls.append(yellow_ball)
    orange_ball = BallInfo(lower_orange, upper_orange, orange)
    balls.append(orange_ball)
    blue_ball = BallInfo(lower_blue, upper_blue, blue)
    balls.append(blue_ball)
    purple_ball = BallInfo(lower_purple, upper_purple, purple)
    balls.append(purple_ball)
    red_ball = BallInfo(lower_red, upper_red, red)
    balls.append(red_ball)
    green_ball = BallInfo(lower_green, upper_green, green)
    balls.append(green_ball)
    brown_ball = BallInfo(lower_brown, upper_brown, brown)
    balls.append(brown_ball)
    
    return balls

def add_ball(ball, hsv, frame):
    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, ball.lower_hsv, ball.upper_hsv)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    center = None
 
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        circles = sorted([c for c in cnts if cv2.contourArea(c) >= 200], key=cv2.contourArea)
        # circles = cnts
        for c in circles:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
     
            # only proceed if the radius meets a minimum size
            print(cv2.contourArea(c))
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    ball.bgr, 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

def run_simul():
    filename = "img.jpg"
    # Take each frame
    frame = cv2.imread(filename)

    # Convert BGR to HSV
    # TODO: possible Gaussian blurring
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    balls = init_ballinfo()
    for ball in balls:
        add_ball(ball, hsv, frame)
    while(1):
        
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


    # cv2.destroyAllWindows()

run_simul()