import cv2
import numpy as np
import imutils
from datetime import datetime
from houghlines import computeLines

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

cue_lower = np.array([14,10,200])
cue_upper = np.array([18,128,255])

TABLE_LENGTH = 37.5
TABLE_WIDTH = 17.5625
ESC_KEY = 27
DISPLAY = True
MAX_CONTOUR_AREA = 1000
USING_CAMERA = False
MIN_RADIUS = .029
MAX_RADIUS = .05
RESIZE_FRAME_WIDTH = 800

# Class to store information on each ball
class BallInfo:
    def __init__(self, lower_hsv, upper_hsv, rgb, str_rep):
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv
        self.bgr = (rgb[2], rgb[1], rgb[0])
        self.str_rep = str_rep

class Ball:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def __repr__(self):
        return "%s: (%f, %f)" % (self.color, self.x, self.y)

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

def find_ball(ball, hsv, frame, table_coords, cv_balls):
    minX,minY,maxX,maxY = table_coords
    table_pixel_length = maxX - minX
    table_pixel_width = maxY - minY

    # Threshold the HSV image to get only ball colors
    mask = cv2.inRange(hsv, ball.lower_hsv, ball.upper_hsv)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    if DISPLAY:
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    circles = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    circles = imutils.grab_contours(circles)
    center = None
 
    # only proceed if at least one contour was found
    if len(circles) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid

        # circles = [c for c in cnts if cv2.contourArea(c) >= 200]
        for c in circles:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # only proceed if the radius meets a minimum size
            if (radius > MIN_RADIUS * table_pixel_width and 
                radius < MAX_RADIUS * table_pixel_width):
                norm_x = (x - minX) / (maxX - minX)
                norm_y = (y - minY) / (maxY - minY)
                table_x = norm_x * TABLE_LENGTH
                table_y = norm_y * TABLE_WIDTH
                print("%s (%.3f,%.3f)" % (ball.str_rep, table_x, table_y))
                cv_balls.append(Ball(norm_x, norm_y, ball.str_rep))
                if DISPLAY:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius), ball.bgr, 2)
                    # Seems like using x,y from contour area is better
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

def find_balls(balls, hsv_img, frame, table_coords):
    cv_balls = []
    for ball in balls:
        find_ball(ball, hsv_img, frame, table_coords, cv_balls)
    return cv_balls

def find_cuestick(hsv, frame):

    mask = cv2.inRange(hsv, cue_lower, cue_upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)

    if DISPLAY:
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for cnt in cnts:
        contourArea = cv2.contourArea(cnt)
        if (MAX_CONTOUR_AREA < contourArea):
            rows,cols = hsv.shape[:2]
            print("rows: " + str(rows) + "cols: " + str(cols))
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            if DISPLAY:
                cv2.line(frame,(cols-1,righty),(0,lefty),white,2)
                cv2.circle(frame, (int(x), int(y)), 2, red, 2)
                cv2.circle(frame, (int(cols-1),int(righty)), 2, red, 2)

            return [(int(x),int(y)), (cols-1, righty)]

def main():
    balls = init_ballinfo()

    # table_coords = 28,49,758,400 # TODO: from houghlines.py

    if USING_CAMERA:
        cap = cv2.VideoCapture(1)
    running = True
    while running:
        frame = None
        # Take each frame
        if USING_CAMERA:
            ret, frame = cap.read()
        else:
            filename = "pool.jpg"
            frame = cv2.imread(filename)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        resize_frame_height = int(frame_height / frame_width * RESIZE_FRAME_WIDTH)
        frame = cv2.resize(frame, (RESIZE_FRAME_WIDTH, resize_frame_height))

        table_coords = computeLines(frame, False)
        # Convert BGR to HSV
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv_balls = find_balls(balls, hsv_img, frame, table_coords)

        print(cv_balls)
        print(find_cuestick(hsv_img, frame))

        if DISPLAY:
            cv2.imshow('frame', frame)

        while(1):
            k = cv2.waitKey(5) & 0xFF
            if k == ESC_KEY:
                running = False
                break

    # When everything done, release the capture
    if USING_CAMERA:
        cap.release()
    cv2.destroyAllWindows()

main()
