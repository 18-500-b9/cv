import cv2
import numpy as np

def computeLines(img, displayHoughLines):
  
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, threshold_img = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
  cv2.imshow('threshold',threshold_img)
  edges = cv2.Canny(threshold_img,50,150,apertureSize = 3)
  cv2.imshow('edges',edges)

  lines = cv2.HoughLines(edges,1,np.pi/180,100)
  minX = 2000
  maxX = -2000
  minY = 2000
  maxY = -2000
  for i in range(5):
    line = lines[i]
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if x1 < -900 or x2 < -900:
          minY = min(minY, min(y1, y2))
        if x1 > 900 or x2 > 900:
          maxY = max(maxY, max(y1, y2))
        if y1 < -900 or y2 < -900:
          minX = min(minX, min(x1, x2))
        if y1 > 900 or y2 > 900:
          maxX = max(maxX, max(x1, x2))
        print("(" + str(x1) + ", " + str(y1) + ")  " + "(" + str(x2) + ", " + str(y2) + ")")
        if displayHoughLines:
          cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

  if displayHoughLines:
    cv2.imshow('houghlines.jpg',img)
    # Can average positioning between the 4 intersection points of the 4 lines
    while(1):
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
          break
  print("Mins: (" + str(minX) + "," + str(minY) + ") Maxs: (" + str(maxX) + "," + str(maxY) + ")")
  return minX, minY, maxX, maxY

# def run():
#   img = cv2.imread('pool2.jpg')
#   displayHoughLines = True
#   x1, y1, x2, y2 = computeLines(img, displayHoughLines)
#   print("Returns (" + str(x1) + ", " + str(y1) + ")  " + "(" + str(x2) + ", " + str(y2) + ")")

# run()
