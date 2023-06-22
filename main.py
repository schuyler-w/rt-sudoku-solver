from __future__ import print_function
import solver
import tensorflow as tf
import keras
import numpy as np
import cv2
import imutils

def main():
    model = keras.models.load_model("modelData.h5")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
        
    while ret:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=1000)
        (height, width) = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        adTh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 5)
        lines = cv2.HoughLinesP(adTh, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        image_lines = adTh.copy()
        
        try:
            for x1, y1, x2, y2 in lines[:,0,:]:
                cv2.line(image_lines, (x1, y1), (x2, y2), (255,255,255), 2)
        except:
            pass
 
        contours = sorted(
            imutils.grab_contours(
                cv2.findContours(
                    image_lines.copy(), 
                    cv2.RETR_LIST, 
                    cv2.CHAIN_APPROX_SIMPLE)
                ), 
            key=cv2.contourArea, 
            reverse=True
            )[:5]
        
        frame_copy = frame.copy()
        final = frame
        grid_digits = ['0']*81
        windows = []
        
        for (i, c) in enumerate(contours):
            peri = cv2.arcLength(c, True)
            guess = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(guess) == 4: ## a square has 4 sides
                (x, y, w, h) = cv2.boundingRect(guess)
                aspect_ratio = w / float(h) ## finding a square, w:h should ~ 1:1
                
                if not (0.8 <= aspect_ratio <= 1.2):
                    continue
                
                ## if aspect ratio is "correct"
                main_contour = guess
                full_coordinates = main_contour.reshape(4, 2)
                
                cv2.drawContours(final, [main_contour], -1, (0,0,0), -1)
                
                ## 4 point transformation to get top-down image of main contour
                sudoku = imutils.perspective.four_point_transform(image_lines, full_coordinates)
                sudoku_clear = imutils.perspective.four_point_transform(frame, full_coordinates)
                sudoku_copy = sudoku.copy()
                
                ##
        
    cv2.destroyAllWindows()
    cap.release()
    
if __name__ == "__main__":
    main()