
# import the opencv library
import cv2
from matplotlib import pyplot as plt


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(path)

def detect(img):
    # converting to gray image for faster video processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = cascade.detectMultiScale(gray, 1.2, 3,minSize=(50, 50))
    # if at least 1 face detected
    if len(rects) >= 0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in rects:
            draw_border(img, (x, y), (x + w, y + h), (255, 0, 105),4, 15, 10)


# define a video capture object
vid1 = cv2.VideoCapture(0)
vid2 = cv2.VideoCapture(1)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

while(True):
    # Capture the video frame by frame
    ret1, frame1 = vid1.read()
    detect(frame1)
    # Display the resulting frame
    cv2.imshow('frame1', frame1)

    ret2, frame2 = vid2.read()
    detect(frame2)
    # Display the resulting frame
    cv2.imshow('frame2', frame2)

    frame_1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(frame_1_gray,frame_2_gray)
    
    norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('norm_image', norm_image)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
