
import cv2

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


def detect(path,img):
    cascade = cv2.CascadeClassifier(path)

    img=cv2.imread(img,1)
    # converting to gray image for faster video processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = cascade.detectMultiScale(gray, 1.2, 3,minSize=(50, 50))
    # if at least 1 face detected
    if len(rects) >= 0:
        # Draw a rectangle around the faces
        for (x, y, w, h) in rects:
            draw_border(img, (x, y), (x + w, y + h), (255, 0, 105),4, 15, 10)
        # Display the resulting frame
        cv2.imshow('Face Detection', img)
        # wait for 'c' to close the application
        cv2.waitKey(0)


def main():
    cascadeFilePath = "./haar/haarcascade_frontalface_default.xml"
    img='./hanif.jpg'
    detect(cascadeFilePath,img)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()