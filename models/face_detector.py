import sys
import os
import cv2

# extract the pre-trained face detector
face_cascade = cv2.CascadeClassifier(os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                                 'haarcascade_frontalface_alt.xml'))

def face_detector(img_path):
    """
    This function takes an image path and returns the number of faces detected in it

    Args:
        img_path: path to the image file

    Returns:
        The number of faces detected in the image
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s IMG_PATH' % sys.argv[0])
        sys.exit(1)
    face_count = face_detector(sys.argv[1])
    print('%d faces were detected in "%s"' % (face_count, sys.argv[1]))
