import cv2 as cv
import urllib.request as request
import os

weigths_path = "data/weights/"
cfg_path = "data/cfg/"
class_path = "data/"
image_path = "data/images/"

def load_image(image, x_scale=1, y_scale=1):
    image = image_path + image
    img = cv.imread(image)
    img = cv.resize(img, None, fx=x_scale, fy=y_scale)
    return img

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def download(url, destination, filename):
    destination = destination + filename
    if os.path.isfile(destination):
        print(filename + " already exists at the provided destination!")
        return
    print("Downloading " + filename + "...")
    request.urlretrieve(url, destination)
    print("Finished downloading " + filename + "!")
