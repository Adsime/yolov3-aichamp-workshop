import cv2 as cv
from matplotlib import pyplot as plt

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

    #plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    #plt.show()

