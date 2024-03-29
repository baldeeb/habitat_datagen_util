import cv2 
import matplotlib.pyplot as plt 
import numpy as np

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, boxes):
    if not isinstance(img, np.ndarray):
        img = img.clone().permute(1,2,0).detach().cpu().numpy()
    for a in boxes:
        x_min, y_min = int(a[0]), int(a[1])
        x_max, y_max = int(a[2]), int(a[3])
        color = (0, 255, 0)
        cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)
    plt.imsave('temp.png', img/img.flatten().max())