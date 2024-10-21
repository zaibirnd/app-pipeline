import cv2
import numpy as np
from skimage import io
import pandas as pd


def get_polygons(image_path):
    mask = io.imread(mask_image_path, as_gray=True)
    print(type(mask))
    name = 'name'
    df = pd.DataFrame(columns=['image_name', 'points'])
    # Ensure the image is binary
    _, binary_mask = cv2.threshold((mask * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons
    polygons = [cv2.approxPolyDP(cnt, epsilon=0.01 * cv2.arcLength(cnt, True), closed=True) for cnt in contours]
    #print(len(polygons))
    for polygon in polygons:
        #print(polygon)
        arr = [name, polygon]
        df.loc[len(df.index)]=arr
    return polygons,mask, df

def plot_polygons(polygons,mask ):
    blank_image = np.zeros_like(mask)
    for polygon in polygons:
        #print("Polygon points:", polygon)
        print(polygon)

        # Optionally, you can draw these polygons on a blank image

        cv2.polylines(blank_image, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)
        # Show the polygon image
    return blank_image

# Load the mask image
mask_image_path = 'c.jpg'
polygons, mask, df = get_polygons(mask_image_path)
result = plot_polygons(polygons, mask)
df.to_csv('rs.csv')
cv2.imwrite('result1.jpg', result)
#cv2.imshow('Polygon', result)
#cv2.waitKey(0)

#cv2.destroyAllWindows()
#print(df.head())
