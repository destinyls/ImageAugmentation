import os
import cv2
import numpy as np

if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 7
    sice = (2070, 625)
    video = cv2.VideoWriter("new_demo.mp4", fourcc, fps, sice)
    
    for id in range(998):
        image_path = "compare_005/" + str(id) + ".jpg"

        if not os.path.exists(image_path):
            continue
        
        image = cv2.imread(image_path)

        image[374:376, :, :] = 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Baseline-SMOKE", (1250, 65), font, 1.2, (0,0,0), 4)
        cv2.putText(image, "Ours", (1250, 438), font, 1.2, (0,0,0), 4)

        cv2.putText(image, "25m", (2150, 720), font, 0.8, (0,0,0), 3)
        cv2.putText(image, "|", (2173, 748), font, 0.5, (0,0,0), 3)
        
        cv2.putText(image, "50m", (1840, 720), font, 0.8, (0,0,0), 3)
        cv2.putText(image, "|", (1863, 748), font, 0.5, (0,0,0), 3)

        cv2.putText(image, "75m", (1507, 720), font, 0.8, (0,0,0), 3)
        cv2.putText(image, "|", (1530, 748), font, 0.5, (0,0,0), 3)


        resize_image = cv2.resize(image, sice, interpolation=cv2.INTER_AREA)

        print(image.shape)

        video.write(resize_image)
        cv2.imshow("total image", resize_image)
        cv2.waitKey(1)
    video.release()
