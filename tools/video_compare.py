import os
import cv2
import numpy as np

model_demo1 = "Lite-FPN_WEIGHTED_LOSS_001nd_0021000"
model_demo2 = "Lite-FPN_CORNERS_TRAINVAL_001nd"

image_ids = "/home/yanglei/Desktop/scenes/2011_09_26"

scenes_list = ["2011_09_26_drive_0009_sync", "2011_09_26_drive_0093_sync", "2011_09_26_drive_0014_sync", "2011_09_26_drive_0095_sync"]
# scenes_list = ["2011_09_26_drive_0057_sync", "2011_09_26_drive_0018_sync", "2011_09_26_drive_0059_sync", "2011_09_26_drive_0104_sync"]

#scenes_list = ["2011_09_26_drive_0009_sync", "2011_09_26_drive_0013_sync", "2011_09_26_drive_0014_sync", "2011_09_26_drive_0018_sync", "2011_09_26_drive_0027_sync",
#               "2011_09_26_drive_0056_sync", "2011_09_26_drive_0057_sync", "2011_09_26_drive_0059_sync", "2011_09_26_drive_0093_sync", "2011_09_26_drive_0095_sync",
#               "2011_09_26_drive_0104_sync"]


if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 10
    sice = (2484, 797)
    video = cv2.VideoWriter("compare.mp4", fourcc, 10, sice)
    
    id = 0
    for scene in scenes_list:
        sub_scene_model1 = os.path.join(model_demo1, scene)
        sub_scene_model2 = os.path.join(model_demo2, scene)
        image_ids_path = os.path.join(image_ids, scene, "image_ids.txt")

        with open(image_ids_path, 'r') as f:
            for line in f.readlines():
                image_path1 = os.path.join(sub_scene_model1, line[:-1] + ".jpg")
                image_path2 = os.path.join(sub_scene_model2, line[:-1] + "_revised.jpg")

                if not os.path.exists(image_path1) or not os.path.exists(image_path2):
                    continue

                print(image_path1)

                image1 = cv2.imread(image_path1)
                image2 = cv2.imread(image_path2)

                total_image = np.vstack([image1, image2])

                video.write(total_image)

                cv2.imshow("total image", total_image)
                cv2.imwrite("compare_005/"+str(id)+".jpg", total_image)
                id = id + 1
                cv2.waitKey(1)
                print(total_image.shape)
    
    video.release()
