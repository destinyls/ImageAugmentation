import os

label_name = "/root/SMOKE/datasets/kitti/ImageSets/test.txt"

if __name__ == "__main__":
    with open(label_name,'w') as f:
        for idx in range(7518 * 7):
            frame_name = "{:06d}".format(idx)
            f.write(frame_name)
            f.write("\n")
        