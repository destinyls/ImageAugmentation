import os

label_path = "datasets/kitti/testing/label_2"
if __name__ == "__main__":
    for idx in range(7518):
        label_name = "{:06d}".format(idx) + ".txt"
        label_name = os.path.join(label_path, label_name)
        if not os.path.exists(label_name):
            with open(label_name,'w') as f:
                print(label_name, "created ..." )


