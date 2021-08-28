import os
from posixpath import split

label_path = "/home/yanglei/Dataset/kitti/yang"
split_names = ["cresult_split1", "cresult_split2", "cresult_split3", "cresult_split4", "cresult_split5", "cresult_split6", "cresult_split7"]

if __name__ == "__main__":
    for split_n in split_names:
        split_path = os.path.join(label_path, split_n)
        for idx in range(7518):
            label_name = "{:06d}".format(idx) + ".txt"
            label_name = os.path.join(split_path, label_name)
            if not os.path.exists(label_name):
                with open(label_name,'w') as f:
                    print(label_name, "created ..." )