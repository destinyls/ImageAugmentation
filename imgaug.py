class ImgAug():
    def __init__(self, root, is_train=True):
        super(ImgAug, self).__init__()
        self.root = root
    
    info_path = os.path.join(root, "../kitti_infos_train.pkl")

    db_info_path = os.path.join(root, "../kitti_dbinfos_train.pkl")
    with open(db_info_path, 'rb') as f:
        db_infos = pickle.load(f)
    self.car_db_infos = db_infos["Car"]
    self.ped_db_infos = db_infos["Pedestrian"]
    self.cyc_db_infos = db_infos["Cyclist"]

    if self.split == "train":
            imageset_txt = os.path.join(root, "ImageSets", "train.txt")
        elif self.split == "val":
            imageset_txt = os.path.join(root, "ImageSets", "val.txt")
        elif self.split == "trainval":
            imageset_txt = os.path.join(root, "ImageSets", "trainval.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "ImageSets", "test.txt")
        else:
            raise ValueError("Invalid split!")
    image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        self.image_files = image_files
    self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
    self.num_samples = len(self.image_files)

    def __len__(self):
        return self.num_samples

    def apply_augmentation(self, idx):
        annos, P2, P3 = self.load_annotations(idx)
        use_left = True
        if random.random() < self.right_prob:
            use_left = False
            K = P3[:3, :3]
            P = P3
            img_path = os.path.join(self.image3_dir, self.image_files[idx])
        else:
            K = P2[:3, :3]
            P = P2
            img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = cv2.imread(img_path)
        bboxes = []
        for i, a in enumerate(annos):
            annos[i]["P"] = P
            a = a.copy()
            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            point, box2d, box3d = encode_label(P, rot_y, a["dimensions"], locs)
            if (0 < point[0] < img.shape[1]) and (0 < point[1] < img.shape[0]):
                bboxes.append(box2d)
        bboxes = np.array(bboxes)
        if len(bboxes.shape) > 1:
            bboxes[:,[0, 2]] = np.clip(bboxes[:,[0, 2]], 0, img.shape[1] - 1)
            bboxes[:,[1, 3]] = np.clip(bboxes[:,[1, 3]], 0, img.shape[0] - 1)

        '''  box copy paste  '''
        img_shape_key = f"{img.shape[0]}_{img.shape[1]}"
        annos_increased = []
        if img_shape_key in self.car_db_infos.keys():
            car_db_infos_t = self.car_db_infos[img_shape_key]
            ins_ids = sample(range(len(car_db_infos_t)), min(15, self.max_objs - len(annos)))
            for i in ins_ids:
                ins = car_db_infos_t[i]
                box2d = ins["box2d_l"] if use_left else ins["box2d_r"]
                if ins['difficulty'] > 0:
                    continue
                if len(bboxes.shape) > 1:
                    ious = kitti.iou(bboxes, box2d[np.newaxis, ...])
                    if np.max(ious) > 0.0:
                        continue
                    bboxes = np.vstack((bboxes, box2d[np.newaxis, ...]))
                else:
                    bboxes = bbox[np.newaxis, ...].copy()            
                path = ins["path"] if use_left else ins["path"].replace("image_2", "image_3")
                patch_img_path = os.path.join("datasets/kitti", path)
                patch_img = cv2.imread(patch_img_path)
                img[int(box2d[1]):int(box2d[3]), int(box2d[0]):int(box2d[2]), :] = patch_img
                anno = dict()
                anno["calss"] = ins["name"]
                anno["label"] = TYPE_ID_CONVERSION[ins["name"]]
                anno["truncation"] = -1
                anno["occlusion"] = -1
                anno["alpha"] = ins["alpha"]
                anno["dimensions"] = ins["dim"]
                anno["locations"] = ins["loc"]
                anno["rot_y"] = ins["roty"]
                anno["P"] = ins["P2"] if use_left else ins["P3"]
                annos_increased.append(anno)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img, annos, annos_increased, use_left