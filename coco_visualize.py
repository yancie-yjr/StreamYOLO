from unicodedata import category
import cv2
import random
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

def visualization_bbox2(meta_path, json_path, img_dir):
    meta = json.load(open(meta_path))
    categories = meta["categories"]
    seq_dirs = meta["seq_dirs"]
    images = meta["images"]

    test_annos = json.load(open(json_path))
    visual_dict = {}

    for test_anno in test_annos:
        image_id = test_anno["image_id"]
        category_id = test_anno["category_id"]
        bbox = test_anno["bbox"]
        score = test_anno["score"]
        segmentation = test_anno["segmentation"]
        
        image = images[image_id]
        sid = image["sid"]
        dir_name = seq_dirs[sid]
        image_name = image["name"]
        image_path = os.path.join(img_dir, dir_name, image_name)
        if score > 0.3:
            visual_dict.setdefault(image_path, [])
            visual_dict[image_path].append(bbox)
        
    # print(visual_dict)   
    for image_path, bboxes in visual_dict.items():
        img = cv2.imread(image_path)
        print("Detecting from: \n", image_path)
        for bbox in bboxes:
            x, y, w, h = bbox  # 读取边框x y w h 
            img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
        
        # cv2.imshow(image_path, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    

        path_list = image_path.split('/')
        save_dir = './output/' + path_list[-3] + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + path_list[-1]
        print("Saving to: \n", save_path)
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    meta_json = "./data/Argoverse-HD/annotations/test-meta.json"
    test_json = './yolox_testdev_2017.json'
    img_dir  = './data/Argoverse-1.1/tracking/'
    visualization_bbox2(meta_json, test_json, img_dir)