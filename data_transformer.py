import copy
import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class BrandImageDataset(CustomDataset):
    CLASSES = ('ae', 'hrc', 'nfl', 'mcd', 'sbux', 'subway')
    DATA_PATH = '../data/irl_data_cpy' # TODO: Figure out how to pass this in

    def load_annotations(self, ann_file):
        brand2label = {k : i for i, k in enumerate(self.CLASSES)}
        data_prefix = mmcv.list_from_file(ann_file)
        data_infos = []

        for prefix in data_prefix:
            annotation_f = os.path.sep.join([self.DATA_PATH, 'annotation',
                                            prefix + '.xml'])
            tree = ET.parse(annotation_f)
            root = tree.getroot()
            filename = root.find('path').text
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            image = mmcv.imread(filename)

            data_info = dict(filename=filename, width=width, height=height)

            gt_labels = []
            gt_bboxes = []

            # load annotations
            for obj in root.iter('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                label = obj.find('name').text

                gt_labels.append(brand2label[label])
                gt_bboxes.append((xmin, ymin, xmax, ymax))

            data_anno = dict(
                bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels = np.array(gt_labels, dtype=np.long)
            )

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos

# Only need to run once with newly labeled data to generate train/val splits
if __name__ == "__main__":
    import argparse

    from sklearn.model_selection import train_test_split
    
    # Parse args
    parser = argparse.ArgumentParser(description='process data into mmdet-readable intermediate format')
    parser.add_argument('-d', '--data_root', required=True,
                        help="path to data directory root")
    parser.add_argument('-i', '--images', default='images',
                        help="images directory name within data_root")
    parser.add_argument('-a', '--annotations', default='annotation',
                        help="annotations directory name within data_root")
    parser.add_argument('-t', '--test_size', type=float, default=0.2,
                        help="ratio of original data to be used as validation, default 0.2")
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help="pseudorandom number generator seed to use, default 42")
    argv = vars(parser.parse_args())

    # Split dataset
    DATA_ROOT = argv['data_root']
    IMG_PATH = os.path.sep.join([DATA_ROOT, argv['images']])
    ANN_PATH = os.path.sep.join([DATA_ROOT, argv['annotations']])

    img_fnames = []
    ann_fnames = []

    # Caution - you may need to change the logic here if your annotations and
    #           images have different naming conventions
    for f in os.listdir(ANN_PATH):
        ann_fnames.append(f.split('.')[0])
        img_fnames.append(f.split('.')[0])

    X_train, X_test, y_train, y_test = train_test_split(img_fnames, ann_fnames,
                                                        test_size=argv['test_size'],
                                                        random_state=argv['seed'])

    # Write splits files to data root
    train_f = os.path.sep.join([DATA_ROOT, 'train.txt'])
    val_f = os.path.sep.join([DATA_ROOT, 'val.txt'])

    with open(train_f, 'w+') as f:
        for x in X_train:
            f.write(x + '\n')
        f.close()

    with open(val_f, 'w+') as f:
        for x in X_test:
            f.write(x + '\n')
        f.close()
