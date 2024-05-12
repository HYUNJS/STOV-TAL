import json, os
import os.path as osp

data_root = '/root/datasets/uk600/'
anno_root = osp.join(data_root, 'annotations')
val_anno_filepath = osp.join(anno_root, 'validation_tal.json')
with open(val_anno_filepath, 'r') as fp:
    data = json.load(fp)['database']
    
data