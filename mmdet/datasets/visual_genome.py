import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VisualGenomeXMLDataset(XMLDataset):
    def __init__(self, **kwargs):
        data_root = kwargs.get('data_root')
        self.classes_file = './data/vg/objects_vocab.txt' if data_root is None else osp.join(data_root, 'objects_vocab.txt')
        self.load_classes()
        super(VisualGenomeXMLDataset, self).__init__(**kwargs)

    def load_classes(self):
        classes_list = mmcv.list_from_file(self.classes_file)
        self.CLASSES = tuple(classes_list)

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids_annotation_ids = mmcv.list_from_file(ann_file)
        img_ids = []
        for img_id, img_ann_id in enumerate(img_ids_annotation_ids):
            filename, ann_id = img_ann_id.split()
            xml_path = osp.join(self.img_prefix, ann_id)
            if not osp.exists(xml_path):
                continue
            img_ids.append(int(ann_id.split('/')[-1].split('.xml')[0]))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height, annotation_id=ann_id))
        self.img_ids = img_ids
        self.cat_ids = list(range(1, len(self.CLASSES) + 1))
        return img_infos

    def get_ann_info(self, idx):
        annotation_id = self.img_infos[idx]['annotation_id']
        xml_path = osp.join(self.img_prefix, annotation_id)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.cat2label:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
