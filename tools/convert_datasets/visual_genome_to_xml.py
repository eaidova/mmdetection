import json
import os
import os.path as osp
import xml.etree.cElementTree as ET
import argparse
import mmcv

DEFAULT_DATA_DIR = './data/vg'
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR
obj_vocab = 'objects_vocab.txt'

common_attributes = {
    'white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'silver', 'wooden', 'orange', 'gray',
    'grey', 'metal', 'pink', 'tall', 'long', 'dark'
}


def clean_string(string):
    string = string.lower().strip()
    if len(string) >= 1 and string[-1] == '.':
        return string[:-1].strip()
    return string


def clean_objects(string):
    string = clean_string(string)
    words = string.split()
    if len(words) > 1:
        prefix_words_are_adj = True
        for att in words[:-1]:
            if att not in common_attributes:
                prefix_words_are_adj = False
        if prefix_words_are_adj:
            return words[-1:], words[:-1]
        else:
            return [string], []
    else:
        return [string], []


def read_objects_vocab(vocab_file):
    with open(vocab_file) as f:
        objects = f.readlines()
    return objects


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Visual Genome dataset to Pascal VOC xml format')
    parser.add_argument('--data_dir', type=str, required=False, default=DEFAULT_DATA_DIR)
    parser.add_argument('--out_dir', type=str, required=False, default=DEFAULT_OUT_DIR)
    parser.add_argument('--objects_vocab_file', type=str, required=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    build_xml(args)


def build_xml(args):
    with open(osp.join(args.data_dir, 'scene_graphs.json')) as f:
        data = json.load(f)
    # Load image metadata
    with open(osp.join(args.data_dir, 'image_data.json')) as f:
        metadata = {item['image_id']: item for item in json.load(f)}
    vocab = args.objects_vocab_file if args.objects_vocab_file else osp.join(args.data_dir, obj_vocab)
    objects = read_objects_vocab(vocab)

    # Output clean xml files, one per image
    out_folder = 'xml'
    if not osp.exists(osp.join(args.out_dir, out_folder)):
        os.mkdir(osp.join(args.out_dir, out_folder))
    print('Converted annotations will be stored in {}'.format(osp.join(args.out_dir, out_folder)))
    prog_bar = mmcv.ProgressBar(len(data))
    for sg in data:
        ann = ET.Element("annotation")
        meta = metadata[sg["image_id"]]
        assert sg["image_id"] == meta["image_id"]
        url_split = meta["url"].split("/")
        ET.SubElement(ann, "folder").text = url_split[-2]
        ET.SubElement(ann, "filename").text = url_split[-1]

        source = ET.SubElement(ann, "source")
        ET.SubElement(source, "database").text = "Visual Genome Version 1.2"
        ET.SubElement(source, "image_id").text = str(meta["image_id"])
        ET.SubElement(source, "coco_id").text = str(meta["coco_id"])
        ET.SubElement(source, "flickr_id").text = str(meta["flickr_id"])

        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(meta["width"])
        ET.SubElement(size, "height").text = str(meta["height"])
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(ann, "segmented").text = "0"

        object_set = set()
        for obj in sg['objects']:
            o, _ = clean_objects(obj['names'][0])
            if o[0] in objects:
                ob = ET.SubElement(ann, "object")
                ET.SubElement(ob, "name").text = o[0]
                ET.SubElement(ob, "object_id").text = str(obj["object_id"])
                object_set.add(obj["object_id"])
                ET.SubElement(ob, "difficult").text = "0"
                bbox = ET.SubElement(ob, "bndbox")
                ET.SubElement(bbox, "xmin").text = str(obj["x"])
                ET.SubElement(bbox, "ymin").text = str(obj["y"])
                ET.SubElement(bbox, "xmax").text = str(obj["x"] + obj["w"])
                ET.SubElement(bbox, "ymax").text = str(obj["y"] + obj["h"])

        outFile = url_split[-1].replace(".jpg", ".xml")
        tree = ET.ElementTree(ann)
        prog_bar.update()
        if len(tree.findall('object')) > 0:
            tree.write(osp.join(args.out_dir, out_folder, outFile))


if __name__ == "__main__":
    main()

