import json
import os.path as osp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Visual Genome dataset to COCO format')
    parser.add_argument('--data_dir', type=str, required=False, default='./data/vg')
    parser.add_argument('--out_file', type=str, required=False)
    parser.add_argument('--objects_vocab_file', type=str, required=False)
    parser.add_argument('--image_subset', type=str, required=False)
    args = parser.parse_args()

    return args

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


def generate_image_info(data_dir, imageset):
    with open(osp.join(data_dir, 'image_data.json')) as f:
        raw_img_data = json.load(f)

    images = []
    image_ids = []
    for img in raw_img_data:
        file_name = img['url'].replace('https://cs.stanford.edu/people/rak248/', '')
        if file_name not in imageset:
            continue
        images.append({
            'id': img['image_id'],
            'width': img['width'],
            'height': img['height'],
            'file_name': file_name,
            'coco_id': img['coco_id']}
        )
        image_ids.append(img['image_id'])

    return images, image_ids


def generate_annotation(data_dir, image_ids, label2cid):
    with open(osp.join(data_dir, 'scene_graphs.json')) as f:
        raw_obj_data = json.load(f)
    annotations = []
    for img in raw_obj_data:
        if img['image_id'] not in image_ids:
            continue
        for obj in img['objects']:
            label, _ = clean_objects(obj['names'][0])
            if label[0] not in label2cid:
                continue
            cid = label2cid[label[0]]
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            area = obj['w'] * obj['h']
            ann = {
                'id': obj['object_id'],
                'image_id': img['image_id'],
                'category_id': cid,
                'segmentation': [],
                'area': area,
                'bbox': bbox,
                'iscrowd': 0
            }
            annotations.append(ann)

    return annotations


def generate_categories(data_dir):
    with open(osp.join(data_dir, 'objects_vocab.txt')) as f:
        labels = f.readlines()
    categories = [
        {'id': (n + 1), 'name': label.replace('\n', '')} for n, label in enumerate(labels)]
    label2cid = {c['name']: c['id'] for c in categories}

    return categories, label2cid


def get_images_set(imageset_file):
    with open(imageset_file) as f:
        imageset = list(map(lambda pair: pair.split(' ')[0], f.readlines()))
    return imageset


def main():
    args = parse_args()
    image_subset = args.image_subset or osp.join(args.data_dir, 'test.txt')
    imageset = get_images_set(image_subset)
    images, image_ids = generate_image_info(args.data_dir, imageset)
    categories, label2cid = generate_categories(args.data_dir)
    annotations = generate_annotation(args.data_dir, image_ids, label2cid)
    output_file = args.out_file or osp.join(args.data_dir, 'instances_vg_test.json')
    print('Annotation will be saved to {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(
            {
                'images': images,
                'annotations': annotations,
                'categories': categories
             },
            f
        )


if __name__ == "__main__":
    main()
