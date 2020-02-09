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

def generate_image_info(data_dir, imageset):
    with open(osp.join(data_dir, 'image_data.json')) as f:
        raw_img_data = json.load(f)

    images = []
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

    return images


def generate_annotation(data_dir, label2cid):
    with open( osp.join(data_dir, 'objects.json')) as f:
        raw_obj_data = json.load(f)
    annotations = []
    for img in raw_obj_data:
        for obj in img['objects']:
            synsets = obj['names']
            if synsets[0] not in label2cid:
                continue
            cid = label2cid[synsets[0]]
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
        {'id': (n + 1), 'name': label} for n, label in enumerate(labels)]
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
    images = generate_image_info(args.data_dir, imageset)
    categories, label2cid = generate_categories(args.data_dir)
    annotations = generate_annotation(args.data_dir, label2cid)
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
