import argparse
import os.path as osp
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=False, default='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Load model from {}'.format(args.input_model))
    model = torch.load(args.input_model)
    state_dict = model['state_dict']
    backout_tensor_names = [tensor_name for tensor_name in state_dict if tensor_name.endswith('_backup')]
    if not backout_tensor_names:
        raise ValueError('no backup tensors')
    for tensor_name in backout_tensor_names:
        backup_tensor = state_dict[tensor_name]
        print('Backup tensor {} : {}'.format(tensor_name, backup_tensor.shape))
        original_tensor_name = tensor_name.split('_backup')[0]
        print('Replace original {}, backup will be removed'.format(original_tensor_name))
        model['state_dict'][original_tensor_name] = backup_tensor
        model['state_dict'].pop(tensor_name)
    output_path = args.output_model
    if not output_path:
        output_path = 'restore_{}'.format(osp.basename(args.input_model))
    print('Save to {}'.format(output_path))
    torch.save(model, output_path)


if __name__ == '__main__':
    main()
