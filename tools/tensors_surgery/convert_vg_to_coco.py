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
    tensor_names = [
        tensor_name for tensor_name, tensor in state_dict.items()
        if 1601 in tensor.shape or 1601*4 in tensor.shape
    ]
    for tensor_name in tensor_names:
        tensor = state_dict[tensor_name]
        print('Tensor {} : {}'.format(tensor_name, tensor.shape))
        new_dim = 81 if 1601 in tensor.shape else 81 * 4
        backup_tensor_name = '{}_backup'.format(tensor_name)
        tensor_backup = tensor.clone()
        model['state_dict'][tensor_name] = tensor[:new_dim]
        print('Will be changed to {} : {}'.format(tensor_name, model['state_dict'][tensor_name].shape))
        print('Previous tensor can be restored from backup: {}'.format(backup_tensor_name))
        model['state_dict'][backup_tensor_name] = tensor_backup
    output_path = args.output_model
    if not output_path:
        output_path = '1601_to_81_{}'.format(osp.basename(args.input_model))
    print('Save to {}'.format(output_path))
    torch.save(model, output_path)


if __name__ == '__main__':
    main()
