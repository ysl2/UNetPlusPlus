# ==============
# === Unet++ ===
# ==============
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
import json
import torch

_tasknum = '602'
_taskname = 'Z2'

_myroot = '/home/yusongli/_dataset/shidaoai/img/_out/nn'
_mydata = f'{_myroot}/DATASET'
_mymodel = 'nnUNet'

base = f'{_mydata}/{_mymodel}_raw_data_base'
preprocessing_output_dir = f'{_mydata}/{_mymodel}_preprocessed'
network_training_output_dir_base = f'{_mydata}/{_mymodel}_cropped_data'

num_batches_per_epoch = 250
use_this_for_batch_size_computation_3D = 520000000 * 2  # 505789440


def net():
    """
    Network Factory
    """
    from nnunet.network_architecture.generic_UNetPlusPlus import Generic_UNetPlusPlus
    from nnunet.network_architecture.generic_XNet import Generic_XNet
    from nnunet.network_architecture.generic_hipp_XNet import Generic_XNet as Generic_XNet_hipp

    return Generic_UNetPlusPlus


def splits_final():
    """
    split train val set.
    """
    with open(f'{_myroot}/n_to_s.json', 'r') as f:
        n_to_s = json.load(f)
    pkl_name = 'splits_final.pkl'
    pkl = f'{preprocessing_output_dir}/Task{_tasknum}_{_taskname}/{pkl_name}'
    obj = load_pickle(pkl)
    obj = [
        {
            'train': [f'Z2_{item}' for item in list(n_to_s['training'])],
            'val': [f'Z2_{item}' for item in list(n_to_s['validation'])],
        }
    ]
    save_pickle(obj, pkl)


def batch_size():
    """
    Change batch size.
    """
    pkl_name = 'nnUNetPlansv2.1_plans_3D.pkl'
    pkl = f'{preprocessing_output_dir}/Task{_tasknum}_{_taskname}/{pkl_name}'
    obj = load_pickle(pkl)
    obj['plans_per_stage'][0]['batch_size'] = 22  # default: 30
    save_pickle(obj, pkl)

# def fix_bug():
#     """
#     My personal settings.
#     """
#     import collections
#     from thesmuggler import smuggle
#     pp = smuggle('../../shidaoai_new_project/data/pathparser.py')

#     myd = {'training': [1, 1333], 'validation': [1333, 1665], 'test': [1, 265]}  # [a,b)
#     n_to_s_str = f'{_myroot}/n_to_s_.json_'
#     n_to_s_str_save = f'{_myroot}/n_to_s.json'

#     with open(n_to_s_str, 'r') as f:
#         n_to_s = json.load(f)

#         # add new index
#         n_to_s[list(myd)[0]]['0000'] = None
#         n_to_s[list(myd)[1]]['1332'] = None
#         n_to_s[list(myd)[2]]['0000'] = None

#         for tag in myd.keys():

#             # 1. move things to neighbor
#             templ = [f'{i:04d}' for i in range(myd[tag][0], myd[tag][1])]
#             for key in templ:
#                 newkey = int(key) - 1
#                 n_to_s[tag][f'{newkey:04d}'] = n_to_s[tag][key]
#             del n_to_s[tag][templ[-1]]

#             # 2. reorder index
#             templ2 = list(n_to_s[tag])
#             templ3 = [templ2[-1]]
#             for iii in range(0, len(templ2) - 1):
#                 templ3.append(templ2[iii])
#             temp = [(k, n_to_s[tag][k]) for k in templ3]

#             # 3. reconstruct dict
#             n_to_s[tag] = collections.OrderedDict(temp)
#         pp.savejson(n_to_s, n_to_s_str_save)


def test_net():
    """
    Test network.
    """
    test_model = net(
        1,
        32,
        2,
        4,
        num_conv_per_stage=2,
        conv_op=torch.nn.modules.conv.Conv3d,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=torch.nn.modules.dropout.Dropout3d,
        dropout_op_kwargs={'p': 0, 'inplace': True},
        nonlin=torch.nn.modules.activation.LeakyReLU,
        nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
        final_nonlin=lambda x: x,
        pool_op_kernel_sizes=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
        conv_kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        convolutional_pooling=True,
        convolutional_upsampling=True,
    )
    x = torch.randn([30, 1, 24, 64, 80])
    test_model(x)


if __name__ == '__main__':
    # test_net()
    # splits_final()
    batch_size()
