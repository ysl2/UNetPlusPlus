# ==============
# === Unet++ ===
# ==============
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
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


def custom_plans(plan_obj: dict) -> dict:
    plan_obj['plans_per_stage'][0]['batch_size'] = 22  # default: 30
    return plan_obj


def _test_net():
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
    pass
