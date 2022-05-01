# ==============
# === Unet++ ===
# ==============
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
from thesmuggler import smuggle
import collections
pp = smuggle('../../shidaoai_new_project/data/pathparser.py')
import json

_tasknum = '602'
_taskname = 'Z2'

_myroot = '/home/yusongli/_dataset/shidaoai/img/_out/nn'
_mydata = f'{_myroot}/DATASET'
_mymodel = 'nnUNet'

base = f'{_mydata}/{_mymodel}_raw_data_base'
preprocessing_output_dir = f'{_mydata}/{_mymodel}_preprocessed'
network_training_output_dir_base = f'{_mydata}/{_mymodel}_cropped_data'


def splits_final():
    with open(f'{_myroot}/n_to_s.json', 'r') as f:
        n_to_s = json.load(f)
    pkl = f'{preprocessing_output_dir}/Task{_tasknum}_{_taskname}/splits_final.pkl'
    obj = load_pickle(pkl)
    obj = [
            {
                'train': [f'Z2_{item}' for item in list(n_to_s['training'])],
                'val': [f'Z2_{item}' for item in list(n_to_s['validation'])]
            }
    ]
    save_pickle(obj, pkl)


def fix_bug():
    myd = {
        'training': [1, 1333], # [a,b)
        'validation': [1333, 1665],
        'test': [1, 265]
    }
    n_to_s_str = f'{_myroot}/n_to_s_.json_'
    n_to_s_str_save = f'{_myroot}/n_to_s.json'

    with open(n_to_s_str, 'r') as f:
        n_to_s = json.load(f)

        # add new index
        n_to_s[list(myd)[0]]['0000'] = None
        n_to_s[list(myd)[1]]['1332'] = None
        n_to_s[list(myd)[2]]['0000'] = None

        for tag in myd.keys():

            # 1. move things to neighbor
            templ = [f'{i:04d}' for i in range(myd[tag][0], myd[tag][1])]
            for key in templ:
                newkey = int(key) - 1
                n_to_s[tag][f'{newkey:04d}'] = n_to_s[tag][key]
            del n_to_s[tag][templ[-1]]

            # 2. reorder index
            templ2 = list(n_to_s[tag])
            templ3 = [templ2[-1]]
            for iii in range(0, len(templ2) - 1):
                templ3.append(templ2[iii])
            temp = [(k, n_to_s[tag][k]) for k in templ3]

            # 3. reconstruct dict
            n_to_s[tag] = collections.OrderedDict(temp)
        pp.savejson(n_to_s, n_to_s_str_save)


if __name__ == '__main__':
    splits_final()
