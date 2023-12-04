import argparse

from ntu_gendata import NTUGendata
from src.utils import io

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data Converter for NTU, Kinetics and CP dataset.')
    parser.add_argument('--choice', default='ntu60', type=str, choices=['ntu60', 'ntu120', 'kinetics', 'CP', 'all'],
                        help='Choose between [ntu60, ntu120, kinetics, CP, all]')
    parser.add_argument('--trans', default="true", type=str, help='Transform the data?')
    parser.add_argument('--trans_opt', default=['pad', 'sub', 'parallel_s', 'parallel_h'], nargs="*",
                        choices=['pad', 'parallel_s', 'parallel_h', 'sub', 'view', 'scale', 'none'],
                        help='Options for transformation?')
    # --trans_opt
    # pad
    # sub
    # parallel_h
    # parallel_s
    # view
    # scale
    parser.add_argument('--debug', default='false', type=str, help='Debug mode')
    parser.add_argument('--vis', default='false', type=str, help='Visualize skeleton')

    args = parser.parse_args()
    # cast input to bool
    args.trans = io.str2bool(args.trans)
    args.debug = io.str2bool(args.debug)
    args.vis = io.str2bool(args.vis)
    datasets = ['ntu60', 'ntu120']

    print("Processing: [{}]".format(args.choice))
    if args.choice == "ntu60" or args.choice == "ntu120":
        # data_path = path_args.skeletons.path.raw.ntu
        # save_path = path_args.skeletons.path.processed.ntu
        # ntu60_path = path_args.skeletons.path.raw.ntu60
        # ntu120_path = path_args.skeletons.path.raw.ntu120
        ntu60_path = '/home/espen/Documents/data/ntu/raw/nturgb+d_skeletons'
        ntu120_path = '/home/espen/Documents/data/ntu/raw/nturgbd_skeletons_s018_to_s032'
        root_path = '/home/espen/Documents/data/ntu/npy_files'
        ignore_path = '/home/espen/Documents/data/ntu/raw/ignore.txt'
        # TODO put your paths here
        ntu60_path = '...'
        ntu120_path = '...'
        root_path = '...'
        ignore_path = '...'
        generator = NTUGendata(root_path, ntu60_path, ntu120_path, ignore_path, args)
        generator.process(cores=4)
        # generator.gendata_ntu()
    else:
        raise ValueError("Choice [{}] not supported!".format(args.choice))

    print("Done processing {}".format(args.choice))
