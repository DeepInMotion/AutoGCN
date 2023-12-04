import logging
import os
import sys
import time
from time import strftime, localtime


def set_logging(args):
    """
    Setup logging for current run.

    @param args: parsed user input arguments
    @return: save_dir:str
    """
    ct = strftime('%Y-%m-%d %H-%M-%S')
    save_dir = '{}/{}_{}/{}'.format(args.work_dir, args.config, args.dataset, ct)
    log_folder(save_dir)
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    handler = logging.FileHandler('{}/logfile.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)
    return save_dir


def log_folder(folder):
    """
    Generate a log folder for output.

    @param folder: dir:str
    @return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
