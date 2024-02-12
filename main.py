import sys
import logging
import platform
import argparse
from omegaconf import OmegaConf


def main():
    # load config
    parser = argparse.ArgumentParser(description='AutoGCN')
    parser.add_argument('--config', '-c', default='./config/config_has.yaml', type=str, help='Choose the config file')
    conf = parser.parse_args()
    args = OmegaConf.load(conf.config)

    if args.cont_training:
        # cont. training
        cont_dir = args.cont_dir
        assert cont_dir is not None
        args = OmegaConf.load('{}/config.yaml'.format(cont_dir))
        args.cont_training = True
        args.cont_dir = cont_dir

    # check node and change path
    node = platform.node()
    print("Working on " + str(node))

    if args.mode == "train_has":
        if args.bootstrap:
            from src.utils.bootstrap import BootstrapConfidenceInterval
            bootstrapper = BootstrapConfidenceInterval(args)
            bootstrapper.calculate_ci()
            sys.exit()

        from src.train_has import TrainerHASNTU
        trainer_has = TrainerHASNTU(args)
        if args.random_search:
            trainer_has.random_search()
        else:
            trainer_has.train_controller()
    else:
        logging.error("Mode: [{}] not known!".format(args.mode))
        sys.exit()


if __name__ == '__main__':
    main()
