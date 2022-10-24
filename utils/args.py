import argparse

DATASETS = ["sent140", "femnist", "shakespeare", "celeba", "synthetic", "reddit"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset",
                        help="name of dataset;",
                        type=str,
                        choices=DATASETS,
                        default="femnist")
    parser.add_argument("-model",
                        help="name of model;",
                        type=str,
                        default="cnn")
    parser.add_argument("--num-rounds",
                        help="number of rounds to simulate;",
                        type=int,
                        default=500)
    parser.add_argument("--eval-every",
                        help="evaluate every _ rounds;",
                        type=int,
                        default=1)
    parser.add_argument("--num-groups",
                        help="number of groups;",
                        type=int,
                        default=10)
    parser.add_argument("--clients-per-group",
                        help="number of clients trained per group;",
                        type=int,
                        default=10)
    parser.add_argument("--batch-size",
                        help="batch size when clients train on data;",
                        type=int,
                        default=5)
    parser.add_argument("--num-epochs",
                        help="number of epochs when clients train on data;",
                        type=int,
                        default=1)
    parser.add_argument("-lr",
                        help="learning rate for local optimizers;",
                        type=float,
                        default=0.01)
    parser.add_argument("--seed",
                        help="seed for random client sampling and batch splitting;",
                        type=int,
                        default=2)
    parser.add_argument("--metrics-name",
                        help="name for metrics file;",
                        type=str,
                        default="metrics")
    parser.add_argument("--metrics-dir",
                        help="dir for metrics file;",
                        type=str,
                        default="metrics")
    parser.add_argument("--log-dir",
                        help="dir for log file;",
                        type=str,
                        default="logs")
    parser.add_argument("--log-rank",
                        help="suffix identifier for log file;",
                        type=int,
                        default=0)
    parser.add_argument("--use-val-set",
                        help="use validation set;",
                        action="store_true")
    parser.add_argument("-ctx",
                        help="device for training, -1 for cpu and 0~3 for gpu;",
                        type=int,
                        default=1)
    parser.add_argument("--a",
                        help="The value of a",
                        type=float,
                        default=0.072)
    parser.add_argument("--beta",
                        help="The value of beta", #文章中beta的值
                        type=int,
                        default=10)
    parser.add_argument('--function', type=str, default='linear', help='linear,exp,log')
    parser.add_argument('--grouper', type=str, default='icg', help='icg,random')
    parser.add_argument('--fr', type=float, default=0.3, help="Sampling Rate")
    parser.add_argument("--frequency",
                        help="frequency of full model update;",
                        type=int,
                        default=3)

    parser.add_argument('--mode', type=str, default='FedGSP', help='FedGSP,NaiveGSP,NaiveICG')


    return parser.parse_args()
