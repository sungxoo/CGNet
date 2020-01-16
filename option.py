import argparse


class Option(object):
    def __init__(self):
        super().__init__()
        self._parser = argparse.ArgumentParser()

        # basic options
        self._parser.add_argument('--num_workers', type=int, default=2, help='The workers number')
        self._parser.add_argument('--device', type=str, default='cuda', help='Select cpu or cuda')
        self._parser.add_argument('--logdir', type=str, default='./logs')
        self._parser.add_argument('--resume', type=str, default='None', help='''
                                    1. None: starting from first epoch
                                    2. default: starting from the latest epoch
                                    3. /path/to/some.pth: starting from the some.pth ''')

        self._parser.add_argument('--dataset', type=str, default='/home/sungsoo/sskim/DB/cityscapes', help='Dataset path')
        # training options
        self._parser.add_argument('--train_batch', type=int, default=6, help='Training batch size')
        self._parser.add_argument('--train_size', nargs='+', type=int, default=[768, 768], help='Training image size')
        self._parser.add_argument('--base_weight', type=str, default=None, help='Loading a pre-trained weight or None')
        self._parser.add_argument('--start_epoch', type=int, default=0, help='Training start point')
        self._parser.add_argument('--max_epoch', type=int, default=300, help='Maximum training epochs')
        self._parser.add_argument('--learning_rate', type=float, default=1e-3, help='Training learning rates')
        self._parser.add_argument('--save_interval', type=int, default=10, help='Epoch size for saving checkpoint')
        self._parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for parameter updates')
        self._parser.add_argument('--wegith_decay', type=float, default=2e-4, help='L2 regularization parameters')

        # validation options
        self._parser.add_argument('--val_batch', type=int, default=2, help='The number of batch sizes for validation')
        self._parser.add_argument('--val_size', nargs='+', type=int, default=[2048, 1024],
                                  help='Validation image size [width, height]')

        # lr_scheduler.StepLR options
        self._parser.add_argument('--step_size', type=int, default=100, help='Steps for updating learning rates')
        self._parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler learning rate decay factor')

        # PolyLR
        self._parser.add_argument("--poly_lr", action='store_true', default=False,
                                  help='Turning on polynomial scheduler (default: step scheduler) for learning rate')
        self._parser.add_argument('--power', type=float, default=0.9, help='Polynomial power for learning rate decay')

        # train dataset augmentations
        self._parser.add_argument('--scale_limits', nargs='+', type=float, default=[0.5, 2.0],
                                  help='Image rescale lower bound')
        self._parser.add_argument('--scale_step', type=float, default=0.25, help='Image rescale interval')

        self._opt = self._parser.parse_args()

    def parse(self):
        print('--------------- Options ---------------')
        for k, v in sorted(vars(self._opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------- End -----------------')
        return self._opt