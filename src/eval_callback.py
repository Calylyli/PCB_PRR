import os
from mindspore import save_checkpoint
from mindspore.train.callback import Callback
from src.model_utils.config import config

class EvalCallBack(Callback):


    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", best_ckpt_name="best.ckpt", metrics_name=("mAP", "CMC"), cmc_topk=(1, 5, 10)):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_mAP = 0
        self.best_cmc_scores = None
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.best_ckpt_path = os.path.join(ckpt_directory, best_ckpt_name)
        self.metrics_name = metrics_name
        self.cmc_topk = cmc_topk

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            mAP, cmc_scores = self.eval_function(self.eval_param_dict)
            print('Mean AP: {:4.1%}'.format(mAP), flush=True)
            print('CMC Scores{:>12}'.format(config.dataset_name), flush=True)
            for k in self.cmc_topk:
                print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[config.dataset_name][k - 1]), flush=True)
            if mAP >= self.best_mAP:
                self.best_mAP = mAP
                self.best_cmc_scores = cmc_scores
                self.best_epoch = cur_epoch
                print("update best mAP: {}".format(mAP), flush=True)
                if self.save_best_ckpt:
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)
                    print("update best checkpoint at: {}".format(self.best_ckpt_path), flush=True)

    def end(self, run_context):
        print("End training, the best epoch is {}".format(self.best_epoch), flush=True)
        print("Best result:", flush=True)
        print('Mean AP: {:4.1%}'.format(self.best_mAP), flush=True)
        print('CMC Scores{:>12}'.format(config.dataset_name), flush=True)
        for k in self.cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, self.best_cmc_scores[config.dataset_name][k - 1]), flush=True)
