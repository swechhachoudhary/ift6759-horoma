import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.scoring_function as scoreF
import utils.evaluator as evaluator
import utils.checkpoint_utils as checkpoint_utils

# Note : this code comes from OM Signal block 2 baseline
# Some changes were made to be able to use this for Horoma


class Trainer():

    def __init__(
            self, args, device, criterion_dict, target_labels, weight=None,
            target_vat_dict=None, target_entropy_dict=None, init_iter=0):
        self.args = args
        self.init_iter = init_iter
        self.device = device
        self.target_labels = target_labels
        self.criterion_dict = criterion_dict
        self.target_entropy_dict = target_entropy_dict
        self.target_vat_dict = target_vat_dict
        self.criterion = [
            self.criterion_dict[a] for a in self.target_labels
        ]
        self.entropy = [
            None if self.target_entropy_dict is None else self.target_entropy_dict[a]
            for a in self.target_labels
        ]
        self.vat_type = [
            None if self.target_vat_dict is None else self.target_vat_dict[a]
            for a in self.target_labels
        ]
        self.is_entropy_based = False
        for a in self.entropy:
            if a is not None:
                self.is_entropy_based = True
                break
        self.weight = weight
        if self.weight is None:
            self.weight = [1.0] * len(self.criterion)
        assert len(self.weight) == len(self.criterion)

    def train(self, model, data_iterators, optimizer, tb_prefix='exp/', prefix='neural_network', experiment=None):
        sup_losses = [checkpoint_utils.AverageMeter() for _ in range(len(self.criterion) + 1)]
        vat_losses = checkpoint_utils.AverageMeter()
        perfs = [checkpoint_utils.AverageMeter() for _ in range(3)]

        model.train()

        criterion = self.criterion
        score_param_index = evaluator.get_scoring_func_param_index(
            self.target_labels)
        weight = self.weight
        if weight is None:
            weight = [1.0] * len(criterion)
        assert len(weight) == len(criterion)

        tbIndex = 0

        best_val_metric = 0.0
        best_val_data = None

        for k in tqdm(range(self.init_iter, self.args.iters)):

            # reset
            if k > 0 and k % self.args.log_interval == 0:
                tbIndex += 1
                val_mean_loss, val_metrics, _ = self.eval(
                    model,
                    data_iterators,
                    key='val'
                )

                if val_metrics[0] > best_val_metric:
                    best_val_metric = val_metrics[0]
                    best_val_data = val_metrics
                    filename = self.args.checkpoint_dir + prefix + '_{}.pt'.format('BestModel')
                    checkpoint_utils.set_path(filename)
                    checkpoint_utils.save_checkpoint(model, k, filename, optimizer)

                experiment.log_metric('Train/Loss', sup_losses[0].avg, tbIndex)
                experiment.log_metric('Valid/Loss', val_mean_loss, tbIndex)
                experiment.log_metric('Train/treeClassAcc', perfs[1].avg, tbIndex)
                experiment.log_metric('Valid/treeClassAcc', val_metrics[1], tbIndex)
                experiment.log_metric('Train/treeClassF1', perfs[2].avg, tbIndex)
                experiment.log_metric('Valid/treeClassF1', val_metrics[2], tbIndex)

                train_metrics_avg = [p.avg for p in perfs]
                train_metrics_val = [p.val for p in perfs]

                print(
                    'Iteration: {}\t Loss {:.4f} ({:.4f})\t'.format(
                        k, sup_losses[0].val, sup_losses[0].avg
                    ),
                    'Train_Metrics: {}\t Train_Metrics_AVG {}\t'.format(
                        train_metrics_val, train_metrics_avg
                    ),
                    'Valid_Metrics: {}\t'.format(val_metrics),
                    'Best Perf: {} - {}\t'.format(best_val_metric, best_val_data)
                )
                for a in sup_losses:
                    a.reset()
                for a in perfs:
                    a.reset()
                vat_losses.reset()

                # re-activate train mode
                model.train()

            x_l, y_l = next(data_iterators['labeled'])
            if not isinstance(y_l, (list, tuple)):
                y_l = [y_l]
            x_ul = next(data_iterators['unlabeled'])

            x_l, y_l = x_l.to(self.device), [t.to(self.device) for t in y_l]
            if not isinstance(x_ul, (list, tuple)):
                x_ul = x_ul.to(self.device)
            else:
                x_ul = [t.to(self.device) for t in x_ul]

            optimizer.zero_grad()

            if isinstance(x_ul, (list, tuple)):
                x_ul = x_ul[0]

            x_l = x_l.reshape([x_l.shape[0], 1, 3072])
            outputs = model(x_l)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            reg_loss = 0.0
            if self.is_entropy_based:
                x_ul = x_ul.reshape([16, 1, 3072])
                outputs_ul = model(x_ul)
                if not isinstance(outputs_ul, (list, tuple)):
                    outputs_ul = [outputs_ul]
                supervised_reg_losses = [
                    w * (0.0 if c is None else c(o))
                    for c, o, w in zip(self.entropy, outputs, self.weight)
                ]
                unsupervised_reg_losses = [
                    w * (0.0 if c is None else c(o))
                    for c, o, w in zip(self.entropy, outputs_ul, self.weight)
                ]

                reg_losses = [
                    ((a / (x_l.size(0))) + self.args.alpha * (b / (x_ul.size(0)))) / (1.0 + self.args.alpha)
                    for a, b in zip(supervised_reg_losses, unsupervised_reg_losses)
                ]

                reg_loss = sum(reg_losses)

            y_l[0] = y_l[0].long()
            supervised_losses = [
                w * (c(o, gt) if o.size(1) == 1 else c(o, gt.squeeze(1)))
                for c, o, gt, w in zip(criterion, outputs, y_l, weight)
            ]
            supervised_loss = sum(supervised_losses)

            treeClass_pred, treeClass_true = None, None
            if score_param_index[0] is not None:
                i = score_param_index[0]
                _, pred_classes = torch.max(outputs[i], dim=1)
                treeClass_true = y_l[i].view(-1).tolist()
                treeClass_pred = pred_classes.view(-1).tolist()
                treeClass_pred = np.array(treeClass_pred, dtype=np.int32)
                treeClass_true = np.array(treeClass_true, dtype=np.int32)

            loss = supervised_loss + reg_loss + self.args.alpha

            loss.backward()
            optimizer.step()

            metrics = scoreF.scorePerformance(
                treeClass_pred, treeClass_true
            )

            for i in range(len(supervised_losses)):
                sup_losses[i + 1].update(
                    supervised_losses[i].item(),
                    x_l.shape[0]
                )
            sup_losses[0].update(
                supervised_loss.item(),
                x_l.shape[0]
            )
            for i in range(len(metrics)):
                perfs[i].update(
                    metrics[i],
                    x_l.shape[0]
                )

            if k > 0 and k % self.args.chkpt_freq == 0:
                filename = self.args.checkpoint_dir + \
                    prefix + '_{}.pt'.format(k)
                checkpoint_utils.set_path(filename)
                checkpoint_utils.save_checkpoint(model, k, filename, optimizer)

        filename = self.args.checkpoint_dir + \
            prefix + '_{}.pt'.format(self.args.iters)
        checkpoint_utils.set_path(filename)
        checkpoint_utils.save_checkpoint(model, self.args.iters, filename, optimizer)

    def eval(self, model, data_iterators, key='val'):
        assert key in ('val', 'test')
        assert not (data_iterators[key] is None)
        criterion = self.criterion
        weight = self.weight
        device = self.device

        return evaluator.evaluate(model, device, data_iterators[key], self.target_labels, criterion, weight)
