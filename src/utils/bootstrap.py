import logging
import torch
import thop
import gc
import collections

from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from src.initializer import Initializer
from src.model.student import Student
from src.utils import utils


class BootstrapRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=True, num_samples=None, generator=None):
        super().__init__(data_source, replacement, num_samples, generator)

    def __iter__(self):
        n = len(self.data_source)
        for _ in range(self.num_samples):
            yield np.random.randint(0, n)


class BootstrapConfidenceInterval(Initializer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.student_model = None
        self.best_state = None
        self.argmax_epochs = self.args.argmax_epochs

        self.model_params = self.args.bootstrap_arch
        self.hp_params = self.args.bootstrap_hyper

        self.eval_interval = self.args.eval_interval

        # build student
        self._build_student()

        # split test set into bootstrap samples
        self.alpha = self.args.bootstrap_alpha

    def calculate_ci(self):
        """
        Calculate the confidence interval.
        """
        # number of bootstrap samples --> should be the same as the data
        num_bootstrap_samples = self.feeder_val.__len__()
        logging.info("# of bootstrap samples for dataset is: {}".format(num_bootstrap_samples))
        # number of bootstrap iterations
        num_bootstrap_iteration = self.args.bootstrap_iter

        # custom random sampler
        random_sampler = BootstrapRandomSampler(self.feeder_val, num_samples=num_bootstrap_samples)

        # DataLoader for the test data using the random sampler
        boot_loader = DataLoader(self.feeder_val, batch_size=self.new_batch_size,
                                     num_workers=self.args.num_workers, pin_memory=self.args.pin_memory,
                                     shuffle=False, drop_last=True, sampler=random_sampler)

        # for num, (x, y, _) in enumerate(boot_loader):
        #     print(f"Bootstrap Sample {num + 1}: {y.tolist()}")

        if self.args.ddp:
            self.student_model.to(self.gpu_id)
            self.student_model = DDP(self.student_model, device_ids=[self.gpu_id])
            self.train_loader.sampler.set_epoch(0)
        elif self.gpu_id:
            self.student_model.to(self.gpu_id)

        accuracies = []
        self.student_model.eval()
        start_eval_time = time()

        for i in range(num_bootstrap_iteration):
            logging.info("Bootstrap Iteration: {}/{}".format(i, num_bootstrap_iteration))
            with torch.no_grad():
                num_top1, num_top5 = 0, 0
                num_sample, eval_loss = 0, []
                # cm = np.zeros((self.num_classes, self.num_classes))
                eval_iter = boot_loader if self.no_progress_bar else \
                    tqdm(boot_loader, leave=True, desc="Eval student {}".format(0))

                for num, (x, y, _) in enumerate(eval_iter):
                    # Using GPU
                    x = x.to(self.gpu_id)
                    y = y.to(self.gpu_id)

                    # Calculating Output
                    out, _ = self.student_model(x)

                    # Getting Loss
                    loss = self.loss_func(out, y)
                    eval_loss.append(loss.item())

                    # Calculating Recognition Accuracies
                    num_sample += x.size(0)
                    reco_top1 = out.max(1)[1]
                    num_top1 += reco_top1.eq(y).sum().item()
                    reco_top5 = torch.topk(out, 5)[1]
                    num_top5 += sum([y[n] in reco_top5[n, :] for n in range(x.size(0))])

                    # Calculating Confusion Matrix
                    # for i in range(x.size(0)):
                    #     cm[y[i], reco_top1[i]] += 1

                    # Showing Progress
                    if self.no_progress_bar:
                        logging.info('Batch: {}/{}'.format(num + 1, len(boot_loader)))

                    acc_top1 = round(num_top1 / num_sample, 3)
                    accuracies.append(acc_top1)

            # Showing Evaluating Results
            acc_top1 = round(num_top1 / num_sample, 3)
            acc_top5 = round(num_top5 / num_sample, 3)
            eval_loss = np.mean(eval_loss)
            eval_time = time() - start_eval_time
            eval_speed = len(self.val_loader) * self.new_batch_size / eval_time / len(self.args.gpus)
            logging.info(
                'Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
                    num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
                ))
            logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
                eval_time, eval_speed
            ))
            logging.info('')

        torch.cuda.empty_cache()

        confidence_level = 100 - self.alpha

        lower_percentile = self.alpha / 2.0
        upper_percentile = (100 - self.alpha) + (self.alpha / 2.0)
        lower_bound = max(0.0, np.percentile(accuracies, lower_percentile))
        upper_bound = min(1.0, np.percentile(accuracies, upper_percentile))
        logging.info("Confidence Interval ({}%): [{}, {}] with {} bootstraps and {} iterations.".format(
            confidence_level,
            round(lower_bound, 3),
            round(upper_bound, 3),
            num_bootstrap_samples,
            num_bootstrap_iteration))

        logging.info('50th percentile (median) = %.3f' % np.median(accuracies))

        with open('{}/accuracies.txt'.format(self.save_dir), 'w') as f:
            for item in accuracies:
                f.write(str(item) + '\n')
        logging.info("Done!")

    def _build_student(self, student_id=0):
        """
        Build the student for the bootloader.
        May have to retrain the model due to old bug...
        @param student_id:
        @return:
        """
        # generate student from sampled actions
        # get indexes
        actions_arch = [int(i) for i in self.model_params]
        actions_hyper = [int(i) for i in self.hp_params]

        # convert hyper parameters
        hyper_action_list = dict.fromkeys(self.hyper_choices)
        computations = list(self.hyper_choices.keys())
        for comp, action in zip(computations, actions_hyper):
            hyper_action_list[comp] = self.hyper_choices[comp][action]

        self.student_model = Student(actions_arch, self.arch_choices, student_id, self.args, load=False, **self.kwargs)

        logging.info("Student AP: {}".format(self.student_model.action_list))
        logging.info("Student HP: {}".format(hyper_action_list))
        flops, params = thop.profile(deepcopy(self.student_model), inputs=torch.rand([1, 1] + self.data_shape),
                                     verbose=False)
        logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))

        # update hyperparameters for sampled student
        optimizer_fn = None
        for optimizer_class in self.optim_list:
            if optimizer_class.__name__.lower() == hyper_action_list['optimizers'].lower():
                optimizer_fn = optimizer_class
                break

        assert optimizer_fn is not None

        # fixed optimizer args from config
        optimizer_args = self.args.optimizer_args[hyper_action_list['optimizers']]
        # sampled optimizer args
        optimizer_args['lr'] = hyper_action_list['lr']
        optimizer_args['weight_decay'] = hyper_action_list['weight_decay']

        if optimizer_fn.__name__.lower() not in ['adam', 'adamw']:
            optimizer_args['momentum'] = hyper_action_list['momentum']

        self.new_batch_size = int(hyper_action_list['batch_size'])
        self.update_batch_size(self.new_batch_size)

        self.optimizer = optimizer_fn(params=self.student_model.parameters(), **optimizer_args)

        if self.args.lr_scheduler:
            # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
            self.scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                                      start_factor=self.args.sched_args.start_factor,
                                                                      total_iters=self.args.sched_args.warm_up)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=self.args.sched_args.step_lr,
                                                                  gamma=self.args.sched_args.gamma)
        else:
            self.scheduler = None
        logging.info('LR_Scheduler: {}'.format(self.scheduler.__class__.__name__))

        # load state dict
        try:
            checkpoint = torch.load(self.args.bootstrap_path)  # , map_location=torch.device('cpu'))
        except:
            logging.info('')
            logging.error('Error: loading checkpoint: {}!'.format(self.args.bootstrap_path))
            raise ValueError()

        model_state = ['input_stream', 'main_stream', 'classifier']

        # Check if each dictionary exists in 'model' - bug in old commit
        retrain = any(stream not in checkpoint['model'] for stream in model_state)

        # check if config is the same
        # assert self.student_model.action_list == checkpoint['actions'], "Search Space values are not the same!"

        if self.args.bootstrap_retrain:
            retrain = True

        if not retrain:
            logging.info("Not retraining model but loading states...")
            combined_state_dict = collections.OrderedDict()
            # Copy contents of sub_dicts into combined_state_dict
            combined_state_dict.update(checkpoint['model']['input_stream'])
            combined_state_dict.update(checkpoint['model']['main_stream'])
            combined_state_dict.update(checkpoint['model']['classifier'])

            self.student_model.load_state_dict(combined_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            self.epochs = checkpoint['epoch']
            self.best_state = checkpoint['best_state']
            logging.info("Loaded model with actions: {}".format(checkpoint['actions']))
            logging.info('Best accuracy Top1: {:.2%}'.format(self.best_state['acc_top1']))
            logging.info('Successful!')
        else:
            logging.info("Retraining model!")
            argmax_time = 0
            argmax_id = 9999
            best_state_argmax = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0}
            for epoch in range(self.argmax_epochs):
                epoch_time, train_acc, train_loss = self._train_student(epoch, self.argmax_epochs, argmax_id)
                argmax_time += epoch_time
                is_best = False
                if (epoch + 1) % self.eval_interval == 0 or epoch >= self.argmax_epochs - 15:
                    # and (self.gpu_id == 0 if self.args.ddp else True):
                    # (self.gpu_id == 0 if self.args.ddp):
                    logging.info("Evaluating ARGMAX student quality in epoch {}/{}"
                                 .format(epoch + 1, self.argmax_epochs))
                    acc_top1, acc_top5, cm = self._eval_student(epoch, argmax_id)

                    if acc_top1 > best_state_argmax['acc_top1']:
                        is_best = True
                        best_state_argmax.update({'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm})

                # save model for later

                utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                      self.scheduler.state_dict(), epoch + 1, epoch_time,
                                      self.student_model.action_list, best_state_argmax,
                                      is_best, self.save_dir, argmax_id, argmax=True)

            logging.info("Model {}: Top1 {}, Top5 {}, Training time: {}".format(
                argmax_id, best_state_argmax['acc_top1'], best_state_argmax['acc_top5'], argmax_time))

            logging.info("Done with retraining...")

    def _train_student(self, epoch: int, max_epoch: int, student_id: int):
        """
        Train student network.
        :param epoch: num epochs
        :param max_epoch: max epoch
        :return:
        """
        if self.args.ddp:
            self.student_model.to(self.gpu_id)
            self.student_model = DDP(self.student_model, device_ids=[self.gpu_id])
            self.train_loader.sampler.set_epoch(epoch)
        elif self.gpu_id:
            self.student_model.to(self.gpu_id)

        self.student_model.train()

        start_train_time = time()
        train_loss_list = []
        num_top1, num_sample = 0, 0

        train_iter = self.train_loader if self.no_progress_bar else \
            tqdm(self.train_loader, leave=True, desc="Train student {}".format(student_id))
        for num, (x, y, _) in enumerate(train_iter):

            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

            # put data through student
            out, _ = self.student_model(x)
            loss = self.loss_func(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # if self.scheduler is not None:
            #     self.scheduler.step()

            # Calculating accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()

            # Progress
            lr = self.optimizer.param_groups[0]['lr']
            # if self.writer:
            #     self.writer.add_scalar('learning_rate/student_{}'.format(self.student_model.student_id), lr, num)
            #     self.writer.add_scalar('train_loss', loss.item(), num)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch + 1, max_epoch, num + 1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

            train_loss_list.append(loss.item())

            # Zero out gradients and optimizer states
            # student.zero_grad()
            # optimizer.zero_grad()
            # self.optimizer.state.clear()
        if (epoch + 1) < self.args.sched_args.warm_up and (self.scheduler and self.scheduler_warmup) is not None:
            self.scheduler_warmup.step()
            # logging.info("Current lr: {}".format(self.scheduler_warmup.get_last_lr()))
        else:
            self.scheduler.step(epoch + 1)
            # logging.info("Current lr: {}".format(self.scheduler.get_last_lr()))

        if self.args.ddp:
            logging.info("GPU {} done.".format(self.gpu_id))

        # Train Results
        train_acc = round(num_top1 / num_sample, 3)
        epoch_time = time() - start_train_time
        train_loss = np.mean(train_loss_list)
        if self.writer:
            self.writer.add_scalar('train_acc/student_{}'.format(self.student_model.student_id),
                                   train_acc, epoch + 1)
            self.writer.add_scalar('train_loss/student_{}'.format(self.student_model.student_id),
                                   train_loss, epoch + 1)
            self.writer.add_scalar('train_time/student_{}'.format(self.student_model.student_id),
                                   epoch_time, epoch + 1)

        logging.info(
            'Epoch: {}/{}, Train accuracy: {:d}/{:d}({:.2%}), Train time: {:.2f}s, Mean loss:{:.4f}, lr:{:.4f}'.format(
                epoch + 1, max_epoch, num_top1, num_sample, train_acc, epoch_time, train_loss, lr
            ))
        logging.info('')

        gc.collect()
        torch.cuda.empty_cache()
        return epoch_time, train_acc, train_loss

    def _eval_student(self, epoch: int, student_id: int):
        """
        Eval the student network when it needs to be retrained.
        :param epoch:
        :return:
        """
        self.student_model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_classes, self.num_classes))
            eval_iter = self.val_loader if self.no_progress_bar else \
                tqdm(self.val_loader, leave=True, desc="Eval student {}".format(student_id))
            for num, (x, y, _) in enumerate(eval_iter):

                # Using GPU
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)

                # Calculating Output
                out, _ = self.student_model(x)

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out, 5)[1]
                num_top5 += sum([y[n] in reco_top5[n, :] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar:
                    logging.info('Batch: {}/{}'.format(num + 1, len(self.val_loader)))

        # Showing Evaluating Results
        acc_top1 = round(num_top1 / num_sample, 3)
        acc_top5 = round(num_top5 / num_sample, 3)
        eval_loss = np.mean(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.val_loader) * self.new_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.writer:
            self.writer.add_scalar('eval_acc/student_{}'.format(student_id),
                                   acc_top1, epoch + 1)
            self.writer.add_scalar('eval_loss/student_{}'.format(student_id),
                                   eval_loss, epoch + 1)

        torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm
