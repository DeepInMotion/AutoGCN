import copy
import csv
import gc
import json
import logging
import os
import sys

import thop
from copy import deepcopy
from time import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.initializer import Initializer
from src.model.student import Student
from src.utils import utils
from src.utils.utils import import_class


class TrainerHASNTU(Initializer):
    """
    Entry point for AutoHAS NTU
    Code adapted from https://github.com/jacknewsom/autohas-pytorch
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.student_model = None

        self.train_epochs = args.train_epochs
        self.argmax_epochs = args.argmax_epochs
        self.early_stop = args.early_stop
        self.early_stop_epoch = args.early_stop_epoch
        self.early_stop_acc = args.early_stop_acc
        self.eval_interval = args.eval_interval
        self.early_stop_no_impr = args.early_stop_no_impr
        self.warmup_epochs = args.warmup_epochs
        # number of rollouts in first iteration
        self.warmup_rollouts = args.warmup_rollouts
        # number of rollouts after
        self.num_rollouts_iter = args.rollouts
        # number of updates before saving controller policies
        self.save_policy_frequency = 1

        if args.random_search:
            # staying random
            self.random_iter = args.random_iter
            self.random_epochs_half = args.random_epochs_half
            self.random_best_arch = None
            self.random_best_hyper = None
            self.random_search()

    def random_search(self):
        random_id = 0
        random_argmax_id = 1000
        random_actions_save = []

        logging.info("Starting Random Search for {} iterations...".format(self.random_iter))
        for i in range(self.random_iter):
            logging.info("Iteration random {}/{}".format(i, self.random_iter))
            random_actions = []
            for r in range(self.num_rollouts_iter):
                logging.info("Rollout random {}/{}".format(r, self.num_rollouts_iter))
                # sample random students
                model_params, hp_params, actions, actions_hp = self.sample_student(random_id, argmax=False)
                oom_bool = False
                epoch_no_improve = 0
                last_train_acc = 0.0
                student_time = 0
                best_state = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0}
                logging.info("Training random student {} for {} epochs".format(random_id, self.random_epochs_half))
                try:
                    for epoch in range(self.random_epochs_half):
                        # student training loop
                        epoch_time, train_acc, train_loss = self.train_student(epoch, self.random_epochs_half, random_id)
                        student_time += epoch_time

                        is_best = False
                        if (epoch + 1) % self.eval_interval == 0 or epoch == self.random_epochs_half - 1:
                            # and (self.gpu_id == 0 if self.args.ddp else True):
                            logging.info('Evaluating for epoch {}/{} ...'.format(epoch + 1, self.random_epochs_half))
                            acc_top1, acc_top5, cm = self.eval_student(epoch, random_id)

                            if acc_top1 > best_state['acc_top1']:
                                is_best = True
                                best_state.update({'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm})

                        # save model for later
                        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                              self.scheduler.state_dict(), epoch + 1, epoch_time, 0, best_state,
                                              is_best, self.save_dir, random_id)
                        # early stop check
                        if train_acc < last_train_acc:
                            epoch_no_improve += 1
                        else:
                            epoch_no_improve = 0

                        if self.early_stop and (epoch + 1) >= self.early_stop_epoch:
                            if best_state['acc_top1'] < self.early_stop_acc or \
                                    epoch_no_improve > self.early_stop_no_impr:
                                logging.info("Student {} EARLY STOP after {} epochs".format(random_id, epoch))
                                break

                except torch.cuda.OutOfMemoryError:
                    logging.info("Random ID {} does NOT fit on GPU!".format(random_id))
                    torch.cuda.empty_cache()
                    oom_bool = True

                if not oom_bool:
                    # do not save oom architectures
                    random_actions.append([random_id, actions, actions_hp, best_state])

                # free gpu
                del model_params, hp_params
                del self.student_model, self.optimizer
                torch.cuda.empty_cache()
                gc.collect()
                random_id += 1

            max_value = float('-inf')  # Initialize with negative infinity to find the maximum
            max_index = None

            for i, item in enumerate(random_actions):
                best_state_value = item[3].get('acc_top1', float('-inf'))
                if best_state_value > max_value:
                    max_value = best_state_value
                    max_index = i
            self.random_best_arch = random_actions[max_index][1]
            self.random_best_hyper = random_actions[max_index][2]

            # Train the best student for 75 epochs
            _, _, actions_max, hp_max = self.sample_student(random_argmax_id, argmax=False, random=True)
            argmax_time = 0
            best_state_argmax = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0}
            logging.info("Random HALF-ARGMAX student had a Top-1 of: {}".format(max_value))
            logging.info("RANDOM ARGMAX student AP: {}".format(self.random_best_arch))
            logging.info("RANDOM ARGMAX student HP: {}".format(self.random_best_hyper))
            logging.info("Training RANDOM ARGMAX student...")
            logging.info("Training RANDOM ARGMAX student {} for {} epochs".format(random_argmax_id, self.argmax_epochs))
            try:
                for epoch in range(self.argmax_epochs):
                    epoch_time, train_acc, train_loss = self.train_student(epoch, self.argmax_epochs, random_argmax_id)
                    argmax_time += epoch_time
                    is_best = False
                    if (epoch + 1) % self.eval_interval == 0 or epoch >= self.argmax_epochs - 5:
                        # and (self.gpu_id == 0 if self.args.ddp else True):
                        # (self.gpu_id == 0 if self.args.ddp):
                        logging.info("Evaluating RANDOM ARGMAX student quality in epoch {}/{}"
                                     .format(epoch + 1, self.argmax_epochs))
                        acc_top1, acc_top5, cm = self.eval_student(epoch, random_argmax_id)

                        if acc_top1 > best_state_argmax['acc_top1']:
                            is_best = True
                            best_state_argmax.update({'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm})

                    # save model for later

                    utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                          self.scheduler.state_dict(), epoch + 1, epoch_time,
                                          self.student_model.action_list, best_state_argmax,
                                          is_best, self.save_dir, random_argmax_id, argmax=True)

            except torch.cuda.OutOfMemoryError:
                logging.info("RANDOM ARGMAX student {} does not fit on GPU!".format(random_argmax_id))
                torch.cuda.empty_cache()

            if self.args.ddp:
                destroy_process_group()

            logging.info("RANDOM ARGMAX student {}: Top1 {}, Top5 {}, Training time: {}".format(
                random_argmax_id, best_state_argmax['acc_top1'], best_state_argmax['acc_top5'], argmax_time))

            random_actions.append([random_argmax_id, actions_max, hp_max, best_state_argmax])

            if self.writer:
                self.writer.add_scalar('Argmax_accuracy/student_{}'.format(random_argmax_id),
                                       best_state_argmax['acc_top1'])
                self.writer.add_scalar('Argmax_time/student_{}'.format(random_argmax_id), argmax_time)

            del self.student_model
            torch.cuda.empty_cache()
            random_argmax_id += 1
            # random_actions_save = random_actions
            random_actions_save.extend(random_actions)

            # save action list
            with open('{}/action_list_rs.csv'.format(self.save_dir), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(random_actions_save)

        logging.info("Random Search done!")
        sys.exit()

    def train_controller(self):

        if self.args.cont_training:
            # get iteration from replay memory
            student_id = self.controller.replay_memory.__len__()
            assert student_id > 0
            # load action list
            from src.utils.io import read_actions
            action_path = "{}/action_list.csv".format(self.args.cont_dir)
            action_list = read_actions(action_path)
            argmax_id, argmax_index = max((i[0], index) for index, i in enumerate(action_list))
            argmax_id += 1
            # load rollouts
            from src.utils.io import read_rollouts
            rollouts_path = "{}/rollouts.csv".format(self.args.cont_dir)
            rollouts_save = read_rollouts(rollouts_path, self.gpu_id)
            logging.info("Loaded old actions and rollouts successfully.")

            iteration = argmax_id % 1000
            assert iteration < 2000, "You probably trained too long..."

            # check when argmax has to be trained
            if argmax_index + self.args.rollouts < len(action_list):
                train_argmax_bool = True
            else:
                train_argmax_bool = False
        else:
            iteration = 0
            student_id = 0
            argmax_id = 1000
            rollouts_save = []
            action_list = []
            train_argmax_bool = False

        logging.info("Training NTU controller...")
        while not self.controller.has_converged():
            rollouts = []
            logging.info("Iteration {}".format(iteration))
            # warmup check
            if iteration == 0:
                rollout_num = self.warmup_rollouts
            else:
                rollout_num = self.num_rollouts_iter

            if train_argmax_bool:
                # skip training student
                rollout_num = 0
                train_argmax_bool = False
                logging.info("Due to many trained students now updating controller!")

            for t in range(rollout_num):
                logging.info("Rollout {}/{}".format(t, rollout_num))
                logging.info("Loading student ID: {}".format(student_id))

                # sample student
                model_params, hp_params, actions, actions_hp = self.sample_student(student_id, argmax=False)

                oom_bool = False
                epoch_no_improve = 0
                last_train_acc = 0.0
                student_time = 0
                best_state = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0}
                logging.info("Training student {} for {} epochs".format(student_id, self.train_epochs))
                try:
                    for epoch in range(self.train_epochs):
                        # student training loop
                        epoch_time, train_acc, train_loss = self.train_student(epoch, self.train_epochs, student_id)
                        student_time += epoch_time

                        is_best = False
                        if (epoch + 1) % self.eval_interval == 0 or epoch == self.train_epochs - 1:
                            # and (self.gpu_id == 0 if self.args.ddp else True):
                            logging.info('Evaluating for epoch {}/{} ...'.format(epoch + 1, self.train_epochs))
                            acc_top1, acc_top5, cm = self.eval_student(epoch, student_id)

                            if acc_top1 > best_state['acc_top1']:
                                is_best = True
                                best_state.update({'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm})

                        # save model for later
                        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                              self.scheduler.state_dict(), epoch + 1, epoch_time, 0, best_state,
                                              is_best, self.save_dir, student_id)
                        # early stop check
                        if train_acc < last_train_acc:
                            epoch_no_improve += 1
                        else:
                            epoch_no_improve = 0

                        if self.early_stop and (epoch + 1) >= self.early_stop_epoch:
                            if best_state['acc_top1'] < self.early_stop_acc or \
                                    epoch_no_improve > self.early_stop_no_impr:
                                logging.info("Student {} EARLY STOP after {} epochs".format(student_id, epoch+1))
                                break

                except torch.cuda.OutOfMemoryError:
                    logging.info("Student {} does NOT fit on GPU!".format(student_id))
                    torch.cuda.empty_cache()
                    oom_bool = True

                # save this stuff
                model_params_copy, hp_params_copy = copy.deepcopy(model_params), copy.deepcopy(hp_params)
                # for saving detach
                model_params = [x.cpu() for x in model_params]
                hp_params = [x.cpu() for x in hp_params]

                if not oom_bool:
                    # do not save oom architectures
                    rollouts.append([model_params_copy, hp_params_copy, best_state, student_id])
                    rollouts_save.append([model_params, hp_params, best_state, student_id])
                    action_list.append([student_id, actions, actions_hp, best_state])

                # save rollout_save list
                with open('{}/rollouts.csv'.format(self.save_dir), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(rollouts_save)

                # save action list
                with open('{}/action_list.csv'.format(self.save_dir), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(action_list)

                # free gpu
                del model_params, hp_params
                del self.student_model, self.optimizer
                torch.cuda.empty_cache()
                gc.collect()
                student_id += 1

            if 0 <= self.warmup_epochs <= iteration:
                logging.info("Updating controller...")
                self.controller.update(rollouts)

                # Determine validation accuracy of most likely student
                logging.info("Loading ARGMAX student...")
                model_params, hp_params, actions_max, hp_max = self.sample_student(argmax_id, argmax=True)

                # argmax student training loop
                argmax_time = 0
                best_state_argmax = {'acc_top1': 0, 'acc_top5': 0, 'cm': 0}
                logging.info("Training ARGMAX student...")
                logging.info("Training ARGMAX student {} for {} epochs".format(argmax_id, self.argmax_epochs))
                try:
                    for epoch in range(self.argmax_epochs):
                        epoch_time, train_acc, train_loss = self.train_student(epoch, self.argmax_epochs, argmax_id)
                        argmax_time += epoch_time
                        is_best = False
                        if (epoch + 1) % self.eval_interval == 0 or epoch >= self.argmax_epochs - 15:
                            # and (self.gpu_id == 0 if self.args.ddp else True):
                            # (self.gpu_id == 0 if self.args.ddp):
                            logging.info("Evaluating ARGMAX student quality in epoch {}/{}"
                                         .format(epoch + 1, self.argmax_epochs))
                            acc_top1, acc_top5, cm = self.eval_student(epoch, argmax_id)

                            if acc_top1 > best_state_argmax['acc_top1']:
                                is_best = True
                                best_state_argmax.update({'acc_top1': acc_top1, 'acc_top5': acc_top5, 'cm': cm})

                        # save model for later
                        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                              self.scheduler.state_dict(), epoch + 1, epoch_time,
                                              self.student_model.action_list, best_state_argmax,
                                              is_best, self.save_dir, argmax_id, argmax=True)

                except torch.cuda.OutOfMemoryError:
                    logging.info("ARGMAX student {} does not fit on GPU!".format(argmax_id))
                    torch.cuda.empty_cache()

                if self.args.ddp:
                    destroy_process_group()

                logging.info("ARGMAX student {}: Top1 {}, Top5 {}, Training time: {}".format(
                    argmax_id, best_state_argmax['acc_top1'], best_state_argmax['acc_top5'], argmax_time))

                action_list.append([argmax_id, actions_max, hp_max, best_state_argmax])

                # save action list
                with open('{}/action_list.csv'.format(self.save_dir), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(action_list)

                if self.writer:
                    self.writer.add_scalar('Argmax_accuracy/student_{}'.format(argmax_id),
                                           best_state_argmax['acc_top1'])
                    self.writer.add_scalar('Argmax_time/student_{}'.format(argmax_id), argmax_time)

                if best_state_argmax['acc_top1'] >= 0.97:
                    logging.info('Controller converged!')
                    self.controller.converged = True

                del self.student_model, model_params, hp_params
                torch.cuda.empty_cache()
                argmax_id += 1

            else:
                logging.info("NOT updating controller!")

            average_quality = np.round(np.mean([r[2]['acc_top1'] for r in rollouts]), 3)
            if self.writer:
                self.writer.add_scalar('Accuracy/val_avg', average_quality, iteration)
            logging.info("Average student quality over rollout is {}".format(average_quality))

            # save the controller stuff
            self.save_controller(iteration)

            if iteration >= self.args.max_iter:
                logging.info("Stopping after {} iterations".format(iteration))
                break

            # Update the policies
            if self.args.policy_deletion and (self.args.policy_updates <= iteration):
                logging.info("Deleting policies with threshold parameter: {}".format(self.args.policy_threshold))

                # find under performing policies -> get back indexes to delete them in dict
                hyper_space_update, arch_space_update = self.controller.delete_policies(self.arch_choices_copy,
                                                                                        self.hyper_choices_copy,
                                                                                        self.args.policy_threshold)

                self.hyper_choices = hyper_space_update
                self.hyper_computations = len(hyper_space_update)
                self.hyper_size = sum([len(x) for x in hyper_space_update.values()])

                self.arch_choices = arch_space_update
                self.arch_computations = len(arch_space_update)
                self.size_search = sum([len(x) for x in arch_space_update.values()])

                self.arch_names = []
                self.arch_values = []
                for items in self.arch_choices.items():
                    self.arch_names.append(items[0])
                    self.arch_values.append(items[1])

                self.hyper_names = []
                self.hyper_values = []
                for items in self.hyper_choices.items():
                    self.hyper_names.append(items[0])
                    self.hyper_values.append(items[1])

                logging.info("NEW Architecture Search Space is: {}".format(self.arch_choices))
                logging.info("NEW Hyperparameter Search Space is: {}".format(self.hyper_choices))
                logging.info("NEW Search Space size: {}".format(self.size_search + self.hyper_size))

            del rollouts
            iteration += 1

        # save final controller policy weights after convergence
        save_dir_conv = self.save_dir + '/controller_weights_converged/'
        self.controller.save_policies(save_dir_conv)
        logging.info("Converged after {} iterations!".format(iteration))

    def sample_student(self, student_id: int, argmax: bool, random=False):

        if argmax and not random:
            model_params, hp_params = self.controller.policy_argmax(self.arch_computations, self.hyper_computations)
        if not argmax and not random:
            model_params, hp_params = self.controller.sample(self.arch_computations, self.hyper_computations)

        if random:
            model_params = self.random_best_arch
            hp_params = self.random_best_hyper
            assert model_params and hp_params is not None

        # get indexes
        actions_arch = [int(i) for i in model_params]
        actions_hyper = [int(i) for i in hp_params]

        # convert hyper parameters
        hyper_action_list = dict.fromkeys(self.hyper_choices)
        computations = list(self.hyper_choices.keys())
        for comp, action in zip(computations, actions_hyper):
            hyper_action_list[comp] = self.hyper_choices[comp][action]

        # generate student from sampled actions
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

        return model_params, hp_params, actions_arch, actions_hyper

    def train_student(self, epoch: int, max_epoch: int, student_id: int):
        """
        Train student network.
        @param epoch:
        @param max_epoch:
        @param student_id:
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

        # TODO: DDP sync loss etc.
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

    def eval_student(self, epoch: int, student_id: int):
        """
        Eval the student network.
        @param epoch:
        @param student_id:
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

    def save_controller(self, iteration: int) -> None:
        """
        Save histograms of controller policies
        """
        for idx, p in enumerate(self.controller.policies['archspace']):
            params = self.controller.policies['archspace'][p].state_dict()['params']
            params /= torch.sum(params)

            if self.writer:
                curr_policy = self.arch_names[idx]
                save_dict = {}
                for i in range(len(params)):
                    param_value = self.controller.arch_space.get(curr_policy)[i]
                    dict_name = '{}_{}'.format(curr_policy, param_value)
                    save_dict[dict_name] = params[i]

                self.writer.add_scalars('/Parameters/Arch/{}'.format(curr_policy), save_dict, iteration)

        for idx, p in enumerate(self.controller.policies['hpspace']):
            params = self.controller.policies['hpspace'][p].state_dict()['params']
            params /= torch.sum(params)

            if self.writer:
                curr_policy = self.hyper_names[idx]
                save_dict = {}
                for i in range(len(params)):
                    param_value = self.controller.hyper_space.get(curr_policy)[i]
                    dict_name = '{}_{}'.format(curr_policy, param_value)
                    save_dict[dict_name] = params[i]

                self.writer.add_scalars('/Parameters/Hyper/{}'.format(curr_policy), save_dict, iteration)

        logging.info("Saving controller policies...")
        save_dir = self.save_dir + self.args.controller_dir
        self.controller.save_policies(save_dir)
