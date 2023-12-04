import logging
import os

import numpy as np
import torch
from torch.distributions import Categorical

from src.model.controller.replay_memory import ReplayMemory
from src.model.controller.base_controller import BaseController
from src.model.controller.policy import Policy


class ReinforceController(BaseController):
    """
    REINFORCE Controller
    Code adopted from: https://github.com/jacknewsom/autohas-pytorch
    """
    def __init__(self, archspace, hyperspace, device, save_dir, args, **kwargs):
        super(ReinforceController, self).__init__(**kwargs)

        self.args = args
        self.converged = False
        self.gpu_id = device
        self.reward_map_fn_str = 'lambda x: x'
        # compute model quality as `reward_map_fn`(validation accuracy)
        self.reward_map_fn = eval(self.reward_map_fn_str)

        # use average reward as baseline for rollouts
        self.use_baseline = self.args.use_baseline

        # track policies for architecture space, hyperparameter space
        self.policies = {'archspace': {}, 'hpspace': {}}

        self.arch_space = archspace
        self.hyper_space = hyperspace

        # policy deletion
        self.policy_iteration_threshold = args.policy_threshold
        # Initialize a dictionary to keep track of policy deletion status
        self.policy_deletion_status = {'archspace': {}, 'hpspace': {}}

        # assign polices to arch space
        for idx, (key, value) in enumerate(self.arch_space.items()):
            n_comps = len(value)
            self.policies['archspace'][idx] = Policy(n_comps, self.gpu_id, initial_params=None)

        # assign polices to hyper space
        for idx, (key, value) in enumerate(self.hyper_space.items()):
            n_comps = len(value)
            self.policies['hpspace'][idx] = Policy(n_comps, self.gpu_id, initial_params=None)

        if self.args.cont_training:
            logging.info("Loading already trained policies.")
            dir_weights = save_dir + self.args.controller_dir
            self.load_policies(dir_weights)

        parameters = [self.policies['archspace'][i].parameters() for i in self.policies['archspace']]
        parameters += [self.policies['hpspace'][i].parameters() for i in self.policies['hpspace']]

        assert parameters.__len__() == (self.policies['archspace'].__len__() + self.policies['hpspace'].__len__())
        parameters = [{'params': p} for p in parameters]

        # optimizer for controller parameters
        self.optimizer_controller = torch.optim.Adam(parameters, lr=self.args.controller_lr)

        if self.args.cont_training:
            # get last optimizer
            if self.optim_state is not None:
                self.optimizer_controller.load_state_dict(self.optim_state)

        self.replay_mem = self.args.replay_mem
        if self.replay_mem:
            self.replay_memory = ReplayMemory(self.args.replay_cap, self.args.replay_thres)
            self.replay_batch = self.args.replay_batch

            if self.args.cont_training:
                from src.utils.io import read_rollouts
                rollouts_path = "{}/rollouts.csv".format(self.args.cont_dir)
                rollouts_old = read_rollouts(rollouts_path, self.gpu_id)
                # append rollouts to replay memory
                count = self.replay_memory.push(rollouts_old)
                logging.info("Pushed {} rollouts on the replay memory.".format(count))

        logging.info("Controller optimizer is Adam with lr {}".format(self.args.controller_lr))

    def has_converged(self) -> bool:
        """
        Track convergence.
        :return: bool
        """
        return self.converged

    def sample(self, arch_computations, hyper_computations):
        """
        Randomly sample the model parameters and set of hyperparameters from combined space
        :param arch_computations:
        :param hyper_computations:
        :return:
        """
        architecture_actions = []
        hp_actions = []

        for i in range(arch_computations):
            action = Categorical(self.policies['archspace'][i]()).sample()
            architecture_actions.append(action)

        for i in range(hyper_computations):
            action_hp = Categorical(self.policies['hpspace'][i]()).sample()
            hp_actions.append(action_hp)

        return architecture_actions, hp_actions

    def policy_argmax(self, arch_computations, hyper_computations):
        """
        Return most likely candidate model and hyperparameters from combined space
        :param arch_computations:
        :param hyper_computations:
        :return:
        """
        layerwise_actions = []
        hp_actions = []
        for i in range(arch_computations):
            action = torch.argmax(self.policies['archspace'][i].params)
            layerwise_actions.append(action)

        for i in range(hyper_computations):
            action = torch.argmax(self.policies['hpspace'][i].params)
            hp_actions.append(action)

        # optimizer = torch.argmax(self.policies['hpspace']['optimizers'].params)
        # learning_rate = torch.argmax(self.policies['hpspace']['lr'].params)
        # hp_actions = [optimizer, learning_rate]
        return layerwise_actions, hp_actions

    def update(self, rollouts):
        """
        Perform update step of REINFORCE and sample from replay memory.
        :param rollouts:
        :return:
        """
        # sample from replay memory and save current rollouts
        if self.replay_mem:
            # copy current rollouts for push
            rollouts_replay_push = rollouts.copy()
            # sample from memory
            rollouts_replay = self.replay_memory.sample(self.replay_batch)
            # increase memory size
            if self.replay_memory.__len__() >= self.replay_batch * 2:
                old_replay_batch = self.replay_batch
                self.replay_batch *= 2
                logging.info("Increased replay sample size from {} to {}".format(old_replay_batch, self.replay_batch))
            if rollouts_replay.__len__() == 0:
                logging.info('Did not sample from replay memory!')
            else:
                logging.info('Appending {} rollouts from replay memory.'.format(len(rollouts_replay)))
                for r in rollouts_replay:
                    # check if tensor
                    rollouts.append(r)

        # in this case top1 accuracy
        rewards = [i[2]['acc_top1'] for i in rollouts]
        # exponentiate rewards
        if self.reward_map_fn:
            rewards = [self.reward_map_fn(r) for r in rewards]

        if self.args.dataset == "kinetics":
            # clip rewards for better update
            clip_value = 0.15
            logging.info("Clipping rewards under {} to 1".format(clip_value))
            rewards = [1 if val > clip_value else val for val in rewards]

        # calculate rewards using average reward as baseline
        if self.use_baseline and len(rollouts) > 1:
            avg_reward = np.mean(rewards)
            rewards = [r - avg_reward for r in rewards]

        # calculate log probabilities for each time step
        log_prob = []
        for t in rollouts:
            _log_prob = []
            arch_actions, hp_actions_ = t[:2]
            for i in range(len(self.policies['archspace'])):
                layer_action, layer_policy = arch_actions[i], self.policies['archspace'][i]
                _log_prob.append(Categorical(layer_policy()).log_prob(layer_action))

            for i in range(len(self.policies['hpspace'])):
                hyper_action, hyper_policy = hp_actions_[i], self.policies['hpspace'][i]
                _log_prob.append(Categorical(hyper_policy()).log_prob(hyper_action))
            log_prob.append(torch.stack(_log_prob).sum())

        self.optimizer_controller.zero_grad()
        loss = [-r * lp for r, lp in zip(rewards, log_prob)]
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer_controller.step()

        if self.replay_mem:
            # push current rollouts on replay memory
            count = self.replay_memory.push(rollouts_replay_push)
            logging.info("Pushed {} rollouts onto the Replay memory".format(count))
        logging.info('')

    def delete_policies(self, arch_choices, hyper_choices, threshold: float) -> (list, list):
        """
        Delete policies that are not performing well.
        1) find underperforming policies
            - threshold determination
            - updating indexes and re init policies
        2) update parameters for optimizer
        3)
        :param arch_choices:
        :param hyper_choices:
        :param threshold:
        :return:
        """
        for space in ['archspace', 'hpspace']:
            for idx, policy in self.policies[space].items():
                # get len of current policy and stop if it is one!
                size_policy = policy.params.shape[0]
                if size_policy == 1:
                    # skip if policy has only one value
                    continue
                min_index = torch.argmin(policy.params).item()
                # TODO how to deal with bigger policies???
                # take percentage???
                normalized_diff = round(float(torch.max(policy.params) - torch.min(policy.params)), 4)
                # TODO: npormalize this threshold
                # check if a value is below threshold
                if normalized_diff > threshold:
                    # delete policy
                    policy_without_min = torch.cat((policy.params[:min_index], policy.params[min_index + 1:]))
                    if idx not in self.policy_deletion_status[space]:
                        self.policy_deletion_status[space][idx] = []
                    self.policy_deletion_status[space][idx].append(min_index)
                    self.policies[space][idx] = Policy(None, self.gpu_id, initial_params=policy_without_min.tolist())

        arch_choices = self.__update_search_space(arch_choices, 'archspace')
        hyper_choices = self.__update_search_space(hyper_choices, 'hpspace')
        self.__update_controller()

        return hyper_choices, arch_choices

    def __update_search_space(self, choices, space):
        idx = 0
        for key, value in choices.items():
            if idx in self.policy_deletion_status[space]:
                del_indexes = self.policy_deletion_status[space][idx]
                if len(del_indexes) > 0:
                    # Sort del_indexes in reverse order to avoid index shifting
                    del_indexes.sort(reverse=True)
                    for i in del_indexes:
                        del choices[key][i]
            idx += 1
        return choices

    def __update_controller(self):
        logging.info("Updating the controller optimizer...")

        parameters = [self.policies['archspace'][i].parameters() for i in self.policies['archspace']]
        parameters += [self.policies['hpspace'][i].parameters() for i in self.policies['hpspace']]
        assert parameters.__len__() == (self.policies['archspace'].__len__() + self.policies['hpspace'].__len__())
        parameters = [{'params': p} for p in parameters]
        self.optimizer_controller = torch.optim.Adam(parameters, lr=self.args.controller_lr)

        logging.info("Done with updating controller optimizer!")

    def save_policies(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            torch.save(self.policies['archspace'][k].state_dict(), directory + 'archspace_' + str(k))
        for k in self.policies['hpspace']:
            torch.save(self.policies['hpspace'][k].state_dict(), directory + 'hpspace_' + str(k))

        torch.save(self.optimizer_controller.state_dict(), directory + 'optimizer')

    def load_policies(self, directory):
        if not os.path.isdir(directory):
            # raise ValueError('Directory %s does not exist' % directory)
            self.optim_state = None
            return

        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            _ = torch.load(directory + 'archspace_' + str(k))
            self.policies['archspace'][k].load_state_dict(_)
        for k in self.policies['hpspace']:
            _ = torch.load(directory + 'hpspace_' + k)
            self.policies['hpspace'][k].load_state_dict(_)

        try:
            self.optim_state = torch.load(directory + 'optimizer')
        except:
            self.optim_state = None
            logging.info("Optimizer state dict not saved!")
