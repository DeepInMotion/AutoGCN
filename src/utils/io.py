import argparse
import ast
import re
import torch
import csv


def str2bool(value: str) -> bool:
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_rollouts(file_path: str, gpu_id) -> list:

    rollouts_old = []
    tensor_pattern = r"tensor\((\d+)"
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            # data = [re.findall(pattern, column) for column in row]
            # Remove the unnecessary parts of the string
            # Extract the tensor(x) values as strings
            tensor_strings_1 = re.findall(tensor_pattern, row[0])
            tensor_strings_2 = re.findall(tensor_pattern, row[1])
            tensor_values_1 = [torch.tensor(int(x), device=gpu_id) for x in tensor_strings_1]
            tensor_values_2 = [torch.tensor(int(x), device=gpu_id) for x in tensor_strings_2]
            # tensor_values_1 = torch.tensor([int(value) for value in values])

            dictionary = eval(row[2])
            integer = int(row[3])
            # check if this is necessary
            rollouts_old.append([tensor_values_1, [tensor_values_2], dictionary, integer])

    return rollouts_old


def read_rollouts_(file_path: str, gpu_id) -> list:
    """
    loading old rollouts and appending them to the replay memory
    super hacky might change the data format
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    rollouts_old = []
    pattern = r"\[(.*?)\],\[(.*?)\],(\{.*\}),(\d+)"
    tensor_pattern = r"tensor\((.*?)\)"
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            tensor1 = []
            tensor2 = []
            for t in match.group(1).split(', '):
                tensor_match = re.match(tensor_pattern, t)
                if tensor_match:
                    tensor1.append(torch.tensor(int(tensor_match.group(1))).to(gpu_id))

            for t in match.group(2).split(', '):
                tensor_match = re.match(tensor_pattern, t)
                if tensor_match:
                    tensor2.append(torch.tensor(int(tensor_match.group(1))).to(gpu_id))

            dictionary = eval(match.group(3))
            integer = int(match.group(4))

            rollouts_old.append([tensor1, tensor2, dictionary, integer])

    return rollouts_old


def read_actions(file_path: str) -> list:
    """
    read the action list file
    """
    action_list = []
    number_pattern = r'(\d+(?:\.\d+)?)'
    list_pattern = r'\[([^]]+)\]'
    dict_pattern = r'({[^{}]+})'

    with open(file_path, 'r') as f:
        for line in f:
            number_match = re.search(number_pattern, line)
            student_id = int(number_match.group(1))

            list_match = re.search(list_pattern, line)
            list_actions = [int(x) for x in list_match.group(1).split(',')]

            # dict_match = re.search(dict_pattern, line)
            # dict_string = dict_match.group(0)
            # dict_string_2 = dict_match.group(1)

            dict_matches = re.findall(dict_pattern, line)
            dict_strings = [dict_match.strip() for dict_match in dict_matches]

            action_list.append([student_id, list_actions] + dict_strings)

    return action_list
