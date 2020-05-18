import os
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="Path of log file", type=str, required=True)
    parser.add_argument('--sat_line', help="start line of log file", type=int, required=True)
    parser.add_argument('--end_line', help="end line of log file", type=int, default=-1)

    return parser.parse_args()


def moving_average(input_list, win_size):
    mean = list()
    for i in range(len(input_list) - win_size):
        average = 0
        for k in range(i, i+win_size):
            average += input_list[k]
        mean.append(average / win_size)

    return mean


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.path):
        raise Exception('Wrong log path')

    with open(args.path, "r") as f:
        mes = f.readlines()
        mes = mes[args.sat_line: args.end_line]

    loss = []
    for str in mes:
        str_idx = str.split('||')
        loss_str = str_idx[3].split(':')[1].strip(';')
        loss.append(float(loss_str))

    loss_mean = moving_average(loss, 10)
    plt.plot(loss_mean)
    exit(0)