# https://it.m.wikipedia.org/wiki/Algoritmo_forward-backward
import os
import numpy as np


def forward_algorithm(iterations, transition_model, true_observation_model,
                      false_observation_model, start_distribution_probability):
    # Given alpha_1 = prior probability of z_1 multiplied by
    # conditional probability x_1 given e_1
    print('[INFO] Computing forward algorithm with alpha={}'.format(
        start_distribution_probability))
    previous = start_distribution_probability
    forward_list = []
    separator = ' '
    # Converting dictionary to numpy array, to compute dot
    true_observation_model_arr = np.array(
        list(true_observation_model.values()))
    false_observation_model_arr = np.array(
        list(false_observation_model.values()))
    transition_model_arr = np.array(
        list(transition_model.values()))
    for i in range(iterations):
        sample = os.path.join(
            os.getcwd(), "Samples", "sample{}.txt".format(str(i+1)))
        with open(sample, "r") as file:
            for line in file:
                if(separator in line):
                    features = line.strip().split(separator, maxsplit=1)
                    observation = features[1]
                    if(observation == 'umbrella'):
                        p1 = np.dot(true_observation_model_arr,
                                    transition_model_arr)
                    else:
                        p1 = np.dot(false_observation_model_arr,
                                    transition_model_arr)
                    p2 = np.dot(p1, previous)
                    normalization_factor = (1 / (p2[0] + p2[1]))
                    p2[0] *= normalization_factor
                    p2[1] *= normalization_factor
                    previous = p2
                    forward_list.append(previous)
    print('[INFO] Computed forward algorithm.\n')
    return forward_list


def backward_algorithm(iterations, sequence_length, transition_model,
                       true_observation_model, false_observation_model):
    print('[INFO] Computing backward algorithm with beta={}'.format(
        np.array([1, 1])))
    # given Beta_n = 1
    previous = np.array([1, 1])
    backward_list = []
    separator = ' '
    # Converting dictionary to numpy array, to compute dot
    true_observation_model_arr = np.array(
        list(true_observation_model.values()))
    false_observation_model_arr = np.array(
        list(false_observation_model.values()))
    transition_model_arr = np.array(
        list(transition_model.values()))
    for i in range(iterations):
        sample = os.path.join(
            os.getcwd(), "Samples", "sample{}.txt".format(str(i+1)))
        with open(sample, "r") as file:
            temp = []
            temp2 = []
            for line in file:
                if separator in line:
                    features = line.strip().split(separator, maxsplit=1)
                    temp.append(features[1])
            temp.reverse()
            for k in range(sequence_length):
                observation = temp[k][1]
                if(observation == 'umbrella'):
                    p1 = np.dot(true_observation_model_arr,
                                transition_model_arr)
                else:
                    p1 = np.dot(false_observation_model_arr,
                                transition_model_arr)

                p2 = np.dot(p1, previous)
                normalization_factor = (1 / (p2[0] + p2[1]))
                p2[0] *= normalization_factor
                p2[1] *= normalization_factor
                previous = p2
                temp2.append(previous)
            temp2.reverse()
        backward_list += temp2
    print('[INFO] Computed backward algorithm.\n')
    return backward_list


def forward_backward_algorithm(forward_list, backward_list,
                               start_distribution_probability):
    '''
    Forward-Backward Algorithm, is used to solve one of the
    3 basic problems that HMM can solve, which are:

    1. Evaluation Problem: it is simply the answer to what is the probability
    that a particular sequence of symbols is produced by a model. And,
    we use two algorithms forward algorithm or the backward

    2. Decoding Problem: given a sequence of observations and a model,
    what is the most likely sequence of states produced these observations

    3. Training Problem: given model and set of sequences, find model that best
    fits the data, we can use forward-backward algorithm
    (gives marginal probability for each individual state),
    maximum likelihood estimation,
    or Viterbi training (probability of the most likely sequence of states)

    In our case, Forward Backward would tell you the probability of
    it being "sunny" for each day, Viterbi would give the most likely sequence
    of sunny/rainy days, and the probability of this sequence.

    Forward-Backward Algorithm: On its own, it isn't used for training an
    HMM's parameters, but only for smoothing: computing
    the marginal likelihoods of a sequence of states.

    Forward Backward algorithm works in the following way.

    For each sequence in the training set of sequences.

    1. Calculate forward probabilities with the forward algorithm
    2. Calculate backward probabilities with the backward algorithm
    3. Calculate the contributions of the current sequence to the
        transitions of the model, calculate the contributions of the current
        sequence to the emission probabilities of the model.
    4. Calculate the new model parameters (start probabilities,
        transition probabilities, emission probabilities)
    5. Calculate the new log likelihood of the model
    6. Stop when the maximum number of iterations is passed -in this case-.
    '''
    print('[INFO] Computing forward backward algorithm')
    backward_list_len = len(backward_list)

    forward_backward_list = []
    for i in range(backward_list_len):
        if((i == 0)or(i % 20 == 0)):
            p = start_distribution_probability * backward_list[i]
        elif (i % 20 == 19):
            p = forward_list[i] * np.array([1, 1])
        else:
            p = forward_list[i]*backward_list[i+1]
        normalization_factor = (1 / (p[0]+p[1]))
        p[0] *= normalization_factor
        p[1] *= normalization_factor
        forward_backward_list.append(p)
    print('[INFO] Computed forward backward algorithm.\n')
    return forward_backward_list
