import os
import numpy as np

from utils import dict_to_matrix


def forward_algorithm(iterations, transition_model, true_observation_model,
                      false_observation_model, start_distribution_probability):
    '''
    Algorithm goal to compute P(Z[k], X[1:k]), given the emission, transition,
    starting probabilites. Trick to this algorithm is that we add one more
    element which is z[k-1] and factor things over it using Markov Property
    and d-separation for that one step back, yielding in a recursive property

    P(Z[k], X[1:k]) = Sum(p(x[k]|z[k]) * p(z[k]|z[k-1]) * p(z[k-1], x[1:k-1])),
    the summation is done over Z[k-1]

    Beauty of this algorithm is it is of O(n*(m^2)).

    ----

    Intuitively, given a certain observation x[k] about latent variable z[k],
    we are reasoning about this observation given all the past events.
    '''
    # Given alpha_1 = prior probability of z_1 multiplied by
    # conditional probability x_1 given e_1
    print('[INFO] Computing forward algorithm with alpha={}'.format(
        start_distribution_probability))
    previous = start_distribution_probability
    forward_list = []
    separator = ' '
    last_forward_list = []
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
        last_forward_list.append(forward_list[-1])
    print('[INFO] Computed forward algorithm. Output is saved.\n')
    return forward_list, last_forward_list


def backward_algorithm(iterations, sequence_length, transition_model,
                       true_observation_model, false_observation_model):
    '''
    Algorithm goal to compute P(X[k+1:n]|Z[k]), given the emission, transition,
    starting probabilites. Trick to this algorithm is that we add one more
    element which is z[k+1] and following same steps as the forward algorithm,
    yielding in a recursive property.

    P(X[k+1:n]|Z[k]) = Sum(p(x[k+2]|z[k+1]) * p(x[k+1]|z[k+1]) * p(z[k+1], z[k])),
    the summation is done over Z[k+1]

    ----

    Intuitively, given a certain latent variable z[k],
    we are reasoning about this variable given all the future observations
    x[k+1:n].
    '''

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
    print('[INFO] Computed backward algorithm. Output is saved.\n')
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
    Goal: Compute p(z[k]|x[1:n]) = p(x[k+1:n]|z[k]) * p(z[k], x[1:k])
    Forward_backward_algorithm = backward_step * forward_step

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
    print('[INFO] Computed forward backward algorithm. Output is saved.\n')
    return forward_backward_list


def improved_forward_backward(iterations,
                              last_forward_storage,
                              transition_model,
                              true_observation_model,
                              false_observation_model):
    '''
    What we are basically doing is propagating the
    forward message in backward manner

    1. Run forward up to current timestamp 't'
    2. Discard all keeping only the last message in forward list
    [Up till here was done during computation of forward_algorithm]
    3. Run backward pass for both b and f
    '''

    transition_model_inv = np.linalg.inv(transition_model)
    finalbackward_fm_storage = []
    umbrella_sequence = []
    separator = ' '

    # Retrieving the samples, to do inference on
    for i in range(iterations):
        sample = os.path.join(
            os.getcwd(), "Samples", "sample{}.txt".format(str(i+1)))
        with open(sample, "r") as file:
            for line in file:
                if separator in line:
                    features = line.strip().split(separator, maxsplit=1)
                    umbrella_sequence.append(features[1])

    for sequence_index in range(0, len(umbrella_sequence)):
        current_obs = []
        index_last_forward = 0

        current_last_forward = last_forward_storage[index_last_forward]
        backward_fm_storage = []
        for umbrella_index in range(len(umbrella_sequence[sequence_index]), 0, -1):
            backward_fm_normalized_vec = [0, 0]
            umbrella = umbrella_sequence[sequence_index].split(' ')
            current_umbrella = umbrella[1]
            if (current_umbrella == "umbrella"):
                current_obs = true_observation_model
            if (current_umbrella == "not_umbrella"):
                current_obs = false_observation_model
            current_obs = np.linalg.inv(current_obs)
            current_backward_fm = (np.matmul(transition_model_inv, current_obs)).dot(
                current_last_forward)
            backward_fm_normalized_vec[0] = (
                current_backward_fm[0]/((current_backward_fm[0]+current_backward_fm[1])/100))/100
            backward_fm_normalized_vec[1] = (
                current_backward_fm[1]/((current_backward_fm[0]+current_backward_fm[1])/100))/100
            current_last_forward = backward_fm_normalized_vec
            current_storage = [backward_fm_normalized_vec[0],
                               backward_fm_normalized_vec[1]]
            backward_fm_storage.append(current_storage)
            index_last_forward += 1
        finalbackward_fm_storage.append(backward_fm_storage)
    return finalbackward_fm_storage


def fixed_lag_smoothing(sequence_length, start_distribution_probability,
                        true_observation_model, false_observation_model,
                        fixed_lag_constant, transition_model, forward_list):
    inverse_transition = np.linalg.inv(transition_model)
    inverse_true_obs = np.linalg.inv(true_observation_model)
    inverse_false_obs = np.linalg.inv(false_observation_model)
    beta = np.array([[1, 0], [0, 1]])
    fl_sm = np.zeros((sequence_length, 2))
    separator = ' '
    for t in range(sequence_length):
        sample = os.path.join(
            os.getcwd(), "Samples", "sample{}.txt".format(str(t+1)))
        with open(sample, "r") as file:
            temp = []
            for line in file:
                if separator in line:
                    features = line.strip().split(separator, maxsplit=1)
                    temp.append(features[1])
            if (t > fixed_lag_constant):
                if (temp[t] == 'rainy umbrella'):
                    if (temp[t-fixed_lag_constant] == 'rainy umbrella'):
                        beta = inverse_true_obs.dot(inverse_transition).dot(
                            beta).dot(transition_model).dot(true_observation_model)
                    else:
                        beta = inverse_false_obs.dot(inverse_transition).dot(
                            beta).dot(transition_model).dot(true_observation_model)
                else:
                    if (temp[t-fixed_lag_constant] == 'rainy umbrella'):
                        beta = inverse_true_obs.dot(inverse_transition).dot(
                            beta).dot(transition_model).dot(false_observation_model)
                    else:
                        beta = inverse_false_obs.dot(inverse_transition).dot(
                            beta).dot(transition_model).dot(false_observation_model)
                fl = forward_list.dot(beta)
                a = 1/fl[0, 0]+fl[0, 1]
                fl_sm[t, 0] = round(fl[0, 0]*a, 4)
                fl_sm[t, 1] = round(fl[0, 1]*a, 4)
            else:
                if (temp[t] == 'rainy umbrella'):
                    beta = beta.dot(transition_model).dot(
                        true_observation_model)
                else:
                    beta = beta.dot(transition_model).dot(
                        false_observation_model)
    return fl_sm
