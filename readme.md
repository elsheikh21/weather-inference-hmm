# Hidden Markov Models

Use it:

1. Clone the repo

   > git clone github.com/elsheikh21/weather-inference-hmm

2. Start by running `weather_hmm.py`

Requirements: Native python libraries, nothing fancy

Note: anything was needed as saving output to files, changing dictionaries to matrices, is included in `utils.py`

---

## Introduction

HMM are basically Markov Models, but we can't directly observe the states (S),
so, we need to introduce latent variables to be able to make observations about hidden variables to infer the states.

But, What are Markov Models? They are stochastic model describing a sequence of possible events.
Markov Models are directed -growing- bayes networks, that assumes that S*[t+1] is only dependent on S*[t]

## Formalizing the model

To formalize the given model, we need to state

1. (N) number of states, S= {S_1, S_2, ....., S_N}
2. (M) number of Observations, V = {V_1, V_2, ....., V_M}
3. State transition Probabilities, A = mat[a_ij] where aij = P(S_j|S_i)
   i.e. probability of going from state (i) to state (j)
4. Observation(emission) Probabilities, B = (b_j(m)); b_j(m)=P(V_m|S_t)
5. Initial state(prior) probabilities, PI = [pi_i]; pi_i = P(S_i)

And so, the joint distribution of all hidden states & their observations
depend on transition model, observation probabilities, & prior probability
P(S_1:n, V_1:n) = pi(S_1)*B_z1(x_1)*product(T(S_k-1, S_k)\*B_zk(x_k))

Check out file `initialize_hmm.py`

## Sampling

Given, the model prior probabilities, transition probabilities, observations, required to generate a number of samples, to be able to evaluate the sequence.

Check out file `sampling.py`

## Inference

Forward-Backward Algorithm, is used to solve one of the 3 basic problems that HMM can solve, which are:

1.  Evaluation Problem: it is simply the answer to what is the probability
    that a particular sequence of symbols is produced by a model. And,
    we use two algorithms forward algorithm or the backward

2.  Decoding Problem: given a sequence of observations and a model,
    what is the most likely sequence of states produced these observations

3.  Training Problem: given model and set of sequences, find model that best
    fits the data, we can use forward-backward algorithm
    (gives marginal probability for each individual state),
    maximum likelihood estimation,
    or Viterbi training (probability of the most likely sequence of states)

    In our case, Forward Backward would tell you the probability [evaluation problem]
    of weather being "sunny" for each day, Viterbi would give the most likely sequence
    of sunny/rainy days, and the probability of this sequence.

    Forward-Backward Algorithm: On its own, it isn't used for training an
    HMM's parameters, but only for smoothing: computing
    the marginal likelihoods of a sequence of states.

    **How it works?**
    Forward Backward algorithm works in the following way.

    > Goal: Compute p(z[k]|x[1:n]) = p(x[k+1:n]|z[k]) \_ p(z[k], x[1:k])
    > Forward_backward_algorithm = backward_step x forward_step

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

    **Forward Algorithm**
    Algorithm goal to compute P(Z[k], X[1:k]), given the emission, transition,
    starting probabilites. Trick to this algorithm is that we add one more
    element which is z[k-1] and factor things over it using Markov Property
    and d-separation for that one step back, yielding in a recursive property

    > P(Z[k], X[1:k]) = Sum(p(x[k]|z[k]) _ p(z[k]|z[k-1]) _ p(z[k-1], x[1:k-1])),
    > the summation is done over Z[k-1]

    Beauty of this algorithm is it is of O(n\*(m^2)).

    Intuitively, given a certain observation x[k] about latent variable z[k],
    we are reasoning about this observation given all the past events.

    **backward Algorithm**
    Algorithm goal to compute P(X[k+1:n]|Z[k]), given the emission, transition,
    starting probabilites. Trick to this algorithm is that we add one more
    element which is z[k+1] and following same steps as the forward algorithm,
    yielding in a recursive property.

    > P(X[k+1:n]|Z[k]) = Sum(p(x[k+2]|z[k+1]) _ p(x[k+1]|z[k+1]) _ p(z[k+1], z[k])),
    > the summation is done over Z[k+1]

    Intuitively, given a certain latent variable z[k],
    we are reasoning about this variable given all the future observations x[k+1:n].

    **Forward Backward Algorithm Improvement**
    As we noticed forward backward algorithm needs to have both the forward list and backward list stored in memory for computing, where each of them is of length 300, number of sequences created \* number of iterations per sequence.

    So why not? to compute forward backward step iteratively, without storing all values? This is exactly what we did in the `improved_forward_backward` method, reducing space needed in memory.

    How we did this?

    1. Run forward up to current timestamp 't'
    2. Discard all keeping only the last message in forward list
    3. Run backward pass for both b and f

    yielding in a constant storage, independent of 't'

Check out file `hmm_inference.py`
