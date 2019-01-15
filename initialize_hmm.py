def init():
    '''
    HMM are basically Markov Models, but we can't observe the states (S)
    directly, so, we need the states' observations to infer the states.
    But, What are Markov Models, stochastic model describing a sequence of
    possible events.
    Markov Models are directed -growing- bayes networks, that assumes that
    S_(t+1) is only dependent on S_(t)

    To formalize the given model, we need to state
    1. (N) number of states, S= {S_1, S_2, ....., S_N}
    2. (M) number of Observations, V = {V_1, V_2, ....., V_M}
    3. State transition Probabilities, A = mat[a_ij]; aij = P(S_j|S_i)
    i.e. probability of going from state (i) to state (j)
    4. Observation Probabilities, B = (b_j(m)); b_j(m)=P(V_m|S_t)
    5. Initial state probabilities, PI = [pi_i]; pi_i = P(S_i)

    And so, the joint distribution of all hidden states & their observations
    depend on transition model, observation probabilities, & prior probability
    P(S_1:n, V_1:n) = pi(S_1)*B_z1(x_1)*product(T(S_k-1, S_k)*B_zk(x_k))

    '''
    print('[INFO] Initializing Hidden Markov Model...')
    # Define HMM states & observations
    states = ('rainy', 'not_rainy')
    observations = ('umbrella', 'not_umbrella')

    # Define transition model A = [T = P(X_t | X_t-1)]
    # representing all the possible states in HMM
    transition_model = {
        'rainy': [0.7, 0.3],
        'not_rainy': [0.3, 0.7],
    }

    '''
    Define Observation Model [B]
    In Markov Model, multiply a state vector with the
    transition_model_array to get probabilities of subsequent events
    In HMM, states are unknown & we observe the events associated
    with possible states
    
    Thus, the observation model is as follows, based on the givens
    in the exercise,

    An umbrella appearing in our environment will be there 90% of the time,
    if it is "raining", only 10% of the time, it will be forgotten.

    if it is "not raining", an umbrella will appear only 20% of the time,
    and the higher probability goes to no raining, no umbrella.

              | Umbrella | No Umbrella |
    ------------------------------------
    Rainy     | / 0.9 \  | / 0.1 \     |
    Not Rainy | \ 0.2 /  | \ 0.8 /     |
    '''
    observation_model = {
        'rainy': [0.9, 0.1],
        'not_rainy': [0.2, 0.8]
    }

    '''
    Multiplying the state vector & the observation matrix,
    containing only elements along the diagonal
    Each element is the probability of the observed event given each state
    '''
    # True model
    observation_true_model = {
        'rainy': [0.9, 0.0],
        'not_rainy': [0.0, 0.2],
    }

    # False model
    observation_false_model = {
        'rainy': [0.0, 0.1],
        'not_rainy': [0.8, 0.0],
    }

    '''
    Allowing calculation of the probabilities associated with a transition
    to a new state and to observe a given event, given state vector is (pi)
    F(0:1) = pi*T*O(1) >> F(0:t) = F(0:t-1)*T*O(t)
    '''
    print('[INFO] Hidden Markov Model is initialized.\n')
    return (states, observations, transition_model,
            observation_model, observation_true_model,
            observation_false_model)
