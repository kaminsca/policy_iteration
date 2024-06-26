import numpy as np
import pprint as pprint
def get_uniform_policy(num_states=25, num_actions=4):
    """
    Returns a uniform policy, where for each state there is a uniform probability
    distribution over possible actions.

    Parameters:
        num_states - int corresponding to the number of states in the MDP
        num_actions - int corresponding to the number of actions in the MDP

    Returns:
        uniform random policy
    """
    return np.tile(np.ones(num_actions)/num_actions, (num_states, 1))

def visualize_policy(policy):
    """
    Uses the policy to print the appropriate NSWE in the support for that policy
    for each state.

    Parameters:
        policy - (25, 4) numpy array where each row is a probability distribution
                 over moves for a state. If it is deterministic, then the probability
                 will be a one hot vector.
                 If there is a tie between two actions, the tie is broken with equal
                 probabilities. Initialized to uniform random policy.

    Returns:
        None
    """

    for row in range(5):
        for col in range(5):
            i = 5*row + col
            if policy[i, 0] > 0:
                print("N", end="")
            if policy[i, 1] > 0:
                print("E", end="")
            if policy[i, 2] > 0:
                print("S", end="")
            if policy[i, 3] > 0:
                print("W", end="")
            print(" ", end="")
        print()

def gridworld(slip_prob=0.1):
    """
    Constructs the gridworld MDP according to the specification for problem 3

    Parameters:
        slip_prob - the probability that an agent slips and remains in current state
                    instead of transitioning to state s' according to action a.

    Returns:
        P = {
            s1: {a1: [(p(s_1|s1,a1), s_1, reward(s_1,s1,a1)),
                      (p(s_2|s1,a1), s_2, reward(s_2,s1,a1)),
                        ...
                     ],
                 a2: ...,
                        ...
                },
            s2: ...,
            ...
        }
    """
    no_slip = 1 - slip_prob
    P = {}

    for index in range(25):
        key = 's' + str(index)
        a = {}
        # from state A: all four actions yield a reward of +10 and take the agent to A’ (21) deterministically
        if index == 1:
            for action in range (4):
                action_key = 'a' + str(action)
                a[action_key] = [[1, 's21', 10]]
        # From state B, all four actions yield a reward of +5 and take the agent to B’ (13) deterministically
        elif index == 3:
            for action in range (4):
                action_key = 'a' + str(action)
                a[action_key] = [[1, 's13', 5]]
        else:
            for action in range (4):
                action_key = 'a' + str(action)
                res = []
                # action 0: north
                if action == 0:
                    # off grid -- stay still
                    if index < 5:
                        # results: prbability, next state, reward
                        res.append([1, key, -1])
                    else:
                        res.append([no_slip, 's' + str(index - 5), 0])
                        res.append([slip_prob, key, 0])
                # action 1: east
                elif action == 1:
                    if index % 5 == 4:
                        res.append([1, key, -1])
                    else:
                        res.append([no_slip, 's' + str(index + 1), 0])
                        res.append([slip_prob, key, 0])
                # action 2: south
                elif action == 2:
                    if index > 19:
                        res.append([1, key, -1])
                    else:
                        res.append([no_slip, 's' + str(index + 5), 0])
                        res.append([slip_prob, key, 0])
                # action 3: west
                else:
                    if index % 5 == 0:
                        res.append([1, key, -1])
                    else:
                        res.append([no_slip, 's' + str(index - 1), 0])
                        res.append([slip_prob, key, 0])
                a[action_key] = res
        P[key] = a
    return P

def policy_eval(P, policy=get_uniform_policy(), theta=0.0001, gamma=0.9):
    """
    Performs the policy evaluation algorithm as specified in the textbook
    to update the V function.

    Parameters:
        P - nested dictionary as returned by gridworld()
        theta - stopping condition
        gamma - discount factor



    Returns:
        V - (5, 5) numpy array where each entry is the value of the corresponding location.
            Initialized with zeros.
    """
    V = np.zeros(25)
    # print(V)
    # print(policy)
    # pprint.pprint(P)
    delta = theta + 1
    while delta > theta:
        V_old = np.copy(V)
        delta = 0
        # loop over states
        for s_index,s in enumerate(policy):
            # s might look like [0.25 0.25 0.25 0.25]
            v = V_old[s_index]
            new_val = 0
            # print()
            # loop over actions probabilities in each state
            for direction, prob in enumerate(s):
                state_key = 's' + str(s_index)
                action_key = 'a' + str(direction)
                transitions = P[state_key][action_key]
                next_states_val = 0
                # loop over next states
                for transition in transitions:
                    # transition might look like [0.9, 's0', -1]
                    s_prime = int(transition[1][1:])
                    # probability * (r + gamma * V(s_prime))
                    next_states_val += transition[0] * (transition[2] + gamma * V_old[s_prime])
                # pi(a|s) * sum of next states (probability * (r + gamma * V(s_prime)))
                new_val += prob * next_states_val
                # print(state_key, action_key, prob, transitions, new_val)
            V[s_index] = new_val
            delta = max(delta, abs(v - V[s_index]))
    V2d = np.reshape(V, (-1, 5))
    return(V2d)

def policy_iter(P, theta=0.0001, gamma=0.9):
    """
    Performs the policy iteration algorithm as specified in the textbook
    to find the optimal policy

    Parameters:
        P - nested dictionary as returned by gridworld()
        theta - stopping condition
        gamma - discount factor

    Returns:
        policy - (25, 4) numpy array where each row is a probability distribution
                 over moves for a state. If it is deterministic, then the probability
                 will be a one hot vector.
                 If there is a tie between two actions, the tie is broken with equal
                 probabilities. Initialized to uniform random policy.
        V - (5, 5) numpy array where each entry is the value of the corresponding location,
                   calculated according to policy.
    """
    # V = policy_eval(P)
    policy = get_uniform_policy()
    options = [0,1,2,3]
    i = 0
    policy_stable = False
    while not policy_stable:
        # evaluate policy 
        V = policy_eval(P, policy=policy)
        V = V.flatten()
        policy_stable = True
        # sweep through state space
        for index, s in enumerate(V):
            # choose old action based on probabilities
            probabilities = np.copy(policy[index]) 
            # choose new action by evaluating next state
            actions = []
            for direction,a in enumerate(policy[index]):
                state_key = 's' + str(index)
                action_key = 'a' + str(direction)
                transitions = P[state_key][action_key]
                next_states_val = 0
                # loop over next states
                for transition in transitions:
                    # sum over s' (p(s',r|s,a)[r + gamma V(s')])
                    s_prime = int(transition[1][1:])
                    next_states_val += transition[0] * (transition[2] + gamma * V[s_prime])
                actions.append(next_states_val)
            # get new action (0,1,2, or 3)
            actions = np.around(actions, 2)
            max_action = max(actions)
            updated = np.zeros(4)
            max_indices = np.where(actions == max_action)[0]
            for j in max_indices:
                updated[j] = 1.0 / len(max_indices)
            policy[index] = updated
            # unchanged: done with policy iteration
            # print('i', index, 'prob', probabilities, 'policy', policy[index])
            comparison = probabilities == policy[index]
            if not comparison.all():
                policy_stable = False
        # print(i)
        # visualize_policy(policy)
        i += 1
    V2d = np.reshape(V, (-1, 5))
    return policy, V2d


if __name__ == '__main__':
    P = gridworld()
    print('P: ')
    pprint.pprint(P)
    eval = policy_eval(P)
    np.set_printoptions(precision=2)
    print('initial evaluation: \n', eval)
    pi_star, v_star = policy_iter(P)
    print('pi*: ')
    visualize_policy(pi_star)
    print('v*: \n', v_star)