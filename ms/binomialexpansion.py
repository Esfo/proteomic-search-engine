from collections import defaultdict, Counter
import itertools
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from time import time
import random
import profile
import heapq
import math
import os

#for more https://matplotlib.org/stable/tutorials/introductory/customizing.html
if os.uname()[1] == 'toaster':
    plt.rcParams['figure.dpi'] = 180
elif os.uname()[1] == 'box':
    plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.facecolor'] = 'gray'
plt.rcParams['figure.facecolor'] = 'gray'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['ytick.labelcolor'] = 'white'
plt.rcParams['xtick.labelcolor'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
chexes = ['#ffffff',
        '#e85d58',
        '#b88cfa',
        '#f5972c',
        '#2ded8d',
        '#4bc8f2',
        '#ea68f2',
        ]
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', chexes)
#n = 0
#for c in chexes:
#    plt.plot([n, n+1], [n, n+1], color=c)
#    n += 1
#plt.show()

from collections import deque
import heapq

def calculate_probability(state, initial_probabilities):
    """Calculate the probability of a state."""
    probability = 1
    for num, prob in zip(state, initial_probabilities):
        probability *= (prob ** num)
    return probability

def generate_combinations_with_probabilities(start_list, initial_probabilities):
    # Convert the start list to a tuple for set operations
    start_state = tuple(start_list)
    # Calculate the starting probability
    start_probability = calculate_probability(start_state, initial_probabilities)
    # Initialize a heap with the starting state and its probability
    heap = [(-start_probability, start_state)]
    # Initialize a set with the starting state to keep track of visited states
    visited = set([start_state])

    # List to hold the final states and their probabilities
    final_states_with_probabilities = []

    # Heap based BFS over the states of the list
    while heap:
        # Pop the state with the highest probability
        current_probability, current_state = heapq.heappop(heap)
        # Add the current state and its probability to the final states (negate probability back)
        final_states_with_probabilities.append((current_state, -current_probability))
        
        # Iterate over each index in the state
        for i in range(len(current_state) - 1):
            # Only generate new states if the current element is not '0'
            if current_state[i] > 0:
                # Increment the next element and decrement the current one
                new_state = list(current_state)
                new_state[i] -= 1
                new_state[i + 1] += 1
                new_state = tuple(new_state)
                # Calculate the new probability
                new_probability = calculate_probability(new_state, initial_probabilities)
                # Check if we have already visited this state
                if new_state not in visited:
                    # Add the new state to the heap and visited set
                    heapq.heappush(heap, (-new_probability, new_state))
                    visited.add(new_state)

    return final_states_with_probabilities

# Test the function with lists of different lengths and their probabilities
test_lists_with_probs = {
    'length_2': ([31, 2], [0.7, 0.3]),
    'length_4': ([30, 1, 1, 0], [0.5, 0.3, 0.1, 0.1]),
    'length_5': ([28, 2, 1, 0, 0], [0.4, 0.3, 0.2, 0.1, 0.0])
}

# Generate combinations with probabilities for each test case
test_results_with_probs = {}
for length, (test_list, probs) in test_lists_with_probs.items():
    test_results_with_probs[length] = generate_combinations_with_probabilities(test_list, probs)

# Display the combinations for each test case sorted by probability
for length, states_with_probs in test_results_with_probs.items():
    print(f"Combinations for list of {length} ordered by probability:")
    for state, prob in sorted(states_with_probs, key=lambda x: x[1], reverse=True)[:10]:
        print(f"State: {state}, Probability: {prob}")
    print("\n")
