import os
import sys
import random

from math import exp

from pyneat import pyneat

# Experiment Params
input_nodes = 2
output_nodes = 2

population_size = 1000
num_generations = 100

# Population cutoff percentage
cutoff = 0.15

# Percentage chance to mutate
mutation = 0.15

# Constants used in speciation
c1 = 1
c2 = 1
c3 = 1

# Log Experiment Params
print('='*30)
print("Population Size:\t{}".format(population_size))
print("Fitness Cutoff:\t\tTop {}%".format(int(cutoff*100)))
print("Chance of Mutation:\t{}%".format(int(mutation*100)))
print('='*30,'\n')

# Create the gene pool
gene_pool = pyneat.GenePool(population_size,
                     num_generations,
                     cutoff,
                     mutation,
                     [c1,c2,c3],
                     logging=1,
                     num_threads=8)

# Define a fitness function a genome can use to evaluate itself
def fitness_func(self):
    global input_nodes

    # Remember this isn't backprop, the batch
    # is just for evaluating loss
    batch_size = 32

    # Create a batch of random inputs
    vec_in = []
    for _ in range(batch_size):
        x = []
        for _ in range(input_nodes):
            x.append(float(random.randint(0,1)))
        vec_in.append(x)

    # Calculate network outputs
    vec_out = [self.network.forward(x) for x in vec_in]

    # This network will evolve to detect if all inputs are high
    expected = []
    for x in vec_in:
        a = [1.0,-1.0] if sum(x) == input_nodes else [-1.0,1.0]
        expected.append(a)

    # Fitness = the inverse loss
    mse = []
    for y, y_ in zip(vec_out, expected):
        for i in range(len(y)):
            error = (y[i] - y_[i]) ** 2
            mse.append(error)
    self.fitness = 1 - (sum(mse) / len(mse))

    # In this experiment, we have a quantifiable optimum,
    # thus we should save this particular genome.
    if self.fitness > 0.98:
        print('\n~ found optimal network: {} ~'.format(self.fitness))

        # Create and run 5 tests
        tests = [[random.uniform(0.0,1.0) for _ in range(input_nodes)] for _ in range(5)]
        for x in tests:

            # In this case we reset the network to ignore recurrent connections,
            # we haven't been using them during evolution, thus we won't here.
            self.network.reset
            y = self.network.forward(x)

            # Simplify the log information
            input_text = 'all active' if sum(x) == input_nodes else 'not active'
            output_text = str([round(x,3) for x in y])[1:-1]
            print("{}\t{}".format(input_text, output_text))

            # TODO: Here you'd wanna save the best if you have a max fitness

        sys.exit()

# Define a distribution function for parent selection
def distribution_func(x):
    return exp(-10 * (x / int(population_size * cutoff)))

# Initialise the gene pool
gene_pool.init(input_nodes, output_nodes, fitness_func, distribution_func)

# Run an evolutionary period
gene_pool.evolve()
