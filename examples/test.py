import sys
import random
from math import exp

from pyneat import pyneat

population_size = 100
num_generations = 1000

cutoff = 0.25
mutation = 1.0

c1 = c2 = c3 = 1

input_nodes = 4
output_nodes = 2

def test_fit(self):
	self.fitness = random.uniform(0.0, 10.0)

def test_dist(x):
    return exp(-10 * (x / int(population_size * cutoff)))

gene_pool = pyneat.GenePool(population_size,
                     num_generations,
                     cutoff,
                     mutation,
                     [c1,c2,c3],
                     logging=1,
                     num_threads=8)

gene_pool.init(input_nodes, output_nodes, test_fit, test_dist)
gene_pool.evolve()

