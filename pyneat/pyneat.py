# standard library
import inspect
import random
import sys
import os

from collections import namedtuple
from multiprocessing import Pool

from enum import Enum
from math import exp

# third-party library
from tabulate import tabulate



# Helper Abstractions
Neuron = namedtuple('Neuron',
                        ['type', 'id'])

Connection = namedtuple('Connection',
                        ['input', 'output', 'weight', 'enabled', 'innovation'])

class Type(Enum):
    IN = 0
    HIDDEN = 1
    OUT = 2
    BIAS = 3

class Saver(Enum):
    DISABLED = 0
    UNBIASED_GENOME = 1
    BIASED_GENOME = 2
    UNBIASED_POPULATION = 3
    BIASED_POPULATION = 4
    USER_DEFINED = 5



# Activation Functions
def sigmoid(x):
    if x > 3:
        return 1.0
    elif x < -3:
        return -1.0
    else:
        return 2 / (1 + exp(-4.9*x)) - 1

def relu(x):
    return x if x > 0 else 0

# Nodes - acts like an I/O processor
class Node():
    def __init__(self, idx, node_type):
        self.id = idx
        self.type = node_type

        self.cache = []
        self.output = 0
        self.ready = False

        self.input_counter = 0
        self.input_nodes = []

        self.weights = []
        self.output_nodes = []

        self.recurrent_out = []

    def detect_recurrence(self):
        # recurrence with output nodes
        if self.type == Type.OUT:

            # we assume that all output nodes must be recurrent
            if len(self.output_nodes) > 0:
                for node in self.output_nodes:
                    node.input_nodes.remove(self)
                    self.recurrent_out.append(node)
                self.output_nodes = []

        # recurrence with hidden nodes
        elif self.type == Type.HIDDEN:

            # check if node is in inputs
            for node1 in self.output_nodes:
                if node1 in self.input_nodes:

                    # check if that node, has an output to this node
                    for node2 in node1.output_nodes:
                        if node2.id == self.id:
                            # remove input node in-connection
                            self.input_nodes.remove(node1)

                            # shuffle output node to recurrent
                            node1.output_nodes.remove(self)
                            node1.recurrent_out.append(self)

        # recurrence with input nodes
        elif self.type == Type.IN:

            # we assume that all input nodes must be recurrent
            if len(self.input_nodes) > 0:
                for node in self.input_nodes:
                    # shuffle output node to recurrent
                    node.output_nodes.remove(self)
                    node.recurrent_out.append(self)

                    # remove node connection
                    self.input_nodes.remove(node)

    def forward(self):
        # if we've received all inputs
        if len(self.cache) >= self.input_counter and self.ready == False:

            # forward our output to all nodes
            if self.type == Type.BIAS:
                output = 1.0
            else:
                output = sigmoid(sum(self.cache))
            for i, node in enumerate(self.output_nodes):
                node.cache.append(output * self.weights[i])
                node.input_counter += 1

            # store the output for recurrent pass
            self.output = output
            self.cache = []

            # signal that this node has no jobs
            self.ready = True
            self.input_counter = 0

    def r_forward(self):
        # forward the last output to recurrent connections
        l = len(self.output_nodes)
        for i, node in enumerate(self.recurrent_out):
            node.cache.append(self.output * self.weights[l + i])

    def reset(self):
        self.cache = []
        self.output = 0
        self.input_counter = 0
        self.ready = False

# Network - builds the routing between nodes
class Network():
    def __init__(self, genome):
        # neurons
        self.nodes = []
        for neuron in genome.neurons:
            self.nodes.append(Node(neuron.id, neuron.type))

        # connections
        self.connections = genome.connections

    def find_node(self, node_id):
        # TODO: sort when copying from genome,
        #       then index into the array/dict instead.
        for node in self.nodes:
            if node.id == node_id:
                return node

    def build(self):
        # build all neuron connections
        for connection in self.connections:
            if connection.enabled == True:
                # find corresponding node
                input_node = self.find_node(connection.input)
                output_node = self.find_node(connection.output)

                # update connection
                input_node.weights.append(connection.weight)
                input_node.output_nodes.append(output_node)
                output_node.input_nodes.append(input_node)

        # after all connections are done, detect cycles
        for node in self.nodes:
            node.detect_recurrence()

    def forward(self, x):
        # get input and output nodes
        input_nodes = []
        output_nodes = []
        for node in self.nodes:
            if node.type == Type.IN:
                input_nodes.append(node)
            elif node.type == Type.OUT:
                output_nodes.append(node)

        # split x to feed to input neurons
        for i, node in enumerate(input_nodes):
            node.cache.append(x[i])

        # begin forward pass 1 - propagation
        length = len(output_nodes)
        while True:

            # forward propagate some nodes
            for node in self.nodes:
                node.forward()

            # check if the output nodes are ready
            num_ready = 0
            for node in output_nodes:
                if node.ready == True:
                    num_ready += 1
            if num_ready == length:
                break

        # reset node state signals
        for node in self.nodes:
            node.ready = False

        # begin forward pass 2 - recurrence
        for node in self.nodes:
            node.r_forward()

        # return output
        return [x.output for x in output_nodes]

    def reset(self):
        for node in self.nodes:
            node.reset()

    def __str__(self):
        # shortcut
        nodes = self.nodes

        # tabulate headers
        headers = ['ID', 'IN', 'OUT', 'R']

        # all node id's
        node_ids = [x.id for x in nodes]

        # all respective node inputs
        node_inputs = []
        for node in nodes:
            inputs = []
            for node_in in node.input_nodes:
                inputs.append(node_in.id)
            node_inputs.append(inputs)

        # all respective node outputs
        node_outputs = []
        for node in nodes:
            outputs = []
            for node_out in node.output_nodes:
                outputs.append(node_out.id)
            node_outputs.append(outputs)

        # all respective node recurrents
        recurrents = []
        for node in nodes:
            recur = []
            for node_rec in node.recurrent_out:
                recur.append(node_rec.id)
            recurrents.append(recur)

        # swizzle into mega list
        swizzle = []
        for i in range(len(nodes)):
            current_node_info = []
            current_node_info.append([node_ids[i]])
            current_node_info.append(node_inputs[i])
            current_node_info.append(node_outputs[i])
            current_node_info.append(recurrents[i])
            swizzle.append(current_node_info)

        return tabulate(swizzle, headers=headers, tablefmt="fancy_grid")



# Global innovation counter
_Innovation_Counter = 0
_Connection_Cache = []
_Innovation_Cache = []

# Genome - Network Blueprint
class Genome:
    def __init__(self):
        # information
        self.neurons = []
        self.connections = []

        # measurements
        self.network = 0
        self.get_fitness = 0
        
        self.fitness = 0
        self.adjusted_fitness = 0
        self.shared = []

        # logging information
        self.generation_counter = 0
        self.save_path = 0

    def init(self, num_inputs, num_outputs, fitness_function):
        global _Innovation_Counter

        # activate fitness function
        self.get_fitness = fitness_function

        # grow neurons
        for i in range(num_inputs):
            self.neurons.append(Neuron(Type.IN, i))
        for i in range(num_outputs):
            self.neurons.append(Neuron(Type.OUT, num_inputs + i))
        
        # add a bias
        self.neurons.append(Neuron(Type.BIAS, num_inputs + num_outputs))

        # grow connections
        for i in range(num_inputs):
            for j in range(num_outputs):
                self.connections.append(Connection(i, num_inputs + j,
                                                    random.uniform(-1.0, 1.0),
                                                    True,
                                                    (i*num_outputs)+j))

        # update innovation
        _Innovation_Counter = num_inputs + num_outputs + 1

        return self

    def reset(self):
        self.fitness = 0
        self.adjusted_fitness = 0
        self.shared.clear()
        self.network.reset()
        return self

    def find_neuron(self, neuron_id):
        for n in self.neurons:
            if n.id == neuron_id:
                return n

    def find_connection(self, innovation):
        # TODO: a lot of linear searches in the codebase
        for c in self.connections:
            if c.innovation == innovation:
                return c

    def build(self):
        if self.network == 0:
            self.network = Network(self)
            self.network.build()
        return self

    def evaluate(self, counter):
        self.generation_counter = counter
        return self.build().get_fitness(self)

    def save(self, path):
        self.save_path = path

    def load(self, path, direct=""):
        if not os.path.exists(path):
            print("The path specified does not exist.\nAre you sure you've trained?")
            sys.exit()

        # if we have to load a file
        if direct == "":
            # load file
            with open(path, 'r') as f:
                data = f.read()

            # separate from other genomes
            last_char = -1
            for i, c in enumerate(data):
                if last_char == '\n' and c == '}':
                    data = data[:i+1]
                else:
                    last_char = c
        else:
            data = direct

        # get neurons and connections
        args = []
        start_brace = -1
        end_brace = -1
        for i, c in enumerate(data):
            if c == '{':
                start_brace = i + 1
            if c == '}':
                end_brace = i
            if start_brace != -1:
                if end_brace != -1:
                    args.append(data[start_brace:end_brace])
                    start_brace = -1
                    end_brace = -1

        # clean
        for i in range(len(args)):
            args[i] = args[i].replace('\t', '')
            args[i] = [x for x in args[i].split('\n') if len(x) > 0]

        # parse neurons
        for n in args[0]:
            n = n.replace('[', '').replace(']', '').split(',')
            new_neuron = Neuron( Type(int(n[0])), int(n[1]) )
            self.neurons.append(new_neuron)

        # parse connections
        for c in args[1]:
            c = c.replace('[', '').replace(']', '').split(',')
            new_connection = Connection(int(c[0]),
                                        int(c[1]),
                                        float(c[2]),
                                        bool(c[3]),
                                        int(c[4]))
            self.connections.append(new_connection)

        return self

    def _save_to_file(self, save_path="", generation=0, append=False):
        # Make sure no empties
        if self.save_path == 0:
            if save_path == "":
                print("Error: no directory specified to save to.")
                sys.exit()

        # Pathing defaults
        if self.save_path == 0:
            this_path = save_path + "/gen_{}".format(generation) + ".save"
        else:
            this_path = self.save_path

        # Create path if not available
        if not os.path.exists(this_path.split('/')[0]):
            os.makedirs(this_path.split('/')[0])

        # Write/Append mode, for genome/population
        if append == True:
            file_mode = 'a'
        else:
            file_mode = 'w'

        # Get all relevant information
        meta = [self.generation_counter, self.fitness]
        neurons = [[n.type.value, n.id] for n in self.neurons]
        connections = []
        for c in self.connections:
            str_c = [c.input, c.output, c.weight, int(c.enabled), c.innovation]
            connections.append(str_c) 

        # Stringify
        with open(this_path, file_mode) as f:
            print(meta, file=f)
            print('{', file=f)
            print('\t{', file=f)
            for n in neurons:
                print('\t\t{}'.format(n), file=f)
            print('\t}', file=f)
            print('\t{', file=f)
            for c in connections:
                print('\t\t{}'.format(c), file=f)
            print('\t}', file=f)
            print('}', file=f)

    def __str__(self):
        headers = ['Nodes', 'Connections']
        connection_str = [str([x.input, x.output])[1:-1] for x in self.connections]
        for i, connection in enumerate(connection_str):
            if self.connections[i].enabled == True:
                connection_str[i] = connection.replace(',', ' ->')
            else:
                connection_str[i] = connection.replace(',', ' -x')

        swizzle = []
        l_node = len(self.neurons)
        l_con = len(connection_str)
        length = l_node if l_node > l_con else l_con

        for i in range(length):
            a = ''
            b = ''

            if i < l_node:
                a = str(self.neurons[i].id)

            if i < l_con:
                b = connection_str[i]

            swizzle.append([a,b])

        return tabulate(swizzle, headers=headers)

# Creates offspring between two genomes
def crossover(g1, g2):
    child = Genome()
    child.get_fitness = g1.get_fitness

    # find ending of lined up genes
    innovations_1 = [x.innovation for x in g1.connections]
    innovations_2 = [x.innovation for x in g2.connections]
    recurring = list(set(innovations_1).intersection(set(innovations_2)))

    # these lined up genes have equal chance
    for innovation in recurring:
        # flip a coin
        choice = g1 if random.uniform(0, 1) > 0.5 else g2

        # inherent gene
        child.connections.append(choice.find_connection(innovation))

    # now pull disjoint & excess genes
    if g1.fitness != g2.fitness:
        # find more fit excess genes
        if g1.fitness > g2.fitness:
            choice = g1
            innovation_choice = innovations_1
        else:
            choice = g2
            innovation_choice = innovations_2

        # inheret
        remainder = [x for x in innovation_choice if x not in recurring]
        for innovation in remainder:
            child.connections.append(choice.find_connection(innovation))

    else:
        # inherent all excess genes
        remainder_1 = [x for x in innovations_1 if x not in recurring]
        remainder_2 = [x for x in innovations_2 if x not in recurring]
        for innovation in remainder_1:
            child.connections.append(g1.find_connection(innovation))
        for innovation in remainder_2:
            child.connections.append(g2.find_connection(innovation))

    # grab neurons
    if g1.fitness != g2.fitness:
        # inheret more fit nodes
        choice = g1 if g1.fitness > g2.fitness else g2
        child.neurons = choice.neurons[:]
    else:
        # inheret all nodes
        neurons_1 = [x.id for x in g1.neurons]
        neurons_2 = [x for x in [x.id for x in g2.neurons] if x not in neurons_1]
        for neuron_id in neurons_1:
            child.neurons.append(g1.find_neuron(neuron_id))
        for neuron_id in neurons_2:
            child.neurons.append(g2.find_neuron(neuron_id))

    return child

# Mutates all weights with chance
def mutate_weights(g, chance):
    new_connections = []
    for connection in g.connections:
        # roll dice to see if a mutation is made
        if random.uniform(0,1) > (1-chance):
            # find out what type of weight change
            choice = random.uniform(0,1)

            # new number
            if choice < 0.2:
                new_connections.append(Connection(connection.input,
                                connection.output,
                                random.uniform(-1,1),
                                connection.enabled,
                                connection.innovation))

            # sign swap
            elif choice >= 0.2 and choice < 0.4:
                new_connections.append(Connection(connection.input,
                                connection.output,
                                -connection.weight,
                                connection.enabled,
                                connection.innovation))

            # shift
            elif choice >= 0.4 and choice < 0.8:
                shift = random.uniform(-0.3, 0.3)
                new_connections.append(Connection(connection.input,
                                connection.output,
                                connection.weight + (connection.weight * shift),
                                connection.enabled,
                                connection.innovation))

            # activate activate connection
            else:
                new_connections.append(Connection(connection.input,
                                    connection.output,
                                    connection.weight,
                                    True,
                                    connection.innovation))


        # keep current connection
        else:
            new_connections.append(connection)

    # assign genome's new connections
    g.connections = new_connections[:]

# Adds a neuron to genome - 
def add_node(g):
    global _Innovation_Counter
    global _Innovation_Cache
    global _Connection_Cache

    # grow a single neuron
    g.neurons.append(Neuron(Type.HIDDEN, g.neurons[-1].id + 1))

    # pick a connection and disable it
    choice = random.choice(g.connections)
    g.connections.remove(choice)
    g.connections.append(Connection(choice.input,
                    choice.output,
                    choice.weight,
                    False,
                    choice.innovation))

    # check if this connection has been grown before
    c_temp = [choice.input, g.neurons[-1].id]
    if c_temp in _Connection_Cache:
        this_innovation = _Innovation_Cache[_Connection_Cache.index(c_temp)]
    else:
        _Innovation_Counter += 1
        this_innovation = _Innovation_Counter
        _Connection_Cache.append(c_temp)
        _Innovation_Cache.append(_Innovation_Counter)

    # add a connection from input to new node
    g.connections.append(Connection(choice.input,
                    g.neurons[-1].id,
                    1.0,
                    True,
                    this_innovation))

    # check if this connection has been grown before
    c_temp = [g.neurons[-1].id, choice.output]
    if c_temp in _Connection_Cache:
        this_innovation = _Innovation_Cache[_Connection_Cache.index(c_temp)]
    else:
        _Innovation_Counter += 1
        this_innovation = _Innovation_Counter
        _Connection_Cache.append(c_temp)
        _Innovation_Cache.append(_Innovation_Counter)

    # add a connection from new node to output
    g.connections.append(Connection(g.neurons[-1].id,
                    choice.output,
                    choice.weight,
                    True,
                    this_innovation))

# Adds a connection to genome
def add_connection(g):
    global _Innovation_Counter
    global _Innovation_Cache
    global _Connection_Cache

    # find already connected pairs
    connected = [[0] for x in g.neurons]
    for c in g.connections:
        connected[c.input - 1].append([c.input, c.output])
    connected = [x[1:] for x in connected]

    # flatten
    connected = [x for y in connected for x in y]

    # find unconnected pairs
    unconnected = []
    for i in range(len(g.neurons)):
        for j in range(len(g.neurons)):
            unconnected.append([g.neurons[i].id, g.neurons[j].id])
    unconnected = [x for x in unconnected if x not in connected]

    # safety - if fully traversable, leave
    if len(unconnected) == 0:
        return

    # check if this connection has been grown before
    new = random.choice(unconnected)
    if new in _Connection_Cache:
        this_innovation = _Innovation_Cache[_Connection_Cache.index(new)]
    else:
        _Innovation_Counter += 1
        this_innovation = _Innovation_Counter
        _Connection_Cache.append(new)
        _Innovation_Cache.append(_Innovation_Counter)

    # choose 1 at random
    g.connections.append(Connection(new[0],
                    new[1],
                    random.uniform(-1,1),
                    True,
                    this_innovation))

# High level dice-roller for mutations
def mutate(g, weight_chance, structure_chance):
    # roll dice to see if a structural mutation is made
    if random.uniform(0,1) > (1-structure_chance):
        # flip a coin to see what type it is
        if random.uniform(0,1) > 0.5:
            add_node(g)
        else:
            add_connection(g)
    mutate_weights(g, weight_chance)

# Compatibility measure between two genomes
def compare(g1, g2, c1, c2, c3):
    # cache innovations
    cache1 = [x.innovation for x in g1.connections]
    cache2 = [x.innovation for x in g2.connections]

    # get length of larger genome
    l1 = len(cache1)
    l2 = len(cache2)

    N = l1 if l1 > l2 else l2
    smaller_genome = cache2 if l1 > l2 else cache1
    max_smaller_genome = max(smaller_genome)

    # common genes
    common = list(set(cache1) & set(cache2))

    # excess genes
    excess1 = [c for c in cache1 if c > max_smaller_genome]
    excess2 = [c for c in cache2 if c > max_smaller_genome]
    x = c1 * ((len(excess1) + len(excess2)) / N )

    # disjoint genes
    disjoint1 = [c for c in cache1 if c not in common and c not in excess1]
    disjoint2 = [c for c in cache2 if c not in common and c not in excess2]
    y = c2 * ( (len(disjoint1) + len(disjoint2)) / N )

    # average weight difference
    W = []
    for i in range(len(common)):
        w1 = g1.find_connection(common[i]).weight
        w2 = g2.find_connection(common[i]).weight
        W.append(abs(w1 - w2))
    z = c3 * (sum(W) / len(W))

    return x + y + z



# Thread Pool for Genome Evaluations
_pyneat_thread_pool = 0
def init_thread_pool(num_threads):
    global _pyneat_thread_pool
    _pyneat_thread_pool = Pool(num_threads)

# Evaluation Invoker
def invoke_eval(arg):
    return arg[0].evaluate(arg[1])

# Parallel Evaluation Handler
def p_eval(population, counter):
    global _pyneat_thread_pool
    args = [[p,counter] for p in population]
    return _pyneat_thread_pool.map(invoke_eval, args)

# Parallel adjusted fitness calculator
def invoke_adjust(g):
    g.adjusted_fitness = g.fitness / sum(g.shared)
    return g

# Parallel shared term calculator
def invoke_shared(args):
    # unpack
    g = args[0]
    pop = args[1]
    c1 = args[2]
    c2 = args[3]
    c3 = args[4]
    s = args[5]

    for p in pop:
        # calculate difference and share it
        distance = compare(g, p, c1, c2, c3)
        shared = 0 if distance > s else 1
        
        # store values
        g.shared.append(shared)

    return g

# Parallel adjusted fitness handler
def p_adjust(p, c1, c2, c3, s):
    global _pyneat_thread_pool

    # compare distances in parallel
    args = [[g, p, c1, c2, c3, s] for g in p]
    res = _pyneat_thread_pool.map(invoke_shared, args)

    # calculate adjusted fitness in parallel
    return _pyneat_thread_pool.map(invoke_adjust, res)

# GenePool - abstraction for evoling a collection a genomes.
class GenePool:
    def __init__(self, population_size, num_generations, cutoff, constants, sigma_t,
                    logging=0, num_threads=1, path=""):
        # Evolution Params
        self.population_size = population_size
        self.num_generations = num_generations

        # Population Cutoff
        self.cutoff = cutoff

        # Compatibility constants
        self.c1 = constants[0]
        self.c2 = constants[1]
        self.c3 = constants[2]
        self.sigma_t = sigma_t

        # Levels 0 -> 2
        self.logging = logging

        # Parallel Evaluations
        self.parallel = True if num_threads > 1 else False
        if self.parallel == True:
            init_thread_pool(num_threads)
            
        # Internals
        self.population = []

        self.keys = []
        self.species = {}
        self.representatives = {}
        self.species_top = {}
        self.stagnation_counter = {}

        self.last_top = 0

    def init(self, num_inputs, num_outputs, fitness_func):
        # Create Population
        for i in range(self.population_size):
            self.population.append(Genome().init(num_inputs, num_outputs, fitness_func))

    def load(self, path, fitness_func):
        if not os.path.exists(path):
            print("The path specified does not exist.\nAre you sure you've trained?")
            sys.exit()

        # load file
        with open(path, 'r') as f:
            data = f.read()
        args = []

        # separate all genomes
        while len(data) > 0:
            last_char = -1
            found_new_genome = False
            for i, c in enumerate(data):
                if last_char == '\n' and c == '}':
                    args.append(data[:i+1])
                    data = data[i+1:]
                    found_new_genome = True
                    break
                else:
                    last_char = c
            if found_new_genome == False:
                break

        # create and load population
        self.population_size = len(args)
        for i in range(self.population_size):
            g = Genome().load(path=0, direct=args[i])
            g.get_fitness = fitness_func
            self.population.append(g)

    def adjust_fitness(self):
        # easy alias
        p = self.population

        # calculate all sharing values
        pop_size = len(p)
        for i in range(pop_size):
            for j in range(i, pop_size):
                distance = compare(p[i], p[j], self.c1, self.c2, self.c3)
                sharing = 0 if distance > self.sigma_t else 1
                p[i].shared.append(sharing)
                p[j].shared.append(sharing)

        # calculate adjusted fitness
        for g in p:
            g.adjusted_fitness = g.fitness / sum(g.shared)

    def evolve(self, weight_chance, structure_chance, saver=Saver.DISABLED, save_path='saves'):
        global _Innovation_Cache, _Connection_Cache

        # Init check
        if len(self.population) == 0:
            print("Empty population, you might want to")
            print("  -  run GenePool.init()")
            print("  -  run GenePool.load() with a save file")
            print("  -  make sure your population size is a bit higher.")
            sys.exit()

        # Meta: Warn if no saver
        if saver == Saver.DISABLED:
            _fn_lines = inspect.getsourcelines(self.population[0].get_fitness)
            _fn_lines = "".join(_fn_lines[0])
            idx = _fn_lines.find("self.save(")
            if idx != -1:
                snippet = _fn_lines[idx-100:idx+100]
                idx = snippet.find("self.save(")

                # trim to n newlines before
                n_newlines = 0
                for i in range(len(snippet) - (len(snippet) - idx)):
                    if snippet[idx-i] == '\n':
                        if n_newlines < 3:
                            n_newlines += 1
                        else:
                            snippet = snippet[idx-i+1:]

                # trim to n newlines after
                n_newlines = 0
                for i in range(len(snippet) - idx):
                    if snippet[idx+i] == '\n':
                        if n_newlines < 3:
                            n_newlines += 1
                        else:
                            snippet = snippet[:idx+i]

                print("\nWARNING: Detected usage of save with the saver disabled.\n")
                print(snippet)
                sys.exit()

        # Log Experiment
        print(' ' * 3, '-'*35)
        print(' ' * 5, "Population Size         {}".format(self.population_size))
        print(' ' * 5, "Fitness Cutoff          Top {}%".format(int(self.cutoff*100)))
        print(' ' * 5, "Structural Mutation     {}%".format(int(structure_chance*100)))
        print(' ' * 5, "Weight Mutation         {}%".format(int(weight_chance*100)))
        print(' ' * 3, '-'*35,'\n')

        # Evolution
        for generation in range(self.num_generations):
            # Measure Fitness
            generation_avg = 0
            if self.parallel == True:
                result = p_eval(self.population, generation)
                terminations = []
                self.population = []
                for x in result:
                    terminations.append(x[0])
                    self.population.append(x[1])
                    generation_avg += x[1].fitness
                result.clear()
            else:
                terminations = []
                for g in self.population:
                    terminations.append(g.evaluate(generation))
                    generation_avg += g.fitness
            generation_avg /= self.population_size
                        
            # Rank them
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # Logging Stats
            if self.logging == 3:
                pop_min = self.population[-1].fitness

            # Save genome/population
            if saver != Saver.DISABLED:
                if saver == Saver.UNBIASED_GENOME:
                    self.population[0]._save_to_file(save_path=save_path, generation=generation)

                elif saver == Saver.BIASED_GENOME:
                    if self.population[0].fitness >= self.last_top:
                        self.last_top = self.population[0].fitness
                        self.population[0]._save_to_file(save_path=save_path, generation=generation)

                elif saver == Saver.UNBIASED_POPULATION:
                    for g in self.population:
                        g._save_to_file(save_path=save_path, generation=generation, append=True)

                elif saver == Saver.BIASED_POPULATION:
                    if self.population[0].fitness >= self.last_top:
                        self.last_top = self.population[0].fitness
                        for g in self.population:
                            g._save_to_file(save_path=save_path, generation=generation, append=True)

                elif saver == Saver.USER_DEFINED:
                    for g in self.population:
                        if g.save_path != 0:
                            g._save_to_file()

            # If terminate signal present, exit
            for term in terminations:
                if term == True:
                    sys.exit()
            terminations.clear()

            # Eliminate Worst
            self.population = self.population[:int(self.population_size * self.cutoff)]

            # Calculate the adjusted fitness
            if self.parallel == True:
                self.population = p_adjust(self.population, self.c1, self.c2, self.c3, self.sigma_t)
            else:
                self.adjust_fitness()

            # Speciate
            for g in self.population:
                # see if it belongs to a species
                found_species = False
                for key, r in self.representatives.items():
                    distance = compare(g, r, self.c1, self.c2, self.c3)
                    if distance <= self.sigma_t:
                        self.species[key].append(g)
                        found_species = True
                        break

                # else create a new one
                if found_species == False:
                    # init keychain
                    if len(self.keys) == 0:
                        self.keys.append(0)
                    else:
                        self.keys.append(self.keys[-1] + 1)

                    # create new species
                    new_key = str(self.keys[-1])
                    self.species[new_key] = [g]
                    self.representatives[new_key] = g

                    # init champion holder
                    self.species_top[new_key] = g.fitness
                    self.stagnation_counter[new_key] = 0

            # Logging
            if self.logging > 0:
                # calculate padding offset
                offset = ' ' * (len(str(self.num_generations)) - len(str(generation+1)))
                
                # log information
                if self.logging == 1:
                    print('generation {}{} | top: {:.5f}'.format(generation+1, offset, self.population[0].fitness, generation_avg))
                elif self.logging == 2:
                    print('generation {}{} | {} species | top: {:.5f}'.format(generation+1, offset, len(self.species), self.population[0].fitness))
                elif self.logging == 3:
                    print('generation {}{} | {} species | max: {:.3f}  avg: {:.3f}  min: {:.3f}'.format(generation+1, offset, len(self.species), self.population[0].fitness, generation_avg, pop_min))

            # check if a species has stagnated a generation
            for x in self.keys:
                key = str(x)
                if len(self.species[key]) > 0:
                    s = self.species[key]
                    if self.species_top[key] < s[0].fitness:
                        self.species_top[key] = s[0].fitness
                        self.stagnation_counter[key] = 0
                    else:
                        self.stagnation_counter[key] += 1

            # Dead species cleaner
            dead_keys = []
            for x in self.keys:
                key = str(x)
                if len(self.species[key]) == 0:
                    self.species.pop(key)
                    self.representatives.pop(key)
                    self.species_top.pop(key)
                    self.stagnation_counter.pop(key)
                    dead_keys.append(x)
            
            # Stagnation cleaner
            for x in self.keys:
                # Make sure we dont genocide the population
                if x not in dead_keys and len(self.species) > 1:
                    key = str(x)
                    if self.stagnation_counter[key] >= 15:
                        self.species.pop(key)
                        self.representatives.pop(key)
                        self.species_top.pop(key)
                        self.stagnation_counter.pop(key)
                        dead_keys.append(x)

            # Handle dead keys
            for x in dead_keys:
                self.keys.remove(x)
            dead_keys.clear()

            # Select new representatives
            for key, s in self.species.items():
                self.representatives[key] = random.choice(s)

            # Sum all adjusted fitness
            fitness_total = 0
            species_fitness = {}
            for key, s in self.species.items():
                species_total = 0
                for g in s:
                    species_total += g.adjusted_fitness
                species_fitness[key] = species_total
                fitness_total += species_total
            
            # Get proportions for each speciation
            proportions = {}
            current_pop_target = self.population_size - len(self.species)
            for key, x in species_fitness.items():
                proportions[key] = int((x / fitness_total) * current_pop_target)

            # Make sure we don't under-reproduce
            remainder = current_pop_target - sum(proportions.values())
            for _ in range(remainder):
                c = random.choice(self.keys)
                proportions[str(c)] += 1

            # Assign parents
            parents = []
            for key, species_proportion in proportions.items():
                for _ in range(species_proportion):
                    p1 = random.choice(self.species[key])
                    p2 = random.choice(self.population)
                    parents.append(p1)
                    parents.append(p2)

            # Acquire champions
            self.population.clear()
            for key, s in self.species.items():
                self.population.append(s[0].reset())

            # Clear species
            for key in self.keys:
                self.species[str(key)].clear()

            # Create offspring
            i = 0
            while i < len(parents):
                child = crossover(parents[i], parents[i+1])
                mutate(child, weight_chance, structure_chance)
                self.population.append(child)
                i += 2

            # Clear Cache
            _Innovation_Cache.clear()
            _Connection_Cache.clear()
