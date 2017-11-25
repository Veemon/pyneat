from collections import namedtuple
from tabulate import tabulate
import sys

type_in = 0
type_hidden = 1
type_out = 2

class Node():
    def __init__(self, idx, type):
        self.id = idx
        self.type = type

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
        if self.type == type_out:

            # we assume that all output nodes must be recurrent
            if len(self.output_nodes) > 0:
                for node in self.output_nodes:
                    node.input_nodes.remove(self)
                    self.recurrent_out.append(node)
                self.output_nodes = []

        # recurrence with hidden nodes
        elif self.type == type_hidden:

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
        elif self.type == type_in:

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
        if len(self.cache) >= self.input_counter:

            # forward our output to all nodes
            # TODO big ol sigmoid or relu
            output = sum(self.cache)
            for i, node in enumerate(self.output_nodes):
                node.cache.append(output * self.weights[i])
                node.input_counter += 1

            # store the output for recurrent pass
            self.output = output

            # signal that this node has no jobs
            self.ready = True

    def r_forward(self):
        pass

Connection = namedtuple('Collection',
                        ['input', 'output', 'weight', 'enabled', 'innovation'])

class Network():
    def __init__(self):
        self.nodes = []
        self.connections = []

    def example_nodes(self):
        self.nodes.append(Node(1,type_in))
        self.nodes.append(Node(2,type_in))
        self.nodes.append(Node(3,type_out))
        self.nodes.append(Node(4,type_out))

    def example_connections(self):
        # base connections
        self.connections.append(Connection(1,3,0.5,True,1))
        self.connections.append(Connection(1,4,0.5,True,1))
        self.connections.append(Connection(2,3,0.5,True,1))
        self.connections.append(Connection(2,4,0.5,True,1))

        # recurrent connections

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

        print('Initial Build')
        print(self)

        # after all connections are done, detect cycles
        for node in self.nodes:
            node.detect_recurrence()

        print('After recurrence')
        print(self)

    def forward(self, x):
        # get input and output nodes
        input_nodes = []
        output_nodes = []
        for node in self.nodes:
            if node.type == type_in:
                input_nodes.append(node)
            elif node.type == type_out:
                output_nodes.append(node)

        # split x to feed to input neurons
        for i, node in enumerate(input_nodes):
            node.cache.append(x[i])

        # begin forward pass 1 - propagation
        while True:

            # forward propagate some nodes
            for node in self.nodes:
                node.forward()

            # check if the output nodes are ready
            num_ready = 0
            for node in output_nodes:
                if node.ready == True:
                    num_ready += 1
            if num_ready == len(output_nodes):
                break

        # begin forward pass 2 - recurrence

        # return output
        return [sum(x.cache) for x in output_nodes]

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

# Create genome
net = Network()
net.example_nodes()
net.example_connections()

# Create Network
net.build()
print(net.forward([1,1]))

# at this point the basic idea is that
# 1) we set all inputs and outputs in the nodes directly, based on the connections
# 2) we then detect recurrent nodes:
#   a) if a node is an output, every output of that node must be recurrent
#   b) if a node is a hidden node, swap the two associated nodes contents
#   c) if a node is an input, we assume it cannot recur
# 3) after detecting recurrent nodes,
#   a) we place the output nodes in the recurrent output nodes

# this essentially creates the network, from here to perform a feed forward pass
# 0) depack input into respective input nodes
# 1) cycle through each node
# ~ Forward Pass 1/2
# 2) if all the nodes inputs are satisfied, summate the node cache and send this
#    value to the the nodes outputs
# ~ Forward Pass 2/2
# 3) forward the nodes exit value to all recurrent outputs
