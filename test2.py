from collections import namedtuple

type_in = 0
type_hidden = 1
type_out = 2

class Node():
    def __init__(self, idx, type):
        self.id = idx
        self.type = type

        self.input_nodes = []
        self.output_nodes = []

        self.recurrent_out = []

    def new_input(self, input_node):
        self.input_nodes.append(input_node)

    def new_output(self, output_node):
        self.output_nodes.append(output_node)

    def recur_input_node(self, input_node):
        self.input_nodes.remove(input_node)

    def detect_recurrence(self):
        # recurrent output node
        if self.type == type_out:
            if len(self.output_nodes) > 0:
                for node in self.output_nodes:
                    node.recur_input_node(self)
                    self.recurrent_out.append(node)
                self.output_nodes = []

        # recurrent hidden nodes


Connection = namedtuple('Collection',
                        ['input', 'output', 'weight', 'enabled', 'innovation'])

class Agent():
    def __init__(self):
        self.nodes = []
        self.connections = []

    def example_nodes(self):
        self.nodes.append(Node(1,type_in))
        self.nodes.append(Node(2,type_in))
        self.nodes.append(Node(3,type_in))
        self.nodes.append(Node(4,type_out))
        self.nodes.append(Node(5,type_hidden))

    def example_connections(self):
        c1 = Connection(1, 4, 0.7, True, 1)
        c2 = Connection(2, 4, -0.5, False, 1)
        c3 = Connection(3, 4, 0.5, True, 1)
        c4 = Connection(2, 5, 0.2, True, 1)
        c5 = Connection(5, 4, 0.4, True, 1)
        c6 = Connection(1, 5, 0.6, True, 1)
        c7 = Connection(4, 5, 0.6, True, 1)

        self.connections.append(c1)
        self.connections.append(c2)
        self.connections.append(c3)
        self.connections.append(c4)
        self.connections.append(c5)
        self.connections.append(c6)
        self.connections.append(c7)

    def find_node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node

    def connect_nodes(self):
        for connection in self.connections:
            if connection.enabled == True:
                # find corresponding node
                input_node = self.find_node(connection.input)
                output_node = self.find_node(connection.output)

                # update connection
                input_node.new_output(output_node)
                output_node.new_input(input_node)

        # after all connections are done, detect cycles
        for node in self.nodes:
            node.detect_recurrence()

# Create genome
agent = Agent()
agent.example_nodes()
agent.example_connections()

# Create Network
agent.connect_nodes()

for node in agent.nodes:
    # id and input nodes
    print("ID {} | in: ".format(node.id), end='')
    for input_node in node.input_nodes:
        print(input_node.id, end=' ')

    # output nodes
    print("\tout: ", end='')
    for output_node in node.output_nodes:
        print(output_node.id, end=' ')

    # recurrent output nodes
    print("\trec: ", end='')
    for rec_node in node.recurrent_out:
        print(rec_node.id, end=' ')

    print()

# at this point this basic idea is that
# 1) we set all inputs and outputs in the nodes directly, based on the connections
# 2) we then detect recurrent nodes:
#   a) if a node is an output, every output of that node must be recurrent
#   b) if a node is a hidden node, (TODO)
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
