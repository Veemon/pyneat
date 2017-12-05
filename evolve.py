'''
-crossover- TODO

structure -mutations- occur in two ways:

    in -add connection- mutation, a single new connection gene with a random weight is added
    connecting two previously unconnected nodes

    in the -add node- mutation, an existing connection is split and the new node placed where
    the old connection used to be.
        the old connection is disabled
        the new connections are then added
            - the new connection leading into the new node has a weight of 1
            - the new connection leading out of the new node has the weight of the old connection

whenever a new gene appears through structural mutation, a global innovation number is incremented and assigned to that gene

it is possible to ensure that when the same structure arises more than once through independent mutations
in the same generation, each identical mutation is assigned the same innovation number.

when -crossover- occurs, the genes in both genomes with the same innovation numbers are lined up.
these are called matching genes.

genes that do not match are either disjoint or excess, depending on whether they occur within or outside the range of the parents
innovation numbers.

when composing offspring, genes are randomly chosen from either parent at matching genes,
whereas all excess or disjoint genes are always included from the more fit parent.

the number of excess and disjoint genes between a pair of genomes is a natural measure of their compatibility distance.
the more disjoint two genomes are, the less evolutionary history they share, and thus the less compatibile they are.

sigma = c1(E/N) + c2(D/N) + c3*W
sigma is the compatibility difference
c1,c2,c3 are coefficients
E is the number of excess genes
D is the number of disjoint genes
W is the average weight differences of matching genes (including disable genes)
N is the number of genes in the larger genome

this compatibility difference allows for speciation, with a compatibility threshold sigma_t.

in each generation genomes are sequentially placed into species.
each existing species is represented by a random genome inside the species from the previous generation.

a given genome g in the current generation is placed in the first species with which g is compatibile with
the respective genome of that species.
if g is not compatibile with any of the existing species, a new species is created with g as the representative.

explicit fitness sharing is used, where organisms in the same species must share the fitness of their niche.
this is to combat dominant species.

the adjusted fitness for the organism is calculated according to its compatibility distance from every other organism
in the population.

f_i' = f_i / sum( sh( sigma(i,j) ) )
f_i' is the adjusted fitness for organism i
f_i = is the fitness for organism i
sigma(i,j) is the compatibility distance between organisms i and j

sh is the sharing function
    it is set to 0 when the compatibility distance is above the threshold
    otherwise it is set to 1

species then reproduce by first eliminating the lowest performing members from the population.
the entire population is then replaced by the offspring of the remaining organisms in each species.

neat starts out with a uniform population of networks with zero hidden nodes.
all inputs are directly connected to all outputs.
'''

'''
Creates a population of (randomly generated) members
Scores each member of the population based on some goal. This score is called a fitness function.
Selects and breeds the best members of the population to produce more like them
Mutates some members randomly to attempt to find even better candidates
Kills off the rest (survival of the fittest and all), and
Repeats from step 2. Each iteration through these steps is called a generation.
'''
