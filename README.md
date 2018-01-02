# pyneat
This is a python implementation of the original NEAT algorithm, formally known as
Neuro-Evolution of Augmenting Topologies. In summary, NEAT is an evolutionary strategy for
optimizing neural networks with non-static structures.  

Although the library was created for use in experimentation with the game of snake,
https://github.com/Veemon/deep-snake, it is a fully functional - user focused library.
  
To get an idea of what I mean by user focused, refer to the example code:  
*examples/input_detector.py*

## Install
To install simply run *install.sh* .

## Basic Usage
The basic workflow is as follows.
Import the library.
```python
from pyneat import pyneat
```

Create a fitness function for the agent, within file scope.
```python
def fitness_function_foo(self):
  print("I am a genome, here is my representation.", self)
  self.fitness = 1.0
  return True, self 
```

Create a Gene Pool, this will be the primary interface to the algorithm.
Upon calling evolve, the evolutionary loop will run depending on the parameters provided.
```python
gene_pool = pyneat.GenePool(...)
gene_pool.init(...)
gene_pool.evolve(...)
```
There are a lot of parameters, so if you are new to the algorithm I do recommend reading 
the original paper here: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf 
