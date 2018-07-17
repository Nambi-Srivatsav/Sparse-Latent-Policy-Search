GrouPS
==========

GrouPS algorithm for latent space policy search. It combines reinforcement learning and dimensionality reduction, while also including prior structural knowledge about the task

Dependencies
------------

- All code is written in Python 3.
- Please install 'numpy' and 'scipy' libraries

Description of files
--------------------

Files that should NOT be edited:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
main.py                           |  Starts the program and has GrouPS algorithm
update_equations.py               |  Contains the update equations required by GrouPS algorithm

Files that can be edited:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
configuration.py                  |  Contains parameters required by GrouPS algorithm.
get_samples.py                    |  Code connecting the simulators and GrouPS algorithm.



Usage
--------------------

```python
python main.py
```

It loads up the simulator and starts the training. Displays Iteration deatails on the terminal. Stores 'checkpoint.npy' for every iteration. It contains the distributions learned.


Configuration
--------------------

In **configuration.py** , please check the following variable

```python
load_the_latest_state = True  ## Loads the 'checkpoint.npy'
load_the_latest_state = False ## Does not load the saved state
```

To begin training the task from the start please set the above variable to false.
