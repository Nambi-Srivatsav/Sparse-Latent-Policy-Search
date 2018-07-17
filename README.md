Sparse Latent Policy Search
==========


Dependencies
------------

- All code is written in Python 3.
- Please install 'numpy' and 'scipy' libararies

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
It loads up the simulator and starts the training. Displays Iteration deatails on the terminal. Stores 'checkpoint.npy' for every iteration.

**configuration.py** has a parmeter called 'load_the_latest_state'. When it is set is True it loads the 'checkpoint.npy', when it is set to False. It starts
training from the beginning.

