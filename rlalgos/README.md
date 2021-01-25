# RLStructures: rlalgos

The *rlalgos* library is a collection of classical RL algorithms coded using *rlstructures*

To execute an algorithm, use one of the 'run' file in the algorithm directory:
* 'a2c/run_cartpole.py' is a classical cartpole
* 'a2c/run_cartpole_pomdp.py' is a cartpole without car and pole speed as an input. It is used to test recurrent NN implementations
* 'dqn/run_q_cartpole.py' is DQN on cartpole
* etc...


Each algorithm is using Hydra for choosing hyer-parameters (https://github.com/facebookresearch/hydra). The parameters of each algorithm are contained in the 'yaml' file. To execute an algorithm: `python run_file.py -cd script_directory -cn config.yaml`

For instance: `python rlstructures/rlalgos/a2c/run_cartpole.py -cd rlstructures/rlalgos/a2c -cn config.yaml`
