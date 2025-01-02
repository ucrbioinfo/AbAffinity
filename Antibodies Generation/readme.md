This folder contains the script to generate new antibodies from the seed antibodies. 

_Six Approaches to generate - 
                    1. Genetic Algorithm + ABModel substitution matrix
                    2. Genetic Algorithm + PSSM based substitution matrix
                    3. Genetic Algorithm + Random substitution
                    4. Simulated Annealing + ABModel substitution matrix
                    5. Simulated Annealing + PSSM based substitution matrix
                    6. Simulated Annealing + Random substitution_

**Prerequisite:** 
  - this script and `model` folder needs to be in the same directory.

**Run Script:**

**_Approach - 1, 2_**

`script_file.py antibody_id population_size num_mutation mutation_temperature num_generation num_iteration savefile` 

    _Parameters:_ 
      - 'antibody_id', type=int, help='Antibody ID' 
      - 'population_size', type=int, help='Population size'  
      - 'num_mutation', type=int, help='Number of mutation to generate neighbor sequences' 
      - 'mutation_temperature', type=float, help='Mutation temp parameter' 
      - 'num_generation', type=int, help='Number of generation' 
      - 'num_iteration', type=int, help='Number of iterations' 
      - 'savefile', type=bool, help='Save to file' 

**_Approach - 3_**

`script_file.py antibody_id population_size num_mutation num_generation num_iteration savefile` 

    _Parameters:_ 
      - 'antibody_id', type=int, help='Antibody ID' 
      - 'population_size', type=int, help='Population size'  
      - 'num_mutation', type=int, help='Number of mutation to generate neighbor sequences' 
      - 'num_generation', type=int, help='Number of generation' 
      - 'num_iteration', type=int, help='Number of iterations' 
      - 'savefile', type=bool, help='Save to file' 

**_Approach - 4,5,6_**

`script_file.py antibody_id start_seq n_iter temperature cooling_rate num_mutation mut_temperature random_seed_value savefile`  

    _Parameters:_ 
      - 'antibody_id', type=int, help='Antibody ID'  
      - 'start_seq', type=str,  default='', help='Start Seq'  
      - 'n_iter', type=int, help='Number of iterations' 
      - 'temperature', type=float, help='temp parameter' 
      - 'cooling_rate', type=float, help='cooling rate' 
      - 'num_mutation', type=int, help='Number of mutation to generate neighbor sequences' 
      - 'mut_temperature', type=float, help='Mutation temp parameter' 
      - 'random_seed_value', type=int, help='Random seed value'  
      - 'savefile', type=bool,  default=False, help='save to file'



    
