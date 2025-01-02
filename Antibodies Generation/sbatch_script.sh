#!/bin/bash -l

date

#conda activate llm_env 

echo $CUDA_VISIBLE_DEVICES 


echo "Generating Sequences using Genetic Algorithm" 
# antibody_id, pop_size, num_mutation, mut_temperature, num_gen, n_iter, savefile  

# echo "====> ABModel <=====" 

# python generate_GA_abmodel.py 14 50 1 5.0 1000 1 False 
# python generate_GA_abmodel.py 91 50 1 5.0 1000 1 False 
# python generate_GA_abmodel.py 95 50 1 5.0 1000 1 False 

echo "====> PSSM <=====" 
python generate_GA_pssm.py 14 50 1 5.0 1000 1 False 
python generate_GA_pssm.py 91 50 1 5.0 1000 1 False 
python generate_GA_pssm.py 95 50 1 5.0 1000 1 False 

# echo "====> Random <=====" 
# python generate_GA_random.py 14 50 1 1000 1 False 
# python generate_GA_random.py 91 50 1 1000 1 False 
# python generate_GA_random.py 95 50 1 1000 1 False 





# echo "Generating Sequences using Simulated Annealing" 
# antibody_id, pop_size, num_mutation, mut_temperature, num_gen, n_iter, savefile  

# echo "====> ABModel <=====" 
# python generate_SA_ABModel.py 14 '' 1000 500.0 0.95 1 3.0  123 True 
# python generate_SA_ABModel.py 91 '' 1000 500.0 0.95 1 5.0  123 True 
# python generate_SA_ABModel.py 95 '' 1000 500.0 0.95 1 5.0  123 True 

#echo "====> PSSM <=====" 
#  main_sa_pssm(args.antibody_id, args.start_seq, args.n_iter, args.temperature, args.cooling_rate, args.num_mutation, args.mut_temperature, args.random_seed_value, args.savefile) 
# python generate_SA_PSSM.py 14 '' 1000 500.0 0.95 1 3.0  123 True 
# python generate_SA_PSSM.py 91 '' 1000 500.0 0.95 1 5.0  123 True 
# python generate_SA_PSSM.py 95 '' 1000 500.0 0.95 1 5.0  123 True 

# echo "====> Random <=====" 
# python generate_SA_Random.py 14 '' 1000 500.0 0.95 1 3.0  123 True 
# python generate_SA_Random.py 91 '' 1000 500.0 0.95 1 5.0  123 True 
# python generate_SA_Random.py 95 '' 1000 500.0 0.95 1 5.0  123 True 


hostname

echo "ended!" 
