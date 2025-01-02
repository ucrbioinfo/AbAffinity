import torch 
import pandas as pd 
from model.create_dataset import esm_alphabet, convert   
from model.affinity_pred_model import AffinityPredictor
from model.utilities import get_model
import argparse 
import random 
import json 
import math 
import matplotlib.pyplot as plt 

####################################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

model = get_model()

seed_scFv = {
    14: 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSSGGGGSGGGGSGGGGSDVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK',
    91: 'EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVGRGGGYFDYWGQGTLVTVSSGGGGSGGGGSGGGGSQAVLTQPSSLSASPGASVSLTCTLRSGINVGTYRIYWYQQKPGSPPQYLLRYKSDSDKQQGSGVPSRFSGSKDASANAGILLISGLQSEDEADYYCMIWHSSAWVFGGGTKLTVL',
    95: 'EVQLVESGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGRGVIDHWGQGTLVTVSSGGGGSGGGGSGGGGSSSELTQDPAVSVALGQTVRITCEGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVFFGAGTKLTVL'
}

cdrs_region = {
    14: [(25, 35), (49, 65), (98, 107), (156, 173), (188, 195), (227, 236)], 
    91: [(25, 35), (49, 65), (98, 108), (156, 170), (185, 198), (230, 239)],
    95: [(25, 35), (49, 65), (98, 106), (154, 165), (180, 187), (219, 230)]
}

pred_affinity ={14: 0.784029, 91: 1.020438, 95: 1.834988}

model_pred_affinity ={14: 1.3968, 91: 2.451342, 95: 2.217252} 

####################################################################################################

# Add, Remove static region from the sequence for Ab-91 
def remove_static_region(input_seq):
    cdr_sequences = ''
    for cdr in cdrs_region[ANTIBODY_ID]:
        cdr_sequences += input_seq[cdr[0]:cdr[1]] 
    return cdr_sequences

def add_static_region(cdrs):
    j = 0
    seed_seq = list(seed_scFv[ANTIBODY_ID])
    for cdr in cdrs_region[ANTIBODY_ID]: 
        for i in range(cdr[0], cdr[1]):
            seed_seq[i] = cdrs[j]
            j += 1
    return ''.join(seed_seq) 

def get_cdrs(scFv_seq):
    l = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']
    cdrs = {}
    for i,cdr in enumerate(cdrs_region[ANTIBODY_ID]):
        cdrs[l[i]] = scFv_seq[cdr[0]:cdr[1]]
    return cdrs 

def score_sequence(sequence):
    seq = torch.unsqueeze(torch.tensor(convert(sequence)),0).to(device) 
    pred_aff = model(seq).item()  
    return pred_aff  

def get_num_mutation(seq):
    if len(seq) != len(seed_scFv[ANTIBODY_ID]):
        return -1
    return sum(1 for char1, char2 in zip(seq, seed_scFv[ANTIBODY_ID]) if char1 != char2) 

def get_embeddings(sequence_list):
    inputs = sequence_list  
    outputs_list = []
    with torch.no_grad():
        for i in range(0, len(inputs)):
            seq = torch.unsqueeze(torch.tensor(convert(inputs[i])),0).to(device) 
            outputs_batch =  model.encoder(seq, repr_layers=[33], return_contacts=True) 
            outputs_list.append(torch.mean(outputs_batch['representations'][33], dim=1).to('cpu')) 
            print(i+1, end = ", ", flush=True) 
            del seq, outputs_batch  
            torch.cuda.empty_cache() 
    outputs_list = torch.cat(outputs_list).numpy() 
    print('\nEmbeddings shape = ', outputs_list.shape)
    return outputs_list

####################################################################################################

def create_pssm(sequences):
    num_sequences = len(sequences)
    seq_length = len(sequences[0])
    aa_set = set()
    for sequence in sequences:
        aa_set.update(sequence)
    aa_tokens = sorted(list(aa_set))
    
    pssm_matrix = {}
    for i in range(seq_length):
        position_counts = {}
        for aa in aa_tokens:
            count = sum(1 for sequence in sequences if sequence[i] == aa)
            position_counts[aa] = count+1 / num_sequences+1
        pssm_matrix[i] = position_counts
        factor=1.0/sum(pssm_matrix[i].values()) 
        for k in pssm_matrix[i]:
            pssm_matrix[i][k] = pssm_matrix[i][k]*factor
    return pssm_matrix


def get_pssm_matrix():
    df = pd.read_csv('data/data_with_prediction_properties.csv')
    improved_df = df[(df['improve_binding'] == 1) & (df['seed_antibody'] == ANTIBODY_ID)] 
    variable_regions_list = [remove_static_region(x) for x in improved_df['Sequence'].tolist()] 
    psssm_mat = create_pssm(variable_regions_list) 
    return psssm_mat 

def get_pssm_aa(pssm_matrix, i, temperature=1.0):
    probabilities = [x ** (1 / temperature) for x in pssm_matrix[i].values()] 
    probabilities = [x/sum(probabilities) for x in probabilities] 
    idx = random.choices(range(len(probabilities)), weights=probabilities, k=1)  
    return list(pssm_matrix[i].keys())[idx[0]] 

#################################################################################################### 

def mutate(sequence, AB_subs, mutation_rate=0.10, temp=1.0):
    cdr_seq = list(remove_static_region(sequence))                      
    mutated_sequence = ''.join(
        aa if random.random() > mutation_rate else get_pssm_aa(AB_subs, idx, temp) for idx,aa in enumerate(cdr_seq) 
    )
    return add_static_region(mutated_sequence)

def mutate_d(sequence, AB_subs, d=1, temp=1.0):
    cdr_seq = list(remove_static_region(sequence)) 
    mut_idx = random.sample(range(0, len(cdr_seq)), d)  
    for idx in mut_idx:
        cdr_seq[idx] = get_pssm_aa(AB_subs, idx, temp)  
    mutated_cdrs = ''.join(cdr_seq)
    return add_static_region(mutated_cdrs) 

####################################################################################################

def simulated_annealing(initial_solution, initial_temperature, cooling_rate,  num_iterations, d, mut_temp, num_neighbor=50):
    current_solution = initial_solution 
    current_energy = score_sequence(current_solution) 
    print("seed_sequence", current_energy)
    best_solution = current_solution 
    best_energy = current_energy 
    
    temperature = initial_temperature 
    mut_temperature = mut_temp 
    
    subs_matrix = get_pssm_matrix() 
    
    
    better_seqs=[] 
    all_seqs = [] 
    last_5_energies = [] 

    for i in range(num_iterations):
        # find the neighbor sequence to current_solution  
        if isinstance(d, int):
            new_solution = mutate_d(current_solution, subs_matrix, d, mut_temperature)
        elif isinstance(d, float):
            new_solution = mutate(current_solution, subs_matrix, d, mut_temperature)
        new_energy = score_sequence(new_solution) 
        #all_seqs.append((new_solution, new_energy))
        for j in range(num_neighbor-1):
            if isinstance(d, int):
                neighbor_seq = mutate_d(current_solution, subs_matrix, d, mut_temperature)
            elif isinstance(d, float):
                neighbor_seq = mutate(current_solution, subs_matrix, d, mut_temperature)
            neighbor_energy = score_sequence(neighbor_seq)
            #all_seqs.append((neighbor_seq, neighbor_energy))
            if neighbor_energy < new_energy:
                new_solution = neighbor_seq
                new_energy = neighbor_energy 
        
        # Store last 5 energy to see if not improvement  
        last_5_energies.append(new_energy)
        if len(last_5_energies)>5: last_5_energies.pop(0)
        
        # Save the current neighbor 
        all_seqs.append((new_solution, new_energy))
        print("Iteration ", i, "=>", new_energy, flush=True) 
        
        
        # Accept the neighbor under certain conditions
        energy_diff = new_energy - current_energy
        if energy_diff < 0 or random.random() < math.exp(-energy_diff / temperature):
            current_solution = new_solution
            current_energy = new_energy
        
        # Update best solution if applicable 
        if current_energy < best_energy:
            print(current_energy, flush=True)
            better_seqs.append(current_solution)
            best_solution = current_solution
            best_energy = current_energy
            
        # Cool down the temperature
        temperature *= cooling_rate 
        #mut_temperature -=  (i/num_iterations)*(mut_temp-1)  
        if len(last_5_energies) ==5 and len(set(last_5_energies)) == 1:
            temperature = initial_temperature 
            print(f'Temperature reset for SA.', flush=True) 
    
    return best_solution, best_energy, better_seqs, all_seqs  


#################################################################################################### 

def main_sa_pssm(antibody_id, start_seq, n_iter, temperature, cooling_rate, mut_rate, mut_temp, random_seed_value, savefile, num_neighbor=50):
    global ANTIBODY_ID 
    ANTIBODY_ID = antibody_id  
    # Parameters
    if start_seq=='':
        initial_solution = seed_scFv[ANTIBODY_ID]  # Initial solution
    else:
        initial_solution = start_seq 
    initial_temperature = temperature # Initial temperature
    cooling_rate = cooling_rate  # Cooling rate
    num_iterations = n_iter  # Number of iterations
    d = mut_rate # mutation rate/number 
    mut_temperature = mut_temp  # temperature for mutation 
    random.seed(random_seed_value)
    
    print(f" Antibody: {ANTIBODY_ID}\n N_iter:{n_iter} \n Mutation:{d} \n Mut_Temp:{mut_temperature} \n Init_temp: {temperature} \n Cooling_rate: {cooling_rate} \n random seed: {random_seed_value}", flush=True)
    
    
    # Run simulated annealing
    best_solution, best_energy, seq_list, all_seqs = simulated_annealing(initial_solution, initial_temperature, cooling_rate, num_iterations, d, mut_temperature)

    print("Best Solutions: ", seq_list) 
    print("Best Solution:", best_solution) 
    print("Best Energy:", best_energy)

    all_df = pd.DataFrame()
    all_df['Sequence'] = [x for x, y in all_seqs]
    all_df['Pred_affinity'] = [y for x, y in all_seqs]
    all_df['num_mut'] = all_df['Sequence'].apply(get_num_mutation)
    all_df = all_df.drop_duplicates()
    #all_df = all_df[all_df['Pred_affinity']<model_pred_affinity[ANTIBODY_ID]]
    all_df = all_df.sort_values(by="Pred_affinity").reset_index().drop(['index'], axis=1) 
    if savefile:
        all_df.to_csv(f"Ab-{ANTIBODY_ID}-SA-PSSM.csv", index=False) 
    
    return all_df 

#################################################################################################### 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run genetic algorithm for antibody optimization.')
    parser.add_argument('antibody_id', type=int, help='Antibody ID') 
    parser.add_argument('start_seq', type=str,  default='', help='Start Seq') 
    parser.add_argument('n_iter', type=int, help='Number of iterations')
    parser.add_argument('temperature', type=float, help='temp parameter')
    parser.add_argument('cooling_rate', type=float, help='cooling rate')
    parser.add_argument('num_mutation', type=int, help='Number of mutation to generate neighbor sequences')
    parser.add_argument('mut_temperature', type=float, help='Mutation temp parameter')
    parser.add_argument('random_seed_value', type=int, help='Random seed value') 
    parser.add_argument('savefile', type=bool,  default=False, help='save to file ') 
    args = parser.parse_args()

    main_sa_pssm(args.antibody_id, args.start_seq, args.n_iter, args.temperature, args.cooling_rate, args.num_mutation, args.mut_temperature, args.random_seed_value, args.savefile) 
    
