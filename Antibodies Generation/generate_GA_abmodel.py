import torch 
import pandas as pd 
from model.create_dataset import esm_alphabet, convert, get_sequence   
from model.affinity_pred_model import AffinityPredictor
from model.utilities import get_model
import argparse 
import random 
import json 


###################################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

model = get_model().to(device) 

seed_scFv = {
    14: 'EVQLVETGGGLVQPGGSLRLSCAASGFTLNSYGISWVRQAPGKGPEWVSVIYSDGRRTFYGDSVKGRFTISRDTSTNTVYLQMNSLRVEDTAVYYCAKGRAAGTFDSWGQGTLVTVSSGGGGSGGGGSGGGGSDVVMTQSPESLAVSLGERATISCKSSQSVLYESRNKNSVAWYQQKAGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDAAVYYCQQYHRLPLSFGGGTKVEIK',
    91: 'EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAENSLYLQMNSLRAEDTALYYCAKVGRGGGYFDYWGQGTLVTVSSGGGGSGGGGSGGGGSQAVLTQPSSLSASPGASVSLTCTLRSGINVGTYRIYWYQQKPGSPPQYLLRYKSDSDKQQGSGVPSRFSGSKDASANAGILLISGLQSEDEADYYCMIWHSSAWVFGGGTKLTVL',
    95: 'EVQLVESGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGRGVIDHWGQGTLVTVSSGGGGSGGGGSGGGGSSSELTQDPAVSVALGQTVRITCEGDSLRYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFQVFFGAGTKLTVL'
} 

seed_scFv_tokenized = {}
for k in seed_scFv.keys():
    seed_scFv_tokenized[k] = torch.tensor(convert(seed_scFv[k])).to(device) 


cdr_positions = {
    14: [(25, 35), (49, 65), (98, 107), (156, 173), (188, 195), (227, 236)],
    91: [(25, 35), (49, 65), (98, 108), (156, 170), (185, 198), (230, 239)],
    95: [(25, 35), (49, 65), (98, 106), (154, 165), (180, 187), (219, 230)]
}

pred_affinity ={14: 0.784029, 91: 1.020438, 95: 1.834988} 
model_pred_affinity ={14: 1.3968, 91: 2.451342, 95: 2.217252}

###################################################################################################


def seq2token_tensor(input_seq):
    return torch.tensor(convert(input_seq)).to(device) 

def remove_static_region(input_seq):
    cdr_sequences = torch.empty((0,), device=device) 
    for cdr in cdr_positions[ANTIBODY_ID]:
        start_idx = cdr[0] + 1
        end_idx = cdr[1] + 1 
        cdr_slice = input_seq[start_idx:end_idx]
        cdr_sequences = torch.cat((cdr_sequences, cdr_slice), dim=0)
    return cdr_sequences

def add_static_region(cdrs):
    seed_seq = seed_scFv_tokenized[ANTIBODY_ID].clone() 
    j = 0
    for cdr in cdr_positions[ANTIBODY_ID]:
        start_idx = cdr[0]+1
        end_idx = cdr[1]+1 
        for i in range(start_idx, end_idx):
            seed_seq[i] = cdrs[j]
            j += 1
    return seed_seq.to(device)   

def get_cdrs(scFv_seq):
    l = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']
    cdrs = {}
    for i,cdr in enumerate(cdr_positions[ANTIBODY_ID]):
        cdrs[l[i]] = scFv_seq[cdr[0]:cdr[1]]
    return cdrs 

def get_num_mutation(seq):
    if len(seq) != len(seed_scFv_tokenized[ANTIBODY_ID]):
        return -1
    return sum(1 for char1, char2 in zip(seq, seed_scFv_tokenized[ANTIBODY_ID]) if char1 != char2) 


def score_sequence(tokenized_sequence_tensor):
    seq = torch.unsqueeze(tokenized_sequence_tensor,0).to(device) 
    pred_aff = model(seq).item()  
    return pred_aff  

###################################################################################################


def get_subs_matrix():
    with open('model/AB-substitution-matrix.json', 'r') as file:
        sub_matrix = json.load(file)
    probability_matrix = {}
    for key, scores in sub_matrix.items():
        total_score = sum(scores.values())
        probabilities = {sub_key: score / total_score for sub_key, score in scores.items()}
        probability_matrix[key] = probabilities 
    
    # Convert the dictionary into a 2D torch tensor
    tokens = list(probability_matrix.keys())   # Extracting all the keys as tokens
    num_tokens = len(tokens)
    matrix_tensor = torch.zeros(num_tokens, num_tokens)  # Initialize a 2D tensor filled with zeros

    for i, row_token in enumerate(tokens):
        for j, col_token in enumerate(tokens):
            if col_token in probability_matrix[row_token]:
                matrix_tensor[i, j] = probability_matrix[row_token][col_token]
    
    return matrix_tensor, [esm_alphabet.index(s) for s in tokens]  

def get_subs_aa(pssm_matrix, aa_tokens, aa_subs, temperature=1.0):
    i = aa_tokens.index(aa_subs) 
    probabilities = (pssm_matrix[i] ** (1 / temperature)).to(device)
    probabilities /= torch.sum(probabilities)
    #plt.plot(probabilities.to('cpu')) 
    idx = torch.multinomial(probabilities, 1).item()
    return aa_tokens[idx] 

###################################################################################################


def create_init_pop():
    cdr_idx = [] 
    for (s,e) in cdr_positions[ANTIBODY_ID]: cdr_idx.extend(list(range(s, e)))
    amino_acids =  ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] 
    new_sequnces = [] 

    for n in range(len(cdr_idx)):   
        seq = list(seed_scFv[ANTIBODY_ID]) 
        picked_positions = random.sample(cdr_idx, n)
        for pos in picked_positions: 
            seq[pos] = random.choice(amino_acids)
        new_sequnces.append(''.join(seq))
    return new_sequnces 

###################################################################################################


def crossover(population):
    population_size, sequence_length = population.size()
    new_population = torch.empty((0, sequence_length), device='cuda')
    for i in range(population_size - 1):
        crossover_point = torch.randint(1, sequence_length, (1,)).item() 
        child1 = torch.cat((population[i][:crossover_point], population[i + 1][crossover_point:]), dim=0) 
        child2 = torch.cat((population[i + 1][:crossover_point], population[i][crossover_point:]), dim=0) 
        new_population = torch.cat((new_population, child1.unsqueeze(0), child2.unsqueeze(0)), dim=0) 
    return new_population.to(device)

def mutate(population, mutation_rate, mut_temperature,  subs_matrix, aa_tokens):
    mutated_population = torch.empty_like(population).to(device)  
    sequence_length = population.size(1)
    for i, sequence in enumerate(population):
        mutated_sequence = torch.empty_like(sequence).to(device) 
        for j in range(sequence_length):
            if torch.rand(1).item() > mutation_rate:
                mutated_sequence[j] = sequence[j]
            else:
                mutated_sequence[j] = get_subs_aa(subs_matrix, aa_tokens, sequence[j], temperature=mut_temperature)
        mutated_population[i] = mutated_sequence
    return mutated_population

def mutate_d(population, d, mut_temperature, subs_matrix, aa_tokens):
    mutated_population = torch.empty_like(population).to(device)  
    for i, sequence in enumerate(population):
        mut_idx = random.sample(range(0, sequence.shape[0]), d)
        for idx in mut_idx:
            sequence[idx] = get_subs_aa(subs_matrix, aa_tokens, sequence[idx], temperature=mut_temperature)
        mutated_population[i] = sequence 
    return mutated_population


def genetic_algorithm(initial_population, n_gen, population_size, d, mut_temperature, subs_matrix, aa_tokens):
    population = initial_population.clone().to(device) 
    for g in range(n_gen):
        prev_gen_pop = population.clone() 
        pop_cross = crossover(population.to(device)) 
        if isinstance(d, int):
            pop_mut = mutate_d(population.to(device), d, mut_temperature, subs_matrix, aa_tokens)
            pop_mut2 = mutate_d(pop_cross.to(device), d, mut_temperature, subs_matrix, aa_tokens)
        elif isinstance(d, float):
            pop_mut = mutate(population.to(device), d, mut_temperature, subs_matrix, aa_tokens)
            pop_mut2 = mutate(pop_cross.to(device), d, mut_temperature, subs_matrix, aa_tokens)
        #pop_cross2 = crossover(pop_mut.to(device)) 
        
        gen_population = torch.cat((prev_gen_pop, pop_cross, pop_mut, pop_mut2), dim=0).to(device)  
        unique_population = torch.unique(gen_population, dim=0).to(device) 
        #print(unique_population.shape)
        scores = torch.tensor([score_sequence(add_static_region(sequence)) for sequence in unique_population]).to(device) 
        idxs = scores.argsort() 
        #print(scores, idxs)
        scores = scores[idxs]
        sequences = unique_population[idxs] 
        print(f"Gen {g+1}  Best = {scores[0]}", flush=True) 
        # Select top 10% of the best sequences and randomly select the rest 90%
        top_prcntg = 0.4 
        top_10_percent = sequences[:int(population_size * top_prcntg)]
        remaining_population = sequences[int(population_size * top_prcntg):]
        random_selection_indices = random.sample(range(len(remaining_population)), int(population_size * (1 - top_prcntg)))
        random_selection = remaining_population[random_selection_indices]
        population = torch.cat((top_10_percent, random_selection), dim=0).to(device)
    best_sequence = sequences[:population_size] 
    best_score = scores[:population_size] 
    return best_sequence.to(device) , best_score.to(device) 


###################################################################################################


def main_ga_abmodel(antibody_id, pop_size, mutation_rate, mut_temperature, n_gen, n_iter, savefile=False):
    global ANTIBODY_ID
    ANTIBODY_ID = antibody_id
    POP_SIZE = pop_size 
    MUTATION_RATE = mutation_rate
    NUM_GENERATION = n_gen
    NUM_ITER = n_iter
    print(f"Running code for Ab-{ANTIBODY_ID}", flush=True) 
    df = pd.read_csv('data/data_with_prediction_properties.csv') 
    improved_df = df[(df['seed_antibody'] == ANTIBODY_ID)] 
    sorted_df = improved_df.sort_values(by=['Pred_affinity']) 
    from_library = sorted_df[:POP_SIZE]['Sequence'].tolist() 
    seq_list = create_init_pop()

    init_pop_list = [remove_static_region(seq2token_tensor(sequence)).to(device) for sequence in from_library]  
    init_pop = torch.stack(init_pop_list, dim=0) 
    
    subs_matrix, aa_tokens = get_subs_matrix() 

    # Initialize population
    generated_sequences = torch.empty((0, init_pop.shape[1]), device=device)
    generated_scores = torch.empty((0), device=device)  
    
    # Run genetic algorithm 
    gen_pop = init_pop.clone().to(device) 
    for _ in range(NUM_ITER):
        gen_pop, gen_scores = genetic_algorithm(gen_pop, NUM_GENERATION, POP_SIZE, MUTATION_RATE, mut_temperature, subs_matrix, aa_tokens)    
        generated_sequences = torch.cat((generated_sequences, gen_pop), dim=0)
        generated_scores = torch.cat((generated_scores, gen_scores), dim=0)
        print(f"Generated {generated_scores.shape[0]} new sequences. Best so far = {gen_scores[0]} ", flush=True)    
    
    
    new_data = []
    for i in range(len(generated_sequences)):
        full_seq_tokens = add_static_region(generated_sequences[i])
        row = {}
        row['Sequence'] = get_sequence(full_seq_tokens) 
        row['Pred_affinity'] = generated_scores[i].item() 
        row['num_mutation'] = get_num_mutation(full_seq_tokens)
        row.update(get_cdrs(row['Sequence']))
        new_data.append(row)
    # Convert generated data to DataFrame and save to CSV
    new_df = pd.DataFrame(new_data) 
    if savefile: 
        new_df.to_csv(f'Ab_{ANTIBODY_ID}_GA_ABModel.csv', index=False) 
    return new_df 

###################################################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run genetic algorithm for antibody optimization.')
    parser.add_argument('antibody_id', type=int, help='Antibody ID')
    parser.add_argument('pop_size', type=int, help='Population size') 
    parser.add_argument('num_mutation', type=int, help='Number of mutation to generate neighbor sequences')
    parser.add_argument('mut_temperature', type=float, help='Mutation temp parameter')
    parser.add_argument('num_gen', type=int, help='Number of generation') 
    parser.add_argument('n_iter', type=int, help='Number of iterations')
    parser.add_argument('savefile', type=bool, help='Save to file')
    args = parser.parse_args()

    main_ga_abmodel(args.antibody_id, args.pop_size, args.num_mutation, args.mut_temperature, args.num_gen, args.n_iter, args.savefile) 
    
    
