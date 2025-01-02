import torch 
import torch.nn as nn
import torch.nn.functional as F
import textwrap
import matplotlib.pyplot as plt  
from .esm2 import ESM2 
from .create_dataset import esm_alphabet, convert, idx2token  
import torch

esm_model = ESM2()  
device = 'cuda' if torch.cuda.is_available() else 'cpu'  


class AffinityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = esm_model
        self.predictor = nn.Sequential(
            nn.Linear(1280,1)
        )
        
        
    def forward(self, seq, output_type='affinity'):
        res = self.encoder(seq, repr_layers=[33], return_contacts=False)
        rep = res['representations'][33]
        rep = rep.mean(1)
        y = self.predictor(rep) 

        if output_type == 'affinity':
            return y
        elif output_type == 'likelihood':
            return F.softmax(res['logits'], dim=-1)
    
    
    def make_scFv(self, heavy_seq, light_seq):
        return heavy_seq + 'GGGGSGGGGSGGGGS' + light_seq 
    
    
    def get_affinity(self, sequences, batch_size=4):
        if isinstance(sequences, str):
            sequences = [sequences]
        with torch.no_grad():
            seqs = [torch.tensor(convert(seq)) for seq in sequences]
            seqs_tensor = torch.stack(seqs).to(device)
            num_batches = (len(sequences) + batch_size - 1) // batch_size
            predicted_affinities = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sequences))
                batch_tensor = seqs_tensor[start_idx:end_idx]
                outputs_batch = self(batch_tensor)
                batch_affinities = outputs_batch 
                predicted_affinities.extend(batch_affinities)
        if len(predicted_affinities) == 1:
            return predicted_affinities[0]
        else:
            predicted_affinities = torch.cat(predicted_affinities, dim=0)
            return predicted_affinities 
    
    
    def get_embeddings(self, sequences, mode='res', batch_size=4):
        if mode not in ['res', 'seq']:
            print('Wrong mode selected. Available modes are \'res\' or \'seq\'.')
            return 
        if isinstance(sequences, str):
            sequences = [sequences]
        seqs = [torch.tensor(convert(seq)) for seq in sequences]
        with torch.no_grad():
            seqs_tensor = torch.stack(seqs).to(device)
            num_batches = (len(sequences) + batch_size - 1) // batch_size
            embeddings_batch = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sequences))
                batch_tensor = seqs_tensor[start_idx:end_idx]
                outputs_batch = self.encoder(batch_tensor, repr_layers=[33], return_contacts=False)
                batch_embeddings = outputs_batch['representations'][33]
                if mode == 'seq':
                    batch_embeddings = batch_embeddings.mean(dim=1)
                embeddings_batch.extend(batch_embeddings)
        if len(embeddings_batch) == 1:
            return embeddings_batch[0]
        else:
            return embeddings_batch 
    
    
    def get_contact_map(self, input_sequence, mode='VH-VL', plot=True):
        def wrap_text(text, width):
            return "\n".join(textwrap.wrap(text, width))
        if mode not in ['VH-VL', 'scFv', None]: 
            print('Wrong mode selected. Available modes are \'VH-VL\' or \'scFv\'.')
            return 
        seq_txt = wrap_text(input_sequence, 85)  
        with torch.no_grad():
            toks = convert(input_sequence)
            seq = torch.unsqueeze(torch.tensor(toks),0).to(device) 
            outputs_batch =  self.encoder(seq, repr_layers=[33], return_contacts=True) 

        AAs = [idx2token(idx) for idx in toks][1:-1]
        contacts = outputs_batch['contacts'].to('cpu').numpy()
        combined_contacts = contacts[0][1:-1, 1:-1] 

        HC_last_idx = input_sequence.find('GGGGSGGGGSGGGGS')
        LC_length = len(input_sequence)-(HC_last_idx)-15
        LC_first_idx = HC_last_idx + 15 
        LC_last_idx = LC_first_idx+LC_length 
        
        if mode == 'scFv':
            last_idx = AAs.index('<pad>') 
            plt.subplots(1,1, figsize=(30, 30)) 
            contacts = combined_contacts[:last_idx, :last_idx]
            labels = AAs[:last_idx]  
            x_labels = [labels[i]+str(i+1) for i in range(len(labels))]
            im = plt.imshow(contacts, cmap='Blues', aspect='auto', extent=[-0.5, len(labels) - 0.5, -0.5, len(labels) - 0.5], origin="lower")
            plt.xticks(range(len(labels)), x_labels, rotation=90, ha="center", fontsize=10)
            plt.yticks(range(len(labels)), x_labels)
            plt.title(f'Contact Maps for input antibody sequence:\n {seq_txt}', fontsize=18) 
            plt.tight_layout() 
            plt.show() 
            contactss = combined_contacts[:last_idx, :last_idx] 
            
            
        elif mode == 'VH-VL':
            fig, ax = plt.subplots(1,2, figsize=(40, 20))  
            left_top_contacts = combined_contacts[:HC_last_idx, :HC_last_idx]
            labels = AAs[:HC_last_idx] 
            x_labels = [labels[i]+str(i+1) for i in range(len(labels))]
            im = ax[0].imshow(left_top_contacts, cmap='Blues', aspect='auto', extent=[-0.5, len(labels) - 0.5, -0.5, len(labels) - 0.5], origin="lower")
            ax[0].set_xticks(range(HC_last_idx))
            ax[0].set_yticks(range(HC_last_idx))
            ax[0].set_xticklabels(x_labels, rotation=90, ha="center", fontsize=10)   
            ax[0].set_yticklabels(x_labels, fontsize=10)
            ax[0].set_title('Heavy Chain contacts', fontsize=18)
        
            LC_contacts = combined_contacts[LC_first_idx:LC_last_idx, LC_first_idx:LC_last_idx]
            labels = AAs[LC_first_idx:LC_last_idx]  
            x_labels = [labels[i]+str(i+1) for i in range(len(labels))]
            im = ax[1].imshow(LC_contacts, cmap='Blues', aspect='auto', extent=[-0.5, len(labels) - 0.5, -0.5, len(labels) - 0.5], origin="lower")
            ax[1].set_xticks(range(LC_length))
            ax[1].set_yticks(range(LC_length))
            ax[1].set_xticklabels(x_labels, rotation=90, ha="center", fontsize=10)  
            ax[1].set_yticklabels(x_labels, fontsize=10)
            ax[1].set_title('Ligth Chain contacts', fontsize=18)
            plt.suptitle(f'Contact Maps for input antibody sequence:\n {seq_txt}', fontsize=18)
            plt.tight_layout() 
            plt.show() 
            contactss = combined_contacts[:LC_last_idx, :LC_last_idx] 
        
        elif mode == None:
            last_idx = AAs.index('<pad>') 
            contactss = combined_contacts[:LC_last_idx, :LC_last_idx] 
        return contactss 
   