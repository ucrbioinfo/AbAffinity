
## **Overview**

`AbAffinity` is a Large Languge Model to predict binding affinity of scFv antibody sequence against SARS-CoV-2 HR2 peptide. It takes input of antibody heavy and light chain sequence, and predicts the binding affinity against a peptide in SARS-CoV-2 HR2 peptide. This peptide is common in all the variants of SARS-CoV-2. 

## **Key Features**

- Predict Binding Affinity: Given the input antiobdy seqeunce, predict binding affinity 

- Antibody Representation: Given the input antiobdy seqeunce, provide embedding of the antibody. The model gives both residue level representation and sequence level representation. 

- Attention Contact Map:  Given the input antibody sequence, AbAffinity will give residue-residue attention maps of the antibody. 

## **Installation**

You can install `AbAffinity` from PyPI:

```bash
pip install AbAffinity
```

If you’d like to install the package directly from the source (for example, if you’ve cloned or downloaded the repository), you can do so using the setup.py file. Make sure that you have the model weights downloaded in the AbAffinity directory.

```bash
git clone https://github.com/fbabd/AbAffinity_SARS-CoV-2.git
cd AbAffinity_SARS-CoV-2
cd src 
pip install .
```

## **Usage**

Here's a quick example to get started:

```python
from AbAffinity import AbAffinity

# Example usage
abmodel=AbAffinity() 


#The model takes complete scFv sequences as input. Heavy and Light chain are connected with a linker sequence. Use make_scFv() method from the model to get the complete scFv seqeunce from heavy chain and light chain sequence.

heavy_seq = 'EVQLVESGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGRGVIDHWGQGTLVTVSS' 
light_seq = 'SSELTQDPAVSVALGQTVRITCEGDSLDYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFEVTFGAGTKLTVL'

scFv_seq = abmodel.make_scFv(heavy_seq, light_seq) 
print(scFv_seq)  # Output: EVQLVESGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGRGVIDHWGQGTLVTVSSGGGGSGGGGSGGGGSSSELTQDPAVSVALGQTVRITCEGDSLDYYYANWYQQKPGQAPILVIYGKNNRPSGIADRFSGSNSGDTSSLIITGAQAEDEADYYCSSRDSSGFEVTFGAGTKLTVL

#Use `get_affinity()` method to get the predicted binding affinity of the antibody sequence. 
#You can pass a list of sequences to get embeddings for all. Make sure that you have enough memory to process the sequences altogether. You can tune the batch size for this purpose. Example: `model.get_affinity(list_sequences, batch_size=16)`. Default batch_size is 4. 

pred_affinity = abmodel.get_affinity(scFv_seq)
print(pred_affinity) # Output: tensor([3.1595]) 


# Use `get_embeddings()` method to get the embeddings for input sequences. Use `mode='res'` to get residue wise embeddings, and `mode='seq'` will give seqeunce embedding. 
# You can pass a list of sequences to get embeddings for all. Make sure that you have enough memory to process the sequences altogether. You can tune the batch size for this purpose. Example: `model.get_embeddings(list_sequences, mode='seq', batch_size=16)`. Default batch_size is 4.

res_emb = abmodel.get_embeddings(scFv_seq, mode='res')
print(res_emb.shape)  # Output: torch.Size([258, 1280])
 
seq_emb = abmodel.get_embeddings(scFv_seq, mode='seq')
print(seq_emb.shape) # Output: torch.Size([1280]) 

# Use  `get_contact_map()` method to get the contact maps of the given antibody sequence. It will return a matrix of shape `L x L` where `L` is the length of input sequence. Each value in the matrix represents the contact weight between two residue in the sequence.  
# Use `mode='VH-VL'` if you want to plot the contacts for heavy chain and light chain separately, and `mode='scFv'` to plot single contacts for the entire scFv sequence. 

contacts = abmodel.get_contact_map(scFv_seq, mode = 'scFv')
print(contacts.shape) # Output: contact map figure,  (240, 240)

```


## **License**

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yourusername/AbAffinity/blob/main/LICENSE) file for details.

## **Acknowledgments**