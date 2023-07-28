import torch
import numpy as np

# data loader during training, convert dataset to pytorch object
class dataset_obj(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()} # return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)


def MLM_prediction(model, tokenizer, input_text, device='cuda'):
    model.float().eval().to(device);
    input_text = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**input_text).prediction_logits # after prediction

    print(f"Shape of logits: {logits.shape} means [{logits.shape[1]-2}] words in input sentence and the " \
          f"dictionary size is [{logits.shape[-1]}]")
    # retrieve index of [MASK]
    mask_token_index = (input_text.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    output = tokenizer.decode(predicted_token_id)
    print(f'predicted word is: {output}') 
    
    predicted_token_id = logits[0].argmax(axis=-1)
    sentence = tokenizer.decode(predicted_token_id)
    print(f'whole sentence: {sentence}')


def get_top_k_predictions(input_string, tokenizer, model, k=5, device='cuda') -> str:

    tokenized_inputs = tokenizer(input_string, return_tensors="pt")
    outputs = model(tokenized_inputs["input_ids"].to(device)).prediction_logits

    top_k_indices = torch.topk(outputs, k).indices[0]
    decoded_output = tokenizer.batch_decode(top_k_indices)
    mask_token = tokenizer.encode(tokenizer.mask_token)[1:-1]
    mask_index = np.where(tokenized_inputs['input_ids'].numpy()[0]==mask_token)[0][0]

    decoded_output_words = decoded_output[mask_index]
    print(decoded_output_words)

    return decoded_output_words


# helper functions
def prep_param_lists(model):
    ''' Extract FP16 and FP32 of weights'''
    model_params  = [p for p in model.parameters() if p.requires_grad]
    # FP32 master weights
    master_params = [p.detach().clone().float() for p in model_params]
    for p in master_params:
        p.requires_grad = True
    return model_params, master_params # FP16, FP32

def master_params_to_model_params(model_params, master_params):
    ''' updated FP32 master weights-->copy--> back into FP16 weights'''
    for model, master in zip(model_params, master_params):
        model.data.copy_(master.data)

def model_grads_to_master_grads(model_params, master_params):
    ''' FP16 gradients -->copy--> FP32 master gradients'''
    for model, master in zip(model_params, master_params):
        if master.grad is None:
            master.grad = master.data.new(*master.data.size())
        master.grad.data.copy_(model.grad.data)
        
def BN_convert_float(module):
    ''' Convert all BN layers to FP32'''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module
