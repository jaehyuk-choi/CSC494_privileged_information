from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '/model-weights/Meta-Llama-3-70B-Instruct/'
tokenizer_path = '/model-weights/Meta-Llama-3-70B-Instruct/'

def get_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager"
    ).to("cuda")
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id  # Ensure distinct IDs

    return tokenizer, model

tokenizer, model = get_model(model_path, tokenizer_path)

prompt = "<s>[ROLE] system [/ROLE] [INST] You are a doctor evaluating patients who may or may not have diabetes. [/INST] \
            [ROLE] user [/ROLE] [INST] A male patient aged 60-64 with BMI 29,Hba1c of 7.7, and high blood pressure says their physical health was not good for 10 out of the last 30 days. Does this patient have diabetes or pre-diabetes? Answer only 'yes' or 'no'.[/INST]"
input_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
print(prompt)

generation = model.generate(
    input_tokens['input_ids'], 
    attention_mask=input_tokens['attention_mask'], 
    max_new_tokens=20, 
    top_p=0.1, 
    do_sample=True, 
    temperature=0.7, 
    top_k=40, 
    output_scores=True, 
    return_dict_in_generate=True, 
    output_attentions=True, 
    output_hidden_states=True
)

print(tokenizer.decode(generation.sequences[0][input_tokens['input_ids'].size(1):]))



# from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
# import torch

# model_path = '/model-weights/Meta-Llama-3-8B-Instruct/'
# tokenizer_path = '/model-weights/Meta-Llama-3-8B-Instruct/'

# def get_model(model_path, 
#              tokenizer_path,
#              debug=True):

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)#LlamaTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
#     tokenizer.add_special_tokens({"pad_token": "<pad>"})
#     model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")#LlamaForCausalLM.from_pretrained(model_path).cuda().requires_grad_(False)
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.eos_token_id = tokenizer.eos_token_id

#     return tokenizer, model

# tokenizer, model = get_model(model_path, tokenizer_path)

# prompt = "<s>[INST] Hello."

# input_tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)

# output = torch.cat(model(input_tokens, output_hidden_states=True).hidden_states, dim=0).detach().cpu() # [Layers, Seq_len, Dim]
# generation = model.generate(input_tokens, max_new_tokens=50, top_p=0.1, do_sample=True, temperature=0.7, top_k=40, output_scores=True, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)

# print(tokenizer.decode(generation.sequences[0][input_tokens.size(1):]))
