# Auto generation of advertisement using GPT-2
# create training and test data
import numpy as np
import pandas as pd
import random
import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, TrainingArguments, Trainer

import torch
# from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
# from torch.utils.data import Dataset

# from transformers import GPT2Tokenizer

df_ads = pd.read_csv('JD_ML.csv', encoding='windows-1252')
# df_concise_ads = df.loc[df['conciseness']=="good"]
# print(f'find {len(df_concise_ads)} concise examples')

# remove_cols = [col for col in df_concise_ads.columns if col not in ["id", "title", "abstract", "clean_text"]]
# df_concise_ads.drop(columns=remove_cols, inplace=True)

# df_ads = df_concise_ads
remove_cols = [col for col in df_ads.columns if col not in ["id", "quality", "title", "abstract", "clean_text", "SKILLS", "RESPONSIBILITIES", "REQUIREMENTS", "EXPERIENCE", "QUALIFICATION"]]
df_ads.drop(columns=remove_cols, inplace=True)
print(df_ads.head(5))


# Configurations of global variables
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
MODEL = MODELS[0]

APEX_OPT_LEVEL  = 'O1'
USE_APEX = True
UNFREEZE_LAST_N = 6 #The last N layers to unfreeze for training

SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
                    
MAXLEN = 768  #{768, 1024, 1280, 1600}

TRAIN_SIZE = 0.8

if USE_APEX:
    TRAIN_BATCHSIZE = 4
    BATCH_UPDATE    = 16
else:
    TRAIN_BATCHSIZE = 2
    BATCH_UPDATE    = 32

BATCHSIZE = 4

EPOCHS = 10
LR = 5e-4
EPS  = 1e-8
WARMUP_STEPS = 1e2

SEED = 1

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', 
#                                           bos_token='<|startoftext|>', 
#                                           eos_token='<|endoftext|>', 
#                                           pad_token='<|pad|>')

def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer

tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)

class ADDataset(Dataset):
    
    def __init__(self, df_data, tokenizer, randomize=True):
        self.randomize = randomize
        self.tokenizer = tokenizer 
        self.title     = df_data["title"].tolist()
        self.text      = df_data["clean_text"].tolist()
        self.abstract  = df_data["abstract"].tolist()
        # self.keywords  = df_data["keywords"].tolist()
     
    @staticmethod
    def join_keywords(keywords, randomize=True):
        N = len(keywords)

        #random sampling and shuffle
        if randomize: 
            M = random.choice(range(N+1))
            keywords = keywords[:M]
            random.shuffle(keywords)

        return ','.join(keywords)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # keywords = self.keywords[idx].copy()
        # kw = self.join_keywords(keywords, self.randomize)

        """
        This loop will iterate through each entry in the content of ad corpus.
        For each bit of text it will prepend it with the start of text token,
        then append the end of text token and pad to the maximum length with the 
        pad token. 
        """
        # input = SPECIAL_TOKENS['bos_token'] + self.title[idx] + \
        #         SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token'] + \
        #         self.text[idx] + SPECIAL_TOKENS['eos_token']
            
        input = SPECIAL_TOKENS['bos_token'] + self.title[idx] + SPECIAL_TOKENS['sep_token'] + \
                self.abstract[idx] + SPECIAL_TOKENS['sep_token'] + \
                self.text[idx] + SPECIAL_TOKENS['eos_token']

        encodings_dict = tokenizer(input, truncation=True, max_length=MAXLEN, padding="max_length")   

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        """
        Each iteration then appends either the encoded tensor to a list,
        or the attention mask for that encoding to a list. The attention mask is
        a binary list of 1's or 0's which determine whether the langauge model
        should take that token into consideration or not. 
        """

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}
    

# Split dataframe into training and validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df_ads, train_size=TRAIN_SIZE, random_state=SEED)

# shuffle training dataframes
train_df.sample(frac=1, random_state=SEED)
train_dataset = ADDataset(train_df, tokenizer)
val_dataset = ADDataset(val_df, tokenizer, randomize=False)

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda")
    model.cuda()
    model = model.to(device)
    return model

# tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS
                 )

# - Freeze selective layers:
# - Freeze all layers except last n:
for parameter in model.parameters():
    parameter.requires_grad = False

for i, m in enumerate(model.transformer.h):        
    #Only un-freeze the last n transformer blocks
    if i+1 > 12 - UNFREEZE_LAST_N:
        for parameter in m.parameters():
            parameter.requires_grad = True 

for parameter in model.transformer.ln_f.parameters():        
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():        
    parameter.requires_grad = True

model_dir = f"./output/ad_generation/{MODEL}"
os.makedirs(model_dir, exist_ok=True)

# Fine-tune GPT2 using Trainer
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCHSIZE,
    per_device_eval_batch_size=BATCHSIZE,
    gradient_accumulation_steps=BATCH_UPDATE,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    fp16_opt_level=APEX_OPT_LEVEL,
    warmup_steps=WARMUP_STEPS,    
    learning_rate=LR,
    adam_epsilon=EPS,
    weight_decay=0.01,        
    save_total_limit=1,
    load_best_model_at_end=True,     
)

#---------------------------------------------------#
trainer = Trainer(
    model=model,
    args=training_args,    
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)

#---------------------------------------------------#
# trainer.train()
# trainer.save_model()

# Generating text with Fine-tuned GPT-2 model
tokenizer = get_tokenier(special_tokens=SPECIAL_TOKENS)
model = get_model(tokenizer, 
                  special_tokens=SPECIAL_TOKENS,
                  load_model_path=os.path.join(model_dir, 'pytorch_model.bin'))

# choose the first sample in val dataset for testing
title = val_df.iloc[0]['title']
abstract = val_df.iloc[0]['abstract']
# keywords = ['train', 'lads', 'drinking', 'picture', 'funny', 'instagram']
# kw = ADDataset.join_keywords(keywords, randomize=False)

# prompt = SPECIAL_TOKENS['bos_token'] + title + \
#          SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']

        
prompt = SPECIAL_TOKENS['bos_token'] + title + \
         SPECIAL_TOKENS['sep_token'] + abstract + SPECIAL_TOKENS['sep_token']
print(prompt)

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
generated = generated.to(device)

model.eval()

# # Top-p (nucleus) text generation (10 samples):
# sample_outputs = model.generate(generated, 
#                                 do_sample=True,   
#                                 min_length=50, 
#                                 max_length=MAXLEN,
#                                 top_k=30,                                 
#                                 top_p=0.7,        
#                                 temperature=0.9,
#                                 repetition_penalty=2.0,
#                                 num_return_sequences=10
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#     text = tokenizer.decode(sample_output, skip_special_tokens=True)
#     # a = len(title) + len(','.join(keywords))
#     a = len(title) + len(abstract)
#     print("{}: {}\n\n".format(i+1,  text[a:]))

# Beam-search text generation:
sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                max_length=MAXLEN,                                                      
                                num_beams=5,
                                repetition_penalty=5.0,
                                early_stopping=True,      
                                num_return_sequences=10
                                )

for i, sample_output in enumerate(sample_outputs):
    text = tokenizer.decode(sample_output, skip_special_tokens=True)
    # a = len(title) + len(','.join(keywords))
    a = len(title) + len(abstract)    
    print("{}: {}\n\n".format(i+1,  text[a:]))
    


# tokenizer = get_tokenier()
# model = get_model(tokenizer)
# prompt = title + abstract

# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# device = torch.device("cuda")
# generated = generated.to(device)

# model.eval()
# sample_outputs = model.generate(generated, 
#                                 do_sample=True,   
#                                 max_length=MAXLEN,                                                      
#                                 num_beams=5,
#                                 repetition_penalty=5.0,
#                                 early_stopping=True,      
#                                 num_return_sequences=10
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))