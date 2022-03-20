import os
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from spacy.lang.en import English

# get token of a sentence using spacy's tokenizer
def get_token_text(tokenizer, sentence):
    
    # Tokenize a stream of texts
    string = ''
    for doc in tokenizer.pipe(sentence, batch_size=50):
        string = string + str(doc)
        
    return string

def write_token(data_type, sentences, filename, append = False):
    attribute = 'a' if append else 'w'
    exception_tokens = ['.', '..', '...']
    
    with open(os.path.join(data_type, filename), attribute) as f:
        for sent in sentences:
            token = get_token_text(tokenizer, sent)
            print(token)
            print('*'*100)
            if token in exception_tokens:
                pass
            else:
                f.write(token + ' ')
        f.close()


df = pd.read_csv('JD_ML.csv')
ad_id = df['id'].tolist()
text = df['clean_text'].tolist()

ads_num = len(df)
train_num = int(ads_num * 0.8) #298
test_num = ads_num - train_num #75

nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

for i in range(train_num + test_num):
    id = df.iloc[i]['id']
    text = df.iloc[i]['clean_text']
    
    sentences = sent_tokenize(text)
    if i < train_num:
        filename = 'train_'+str(id)+'.txt'
        write_token('train', sentences, filename, append=True)
        
            
    elif i >= train_num & i < (train_num + test_num):
        filename = 'test_'+str(id)+'.txt'
        write_token('test', sentences, filename, append=True)
        

print('output all train and test files')
    