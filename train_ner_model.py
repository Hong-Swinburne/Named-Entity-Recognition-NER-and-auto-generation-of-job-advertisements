import random
import json
import os
import spacy
from spacy.training.example import Example
from json2spacy import convert
import cupy
import numpy as np



# def load_annotation_json_file(filename):
#     with open(filename, "r") as f:
#         data = json.loads(f.read())

#     return data["annotations"]

def seed_everything(seed):
    spacy.util.fix_random_seed()
    cupy.random.seed(seed)
    np.random.seed(seed)

seed_everything(seed = 1)

# combine multiple lines of annotation data in a json file into a list for training or testing
def prepare_annotation_data(annotation_filename, ubiai_json_format=False):
    annotation_data = []
    
    if ubiai_json_format:
        with open(annotation_filename) as f:
            json_data = json.load(f)
        # convert json data from ubiai format to normal format
        for doc in json_data:
            text = doc["document"]
            ent = []
            doc_dict={}
            for anno in doc["annotation"]:
                annnotation = [anno["start"], anno["end"], anno["label"]]
                ent.append(annnotation)
            doc_dict["entities"] = ent
            annotation_data.append([text, doc_dict])

    else:
        with open(annotation_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                annotation = json.loads(line)["annotations"][0]
                annotation_data.append(annotation)
                
    
    return annotation_data

# count the number of each labeled entities
def get_entity_distribution(input_json_data):
    """
    annotation_filename: name of the annotation file in which each line contains the annotated entities of a single doc
    """
    
    labels = {}
    # load all lines in annotation file to a list 
    for text, annot in input_json_data:
        for start, end, entity in annot["entities"]:
            if entity not in labels:
                    labels[entity] = 1  
            else: 
                labels[entity] += 1

    return labels

def train_transformer_model(TRAIN_DATA, epochs):
    
    # create a blank english model
    # source_nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    source_nlp = spacy.load("en_core_web_trf")
    nlp = spacy.blank("en")

     
    if "transformer" not in nlp.pipe_names:
        transformer = nlp.create_pipe("transformer")
        nlp.add_pipe("transformer", last=False, source=source_nlp)
        
    #add ner component to pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        # nlp.add_pipe("ner", last=True)
        nlp.add_pipe("ner", last=True)

    # # load labels to NER
    # for _, annotations in TRAIN_DATA:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])
            
    # print(f'loaded {len(ner.labels)} labels: {ner.labels}')

    #obtain the other components in the pipeline
    # include_components = ["ner"]
    include_components = ["transformer", "ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in include_components]

    # disable other components, only train ner model
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        # optimizer = nlp.create_optimizer()
        
        for ep in range(epochs):
            print ("Starting epoch " + str(ep+1))
            random.shuffle(TRAIN_DATA)        
            losses = {}
            
            for batch in spacy.util.minibatch(TRAIN_DATA, size=128):
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)                    
                    example = Example.from_dict(doc, annotations)

                    # Update the model
                    nlp.update([example], losses=losses, sgd=optimizer, drop=0.2)
            
            # for text, annotations in TRAIN_DATA:
            #     # create Example
            #     doc = nlp.make_doc(text)
            #     example = Example.from_dict(doc, annotations)
            #     # Update the model
            #     nlp.update([example], losses=losses, sgd=optimizer, drop=0.2)
        
            print(losses)
        
    return nlp


def train_vector_model(TRAIN_DATA, epochs):
    
    # create a blank english model
    # source_nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
    source_nlp = spacy.load("en_core_web_sm")
    nlp = spacy.blank("en")

    if "tok2vec" not in nlp.pipe_names:
        tok2vec = nlp.create_pipe("tok2vec")
        nlp.add_pipe("tok2vec", last=False, source=source_nlp)
        
    #add ner component to pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        # nlp.add_pipe("ner", last=True)
        nlp.add_pipe("ner", last=True, source=source_nlp)

    # # load labels to NER
    # for _, annotations in TRAIN_DATA:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])
            
    # print(f'loaded {len(ner.labels)} labels: {ner.labels}')

    #obtain the other components in the pipeline
    # include_components = ["ner"]
    include_components = ["tok2vec", "ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in include_components]

    # disable other components, only train ner model
    with nlp.disable_pipes(*other_pipes):
        # optimizer = nlp.begin_training()
        optimizer = nlp.create_optimizer()
        
        for ep in range(epochs):
            print ("Starting epoch " + str(ep+1))
            random.shuffle(TRAIN_DATA)        
            losses = {}
            
            for batch in spacy.util.minibatch(TRAIN_DATA, size=1024):
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)                    
                    example = Example.from_dict(doc, annotations)

                    # Update the model
                    nlp.update([example], losses=losses, sgd=optimizer, drop=0.2)
            
            # for text, annotations in TRAIN_DATA:
            #     # create Example
            #     doc = nlp.make_doc(text)
            #     example = Example.from_dict(doc, annotations)
            #     # Update the model
            #     nlp.update([example], losses=losses, sgd=optimizer, drop=0.2)
        
            print(losses)
        
    return nlp

def train_baseline_model(TRAIN_DATA, VAL_DATA, epochs, model_path, eval_freq=10, batch_size=128):
    
    # create a blank english model
    nlp = spacy.blank("en")

    #add ner component to pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner", last=True)

    # # load labels to NER
    # for _, annotations in TRAIN_DATA:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])
            
    # print(f'loaded {len(ner.labels)} labels: {ner.labels}')

    #obtain the other components in the pipeline
    include_components = ["ner"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in include_components]

    # disable other components, only train ner model
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        # optimizer = nlp.create_optimizer()
        
        train_loss, val_loss, val_f1, val_recall, val_precision = [], [], [], [], []
        val_max_f1 = -1
        best_model_dir = os.path.join(model_path, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        for ep in range(epochs):
            print ("Starting epoch " + str(ep+1))
            random.seed(1)
            random.shuffle(TRAIN_DATA)
            losses = {}
            eval_losses = {}
            
            for batch in spacy.util.minibatch(TRAIN_DATA, size=batch_size):
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)                    
                    example = Example.from_dict(doc, annotations)

                    # Update the model
                    nlp.update([example], losses=losses, sgd=optimizer, drop=0.2)
            
            # for text, annotations in TRAIN_DATA:
            #     # create Example
            #     doc = nlp.make_doc(text)
            #     example = Example.from_dict(doc, annotations)
            #     # Update the model
            #     nlp.update([example], losses=losses, sgd=optimizer, drop=0.2)
            
            loss = losses["ner"]
            train_loss.append(loss)
            print(f"training loss:{loss}")
            
            # evaluate on val data every xxx epochs
            if (ep+1) % eval_freq == 0:
                # if (ep+1) == eval_freq:
                val_exams = []
                for batch in spacy.util.minibatch(VAL_DATA, size=batch_size):
                    for text, annotations in batch:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        nlp.update([example], losses=eval_losses, sgd=None, drop=0)
                        
                        val_exams.append(Example.from_dict(doc, annotations))

                val_scores = nlp.evaluate(val_exams)
                # keep the model that performance best on val set
                if val_scores["ents_f"] > val_max_f1:
                    val_max_f1 = val_scores["ents_f"]
                    nlp.to_disk(best_model_dir)
                    best_performamce_epoch = ep+1
                    best_f1 = val_max_f1
                    checkpoint_val_loss = eval_losses["ner"]
                
                val_f1.append(val_scores["ents_f"])
                val_recall.append(val_scores["ents_r"])
                val_precision.append(val_scores["ents_p"])
                val_loss.append(eval_losses["ner"])
                loss = eval_losses["ner"]
                f1 = val_scores["ents_f"]
                recall = val_scores["ents_r"]
                precision = val_scores["ents_p"]
                print(f"evaluate on val set, ner_loss: {loss}, f1 score: {f1}, precision: {precision}, recall: {recall}")
                
        print(f'saved model with best performance at epoch {best_performamce_epoch}, val loss:{checkpoint_val_loss}, val f1 score:{best_f1}')
        
    return nlp, train_loss, val_loss, val_f1, val_recall, val_precision

def ner_infer(filename, nlp):
    text = str()
    with open(os.path.join('/fred/oz193/ads_txt', filename), 'r') as f:
        contents = f.readlines()
        for content in contents:
            text = text + content
    print("*"*100)
    print(text)

    doc = nlp(text)

    print("="*100)
    print(f'test file:{filename}')
    print ("found entities:")
    # extract all entities and its labels in the text
    if len(doc.ents) == 0:
        print ("No entities found.")
    else:
        for ent in doc.ents:
            print (ent.text, ent.label_)
        # render entities in Jupyter notebook
        spacy.displacy.render(doc, style="ent", jupyter=True)


# load train data from multiple lines in a json file
# train_data = prepare_annotation_data("train_10.json")   
train_data = prepare_annotation_data("train_100.json", True)
labels = get_entity_distribution(train_data)
print(labels)

# convert train data from json format to spacy binary
# convert(train_data, "./train_100.spacy")

# load train data
# train_data = load_annotation_json_file("training_data.json")

# load test data
# test_data = load_annotation_json_file("test.json")
val_data = prepare_annotation_data("val_20.json", True)
labels = get_entity_distribution(val_data)
print(f"val labels distribution: {labels}")
# convert val data from json format to spacy binary
# convert(val_data, "./val_20.spacy")


# train model
spacy.prefer_gpu()
epochs = 20

Models = ['baseline', 'vector_based', 'transformer']
model = Models[0]
if model == 'baseline':
    model_path = "./output/baseline"
    trained_nlp, train_loss, val_loss, val_f1, val_recall, val_precision = train_baseline_model(train_data, val_data, epochs, model_path, eval_freq=1)
elif model == 'vector_based':
    model_path = "./output/vector_based"
    trained_nlp = train_vector_model(train_data, epochs)
elif model == 'transformer':
    model_path = "./output/transformer"
    trained_nlp = train_transformer_model(train_data, epochs)
    
# save last trained model
trained_nlp.to_disk(model_path)

# draw loss vs epoch curve
import seaborn as sns
import matplotlib.pyplot as plt

def plot_performance_curve(train_loss, val_loss, val_f1, val_recall, val_precision, epochs, model_name):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the loss curve and performance values.
    epochs = list(range(1, epochs + 1))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, train_loss, 'b-o', label="Train Loss")
    ax1.plot(epochs, val_loss, 'r-o', label="Val Loss")
    ax1.set_ylabel('Loss')
    ax1.set_xlabel("Epoch")
    ax1.legend(loc='upper left')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, 'g-x', label="F1 score (Val data)")
    ax2.plot(epochs, val_recall, 'y-x', label="Recall (Val data)")
    ax2.plot(epochs, val_precision, 'm-x', label="Precision (Val data)")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation set performance')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)

    plt.xticks(epochs)
    plt.title("Performance on training and validation data")
    plt.savefig(f"{model_name}_performance.png")
    plt.show()

plot_performance_curve(train_loss, val_loss, val_f1, val_recall, val_precision, epochs, model)

"""
==========================================================================
Model evaluation
==========================================================================
"""
# load model
trained_nlp = spacy.load(os.path.join(model_path, "best_model"))
print('NER model loaded')


# predict NER on test data using trained_nlp
print("="*100)
print ("evaluate trained NER model on test set")


test_exams = []
for text, annotations in test_data:
    doc = trained_nlp.make_doc(text)
    test_exams.append(Example.from_dict(doc, annotations))

scores = trained_nlp.evaluate(test_exams)
print(scores)


"""
==========================================================================
Model inference
==========================================================================
"""
# inference
print("="*100)
print ("inference trained NER model")
# filenames = ['ad_38215946.txt']
filenames = ['ad_38998815.txt', 'ad_38997579.txt', 'ad_38997179.txt']
for filename in filenames:
    ner_infer(filename, trained_nlp)
