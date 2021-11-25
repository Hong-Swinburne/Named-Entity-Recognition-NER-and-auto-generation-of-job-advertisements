# convert Json Annotations to Spacy binary format which was used in Spacy 3.0+
import warnings
import spacy
from spacy.tokens import DocBin


def convert(input_json_data, output_path, lang="en"):
    nlp = spacy.blank(lang)
    db = DocBin()
    total_labels = 0
    missed_labels = 0
    for text, annot in input_json_data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            total_labels += 1
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
                missed_labels += 1
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)
    
    print(f'total labels:{total_labels}, missed labels:{missed_labels}')
    
def convert_ubiai_json(json_list, output_path, lang="en"):
    nlp = spacy.blank(lang)
    db = DocBin()
    total_labels = 0
    missed_labels = 0
    for data in json_list:
        doc = nlp.make_doc(data["document"])
        ents = []
        for annot in data["annotation"]:
            start = annot["start"]
            end = annot["end"]
            label = annot["label"]
            span = doc.char_span(start, end, label)
            total_labels += 1
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
                missed_labels += 1
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)
    
    print(f'total labels:{total_labels}, missed labels:{missed_labels}')