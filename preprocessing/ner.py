from transformers import pipeline
class NER:
    def __init__(self):
        model_name = "MostafaAhmed98/AraBert-Arabic-NER-CoNLLpp"
        self.ner=pipeline("ner",model=model_name,tokenizer=model_name,aggregation_strategy="simple")
    def remove_sub_entities(self,items):
        result=[]
        for item in sorted(set(items),key=len,reverse=True):
            if not any(item != existing and item in existing for existing in result):
                result.append(item)
        return result
    def extract_entities(self,text):
        entities={"person":[], "location":[], "organization":[],"misc": []
}

        results=self.ner(text)
        for r in results:
            label = r["entity_group"]
            word = r["word"].strip()
            if "##" in word:
                word=word.replace("##", "")
            if len(word) <= 2:
                continue
            if label == "PER":
                entities["person"].append(word)
            elif label == "LOC":
                entities["location"].append(word)
            elif label == "MISC":
                    entities["misc"].append(word)
            elif label == "ORG":
                entities["organization"].append(word)
        entities["person"] = list(set(entities["person"]))
        entities["location"] = list(set(entities["location"]))
        entities["organization"] = list(set(entities["organization"]))
        entities["misc"] = sorted(set(entities["misc"]))

        entities["person"] = self.remove_sub_entities(entities["person"])
        entities["location"] = self.remove_sub_entities(entities["location"])
        entities["organization"] = self.remove_sub_entities(entities["organization"])
        entities["misc"] = self.remove_sub_entities(entities["misc"])
        return entities