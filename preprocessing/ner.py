import torch
from transformers import pipeline

class NER:
    def __init__(self):
        model_name = "MostafaAhmed98/AraBert-Arabic-NER-CoNLLpp"
        
        device = "cpu"
        self.ner = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device
        )

    def remove_sub_entities(self, items):
        result = []
        for item in sorted(set(items), key=len, reverse=True):
            if not any(item != existing and item in existing for existing in result):
                result.append(item)
        return result

    def extract_entities(self, text):
        res = self.extract_entities_batch([text])
        return res[0] if res else {"person": [], "location": [], "organization": [], "misc": []}

    def extract_entities_batch(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []

        batch_results = self.ner(texts, batch_size=len(texts))
        
        if isinstance(batch_results, dict) or (len(batch_results) > 0 and isinstance(batch_results[0], dict)):
            batch_results = [batch_results]

        output = []
        for results in batch_results:
            entities = {"person": [], "location": [], "organization": [], "misc": []}
            for r in results:
                label = r.get("entity_group", "")
                word = r.get("word", "").replace("##", "").strip()
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

            entities["person"] = self.remove_sub_entities(entities["person"])
            entities["location"] = self.remove_sub_entities(entities["location"])
            entities["organization"] = self.remove_sub_entities(entities["organization"])
            entities["misc"] = self.remove_sub_entities(entities["misc"])
            
            output.append(entities)

        return output