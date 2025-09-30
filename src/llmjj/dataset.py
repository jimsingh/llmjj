from dataclasses import dataclass, asdict
from datasets import load_dataset, Dataset

def infer_labels(query: str, passages: dict[str, list], has_answer: bool) -> list[int]:
    default_label = 3 if has_answer else 2
    return [4 if is_selected else default_label for is_selected in passages["is_selected"]]

def _msmarco_to_querydoc(batch: dict) -> dict[str, list]:
    """Process batched MS MARCO rows and yield one record per passage."""

    records = list()

    for query, passages, answers, well_formed_answers in zip(
        batch["query"],
        batch["passages"],
        batch["answers"],
        batch["wellFormedAnswers"],
    ):
        passage_texts = passages["passage_text"]
        
        answer = answers[0] if answers else ""

        well_formed_answer = well_formed_answers[0] if well_formed_answers else ""
        has_well_formed_answer = bool(well_formed_answer)

        labels = infer_labels(query, passages, has_well_formed_answer)

        for p, label in zip(passage_texts, labels):
            query_doc = {'query': query, 'document': p, 'answer': answer, 'label': label}
            records.append(query_doc)
    
    # repack into a dictionary of lists (columnar format) as ds.map expects)
    keys = records[0].keys()
    return {k: [r[k] for r in records] for k in keys}


def load_msmarco_dataset() -> Dataset:
    msmarco_ds = load_dataset("ms_marco", "v2.1", split="train[:1%]") 
    return msmarco_ds.map(
        _msmarco_to_querydoc,
        batched=True,
        remove_columns=msmarco_ds.column_names,
        load_from_cache_file=False,
    )

if __name__ == "__main__":
    ds = load_msmarco_dataset()
    print("msmarco to querydoc samples:")
    for i, item in enumerate(ds):
        print(f"{i}: 'query': '{item['query'][:30]}...', 'doc': '{item['document'][:30]}...', rating': {item['label']}")
        if i >= 5:
            break
