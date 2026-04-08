# Arms

This repository provides code for training an Router and using a Router.

For training, a pickle file with the list of the following structure is needed,

```python
class JudgedRecord(TypedDict):
    image: Image.Image
    question: str
    lang: str
    answer: str | list[str]
    model_name: str
    model_answer: str
    source_dataset: str
    source_id: str
    record_id: str
    correct: bool
```

Where the `image` is an open image in the `PIL` library. You can modify `pkl_data_files` variable to supple your own dataset.

For inference, you should, in addition, provide `acc_judge_mapper` to `routing.py`, which is a mapper of `{key: boolean}`, where `key` should be in the form of `dataset-name_model-name_record-id`.

For cascaded routing, you may refer to the code in `route_example.py`.
