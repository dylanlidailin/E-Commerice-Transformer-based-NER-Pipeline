# E-commerce Aspect Extraction with Fine-Tuned BERT

This project is an end-to-end solution for the eBay Machine Learning Challenge 2025. The objective is to perform **Named Entity Recognition (NER)** on raw German eBay listing titles to extract key product aspects, such as Manufacturer, Part Type, and Compatible Vehicle Model.

This pipeline handles data ingestion, model fine-tuning, batch inference, and submission-file formatting.

## Methodology

This solution treats the problem as a token-level NER task. A `bert-base-german-cased` model is fine-tuned using the `simpletransformers` library, which provides a high-level API for PyTorch-based Transformer models.

The pipeline is split into three distinct phases:

### Phase 1: Data Ingestion & ETL

1. **Load Data (`load_training_data`):** The `Data/Tagged_Titles_Train.tsv` file is loaded into a Pandas DataFrame.

2. **Handle Continuation Tags:** A critical step is using `keep_default_na=False` during loading. This correctly interprets the `""` (empty string) tags as "continuation tags" for multi-word aspects, rather than as `NaN` (missing) values.

3. **Process Data (`process_loaded_data`):** A transformation function aggregates the token-level data into entity-level data. It iterates through each title, "gluing" tokens with continuation tags to the preceding token (e.g., `["Range", "Rover"]` with tags `["Kompatibles_Fahrzeug_Modell", ""]` becomes `(Aspect: "Kompatibles_Fahrzeug_Modell", Value: "Range Rover")`).

### Phase 2: Model Training (Transfer Learning)

1. **Model:** A `bert-base-german-cased` model is used, leveraging Transfer Learning to apply its existing knowledge of the German language to our specific task.

2. **Training Data:** The model is trained on the **raw, token-level data** (5,000 titles). This is crucial, as the model must learn to predict the `""` continuation tag as a valid class, enabling it to identify multi-word aspects.

3. **Training Strategy (`train_ner_model`):** The model is fine-tuned on the full 5,000-title dataset for **3 epochs**. This provides a robust balance between learning the patterns and avoiding overfitting. Training is configured to leverage a GPU (`use_cuda=True`).

4. **Output:** The final, fine-tuned model artifact is saved to the `outputs/` directory.

### Phase 3: Inference & Post-processing

This phase generates the final `quiz_submission.tsv` file for the 25,000-title quiz set.

1. **Load Model (`make_predictions`):** The fine-tuned model from `outputs/` is loaded in `cuda` mode for accelerated inference.

2. **Load Quiz Data:** The `Data/Listing_Titles.tsv` file is loaded, and all records from 5,001 to 30,000 are filtered.

3. **Tokenize & Predict:** Each of the 25,000 titles is tokenized using the challenge's rule (`title.split()`). The `model.predict()` function is then run in batch mode to generate raw, token-level predictions (e.g., `[('Range', 'Kompatibles_Fahrzeug_Modell'), ('Rover', '""')]`).

4. **Post-processing (Aggregation):** The `process_loaded_data` function is re-used to "glue" the raw predicted tokens into final entity-level aspect values.

5. **Post-processing (Business Rules):** A final clean-up filter is applied to enforce the competition's category-specific rules. Any aspect predicted for the wrong category (e.g., `Menge` for a Category 1 item) is re-labeled as `O` (Obscure) to prevent submission errors.

6. **Format Submission:** The final DataFrame is saved to `quiz_submission.tsv` using the exact submission specifications (tab-separated, no header, no index, no quoting).

## How to Run

### 1. Setup

**Data:**
Create a `Data/` directory in the root of the project and place the competition files inside:

```

.
├── Data/
│   ├── Annexure\_updated.pdf
│   ├── Listing\_Titles.tsv
│   └── Tagged\_Titles\_Train.tsv
├── .gitignore
├── ner.py
└── README.md

````

**Installation:**
This project requires a GPU with the CUDA toolkit.
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install libraries
pip install torch
pip install simpletransformers
pip install pandas
pip install tqdm
````

### 2\. Step 1: Train the Model

Set the `RUN_TRAINING` flag to `True` in the `if __name__ == "__main__":` block at the bottom of `ner.py`.

```python
if __name__ == "__main__":
    RUN_TRAINING = True 
    # ...
```

Run the script. This will begin the 3-epoch training process, which will take a significant amount of time, and save the final model to the `outputs/` directory.

```bash
python ner.py
```

### 3\. Step 2: Generate Predictions

Once training is complete, edit `ner.py` and set the `RUN_TRAINING` flag to `False`.

```python
if __name__ == "__main__":
    RUN_TRAINING = False
    # ...
```

Run the script again. It will skip training, load your fine-tuned model from `outputs/`, and run the full prediction and post-processing pipeline on the 25,000 quiz titles.

```bash
python ner.py
```

This will generate the final `quiz_submission.tsv` file in your root directory, ready for submission.

```
```