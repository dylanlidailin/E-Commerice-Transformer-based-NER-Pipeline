import pandas as pd
import os # Imported to check if the file exists
from tqdm import tqdm # Adding a progress bar
import logging # For simpletransformers
from simpletransformers.ner import NERModel, NERArgs # The model
import csv

def load_training_data(filepath="Data/Tagged_Titles_Train.tsv"):
    """
    Loads the NER training data from a TSV file.
    """
    print(f"Attempting to load data from: {filepath}")

    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print(f"Please make sure '{filepath}' exists.")
        return None

    try:
        # Load the TSV file
        train_df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8',
            keep_default_na=False,
            na_values=None
        )
        
        print("Data loaded successfully.")
        
        # Verify that empty strings are preserved in the 'Tag' column
        empty_tags_count = (train_df['Tag'] == '').sum()
        print(f"Found {empty_tags_count} empty strings (continuation tags) in 'Tag' column.")
        
        nan_tags_count = train_df['Tag'].isna().sum()
        if nan_tags_count > 0:
            print(f"Warning: Found {nan_tags_count} NaN values in 'Tag' column. Check data.")
        
        return train_df

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def process_loaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the raw token DataFrame (from loading or prediction)
    to combine continuation tags ("") into full aspect values.
    
    Args:
        df (pd.DataFrame): A DataFrame with "Record Number", "Category", 
                           "Title", "Token", and "Tag" columns.
    
    Returns:
        pd.DataFrame: A DataFrame with "Record Number", "Category", 
                      "Aspect Name", and "Aspect Value" columns.
    """
    print("\nProcessing loaded data to combine continuation tags...")
    
    # List to hold all the processed rows (as dictionaries)
    processed_rows = []
    
    # Group by Record Number to process each title individually
    grouped = df.groupby(["Record Number", "Category", "Title"])
    
    # Using tqdm for a progress bar
    for (record_number, category, title), group in tqdm(grouped, total=len(grouped)):
        
        current_aspect_name = None
        current_aspect_value = [] # Use a list to build multi-token values

        # Iterate through each token in this title
        for _, row in group.iterrows():
            token = str(row['Token']) # Ensure token is a string
            tag = row['Tag']
            
            if tag != "":
                # This is a new aspect tag.
                
                # First, save the PREVIOUS aspect if one was being built
                if current_aspect_name:
                    processed_rows.append({
                        "Record Number": record_number,
                        "Category": category,
                        "Aspect Name": current_aspect_name,
                        "Aspect Value": " ".join(current_aspect_value)
                    })
                
                # Now, start the NEW aspect
                current_aspect_name = tag
                current_aspect_value = [token]
                
            else:
                # This is a continuation tag (tag == "").
                # Append the token to the current aspect value.
                if current_aspect_name:
                    current_aspect_value.append(token)
            
        # After the loop for this title ends, save the last aspect
        if current_aspect_name:
            processed_rows.append({
                "Record Number": record_number,
                "Category": category,
                "Aspect Name": current_aspect_name,
                "Aspect Value": " ".join(current_aspect_value)
            })

    print("Data processing complete.")
    
    # Convert the list of dictionaries into a new DataFrame
    processed_df = pd.DataFrame(processed_rows)
    
    # Re-order columns as specified for submissions
    if not processed_df.empty:
        processed_df = processed_df[["Record Number", "Category", "Aspect Name", "Aspect Value"]]
    
    return processed_df

def train_ner_model(train_df: pd.DataFrame):
    """
    Trains a new NER model using simpletransformers.
    We will train on the FULL 5,000 titles for 3 epochs.

    Args:
        train_df (pd.DataFrame): The raw DataFrame from load_training_data()
    """
    print("\n--- Starting Model Training ---")
    
    # 1. Reformat data for simpletransformers
    train_df_ner = train_df.rename(columns={
        "Record Number": "sentence_id",
        "Token": "words",
        "Tag": "labels"
    })
    
    # Only 3 columns are needed
    train_df_ner = train_df_ner[["sentence_id", "words", "labels"]]

    # 2. Get all unique tag labels
    all_labels = train_df["Tag"].unique().tolist()

    # 3. [UPDATED] Configure the model
    model_args = NERArgs()
    
    # --- SETTINGS ---
    model_args.num_train_epochs = 3  # Train for 3 epochs on the full dataset
    model_args.evaluate_during_training = False
    model_args.save_best_model = False
    # --- END SETTINGS ---
    
    model_args.learning_rate = 1e-4
    model_args.overwrite_output_dir = True
    model_args.output_dir = "outputs/" # The final model will be saved here
    model_args.save_model_every_epoch = False
    model_args.train_batch_size = 32
    
    # Set up logging so we can see the training progress
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # 4. Initialize the NERModel
    print("\nInitializing NERModel (this may download the base model)...")
    model = NERModel(
        "bert",
        "bert-base-german-cased",
        labels=all_labels,
        args=model_args,
        use_cuda=False
    )

    # 5. [UPDATED] Train the model
    print("\nStarting training... This will take some time.")
    # We now train on the full 'train_df_ner' dataset
    model.train_model(
        train_df_ner
    )
    print("--- Model training complete. Model saved to 'outputs/'. ---")


def make_predictions(model_path: str, listings_filepath: str):
    """
    Loads the trained model and makes predictions on the quiz data.

    Args:
        model_path (str): The path to the saved model (e.g., "outputs/")
        listings_filepath (str): Path to the "Listing_Titles.tsv" file.
    
    Returns:
        pd.DataFrame: A DataFrame with the *processed, final* predictions.
    """
    
    # --- PHASE 2, STEP 5: Load Data and Model ---
    print(f"\n--- Starting Prediction Phase ---")
    print(f"Loading trained model from: {model_path}")
    
    # 1. Load the trained model
    model = NERModel(
        "bert",
        model_path, # e.g., "outputs/"
        use_cuda=False
    )
    
    # 2. Load the main Listing_Titles.tsv file
    print(f"Loading listing titles from: {listings_filepath}")
    try:
        listings_df = pd.read_csv(
            listings_filepath,
            sep='\t',
            encoding='utf-8',
            quoting=csv.QUOTE_MINIMAL 
        )
    except Exception as e:
        print(f"Error loading {listings_filepath}: {e}")
        return None
        
    # 3. Filter for the Quiz Data
    quiz_df = listings_df[
        (listings_df["Record Number"] >= 5001) &
        (listings_df["Record Number"] <= 30000)
    ].copy()
    print(f"Filtered for Quiz Data: {len(quiz_df)} records.")

    # --- PHASE 2, STEP 6: Generate Predictions ---
    
    # 4. Tokenize the titles
    print("Tokenizing quiz titles...")
    quiz_titles_tokenized = [str(title).split() for title in quiz_df["Title"]]
    
    # 5. Make predictions
    print("Running model predictions... This will take a while.")
    predictions, raw_outputs = model.predict(quiz_titles_tokenized, split_on_space=False)
    print("Prediction complete.")

    # --- PHASE 3: Formatting and Submission ---
    print("Converting raw predictions to submission format...")
    
    raw_prediction_rows = []
    
    # --- THIS IS THE CORRECTED LOOP ---
    print("Matching predictions to Record Numbers...")
    for record_row_values, title_predictions in tqdm(zip(quiz_df.values, predictions), total=len(predictions)):
        
        record_number = record_row_values[0] # Correctly gets "Record Number"
        category = record_row_values[1]      # Correctly gets "Category Id"
        title = record_row_values[2]         # Correctly gets "Title"
        
        for token_tag_dict in title_predictions:
            token, tag = list(token_tag_dict.items())[0]
            
            raw_prediction_rows.append({
                "Record Number": record_number,
                "Category": category,
                "Title": title,
                "Token": token,
                "Tag": tag
            })
    # --- END OF CORRECTION ---

    # Convert our list of dicts into a DataFrame
    raw_preds_df = pd.DataFrame(raw_prediction_rows)
    
    print("Applying 'process_loaded_data' to glue tokens...")
    # Now we re-use our helper function to "glue" the results
    final_submission_df = process_loaded_data(raw_preds_df)
    
    # --- [NEW CODE BLOCK] ---
    # This is the new clean-up step to fix the submission error
    print("Applying category-specific tag rules to clean up predictions...")
    
    # Define the lists of category-specific tags from Annexure_updated.pdf
    cat_1_only_tags = [
        "Bremsscheiben-Aussendurchmesser", "Bremsscheibenart", "Farbe", 
        "Herstellungsland_Und_-Region", "Material", "Oberflächenbeschaffenheit", 
        "Produktlinie", "Stärke", "Technologie"
    ]
    
    cat_2_only_tags = [
        "Anwendung", "Breite", "Länge", "Menge", 
        "SAE Viskosität", "Zähnezahl"
    ]

    # Find rows where Category is 2 but the tag is Cat 1-only
    # and re-label them as 'O'
    cat_1_mismatch_mask = (final_submission_df['Category'] == 2) & \
                          (final_submission_df['Aspect Name'].isin(cat_1_only_tags))
    final_submission_df.loc[cat_1_mismatch_mask, 'Aspect Name'] = 'O'
    
    # Find rows where Category is 1 but the tag is Cat 2-only (THIS IS YOUR ERROR)
    # and re-label them as 'O'
    cat_2_mismatch_mask = (final_submission_df['Category'] == 1) & \
                          (final_submission_df['Aspect Name'].isin(cat_2_only_tags))
    final_submission_df.loc[cat_2_mismatch_mask, 'Aspect Name'] = 'O'

    num_cleaned_1 = cat_1_mismatch_mask.sum()
    num_cleaned_2 = cat_2_mismatch_mask.sum()
    print(f"Cleaned up {num_cleaned_1} Cat 1 tags predicted for Cat 2 items.")
    print(f"Cleaned up {num_cleaned_2} Cat 2 tags predicted for Cat 1 items (like 'Menge').")
    # --- [END OF NEW CODE BLOCK] ---
    
    
    # 6. Save the final submission file
    submission_filepath = "quiz_submission.tsv"
    print(f"Saving final submission file to: {submission_filepath}")
    
    final_submission_df.to_csv(
        submission_filepath,
        sep='\t',
        header=False,
        index=False,
        quoting=csv.QUOTE_NONE,
        encoding='utf-8'
    )
    
    print("--- Prediction and submission file generation complete! ---")
    return final_submission_df


if __name__ == "__main__":
    
    # --- PHASE 1 (Load) and PHASE 2 (Train) ---
    
    # Set this to True to run the new, 3-epoch training.
    # Set to False to skip training and just run predictions.
    RUN_TRAINING = True 
    
    if RUN_TRAINING:
        print("--- RUN_TRAINING is True: Starting model training... ---")
        TRAIN_FILE_PATH = "Data/Tagged_Titles_Train.tsv"
        # 1. Load data
        training_data = load_training_data(TRAIN_FILE_PATH)
        
        # 2. Train model
        if training_data is not None:
            train_ner_model(training_data)
    else:
        # This is the part that will run now
        print("--- RUN_TRAINING is False: Skipping training. ---")
        print("Looking for existing model...")

    
    # --- PHASE 2 (Predict) and PHASE 3 (Submit) ---
    
    # Define the paths for our model and the quiz data
    # [UPDATED] We are back to using "outputs/" since we aren't saving a "best_model"
    MODEL_DIRECTORY = "outputs/" 
    LISTINGS_FILE = "Data/Listing_Titles.tsv"
    
    # Check if the model and data files exist before trying to predict
    if not os.path.exists(MODEL_DIRECTORY):
        print(f"Error: Model directory not found at {MODEL_DIRECTORY}")
        print("Please set RUN_TRAINING = True and run the script to train the model first.")
    elif not os.path.exists(LISTINGS_FILE):
        print(f"Error: Listings file not found at {LISTINGS_FILE}")
        print("Please make sure 'Listing_Titles.tsv' is in the 'Data' directory.")
    else:
        # If everything looks good, run the full prediction and submission process
        print("\nModel and listings file found. Starting prediction process...")
        final_submission = make_predictions(MODEL_DIRECTORY, LISTINGS_FILE)
        
        # Print the first few lines of the result
        if final_submission is not None:
            print("\n--- Final Submission Head (Example): ---")
            print(final_submission.head())