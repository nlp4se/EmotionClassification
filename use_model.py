import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def select_model_path(models_dir="models"):
    """Helper method to list available models and let the user select one."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Directory '{models_dir}' not found. Make sure you have trained and saved the model first.")
    
    dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not dirs:
        raise FileNotFoundError(f"No model directories found in '{models_dir}'.")
        
    print(f"\nAvailable models in '{models_dir}':")
    for i, d in enumerate(dirs):
        print(f"  {i+1}. {d}")
        
    while True:
        try:
            choice = input("\nSelect a model by number (or press Enter for the latest): ").strip()
            if not choice:
                # Default to the most recently modified
                dir_paths = [os.path.join(models_dir, d) for d in dirs]
                return max(dir_paths, key=os.path.getmtime)
            
            choice_idx = int(choice)
            if 1 <= choice_idx <= len(dirs):
                return os.path.join(models_dir, dirs[choice_idx-1])
            else:
                print(f"Please enter a number between 1 and {len(dirs)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def load_predictor():
    """Loads the model and tokenizer and returns a predict function."""
    model_path = select_model_path()
    print(f"Loading model from: {model_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    emotions = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]
    
    def predict_batch(reviews, max_labels=3, threshold=0.4):
        """
        Takes a list of string reviews and returns a list of results.
        lower threshold (e.g. 0.4) allows more recall.
        """
        inputs = tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        results = []
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.logits shape: (batch_size, num_emotions)
            probs_batch = torch.sigmoid(outputs.logits)
        
        # Process each review in the batch
        for i in range(len(reviews)):
            probs = probs_batch[i]
            k = min(max_labels, len(emotions))
            
            top_probs, top_indices = torch.topk(probs, k=k)
            
            review_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                if prob.item() >= threshold:
                    review_predictions.append((emotions[idx], round(prob.item(), 3)))
            
            # If no emotion passed the threshold, provide the single highest probability emotion
            if not review_predictions:
                top_prob, top_idx = torch.max(probs, dim=0)
                review_predictions.append((emotions[top_idx], round(top_prob.item(), 3)))

            results.append({
                "review": reviews[i],
                "emotions": dict(review_predictions)
            })
            
        return results
        
    return predict_batch

if __name__ == "__main__":
    # Read reviews from reviews.txt
    reviews_file = "reviews.txt"
    
    if not os.path.exists(reviews_file):
        print(f"❌ Error: Could not find '{reviews_file}'. Please create this file and add your reviews.")
        exit(1)
        
    with open(reviews_file, "r", encoding="utf-8") as f:
        # Read lines, strip whitespace, and ignore empty lines
        sample_reviews = [line.strip() for line in f.readlines() if line.strip()]
        
    if not sample_reviews:
        print(f"❌ Error: '{reviews_file}' is empty.")
        exit(1)
        
    try:
        # Load the pipeline
        predict_emotions = load_predictor()
        
        print("\n" + "="*50)
        print("🔍 ANALYZING APP REVIEWS")
        print("="*50 + "\n")
        
        # Get predictions
        batch_results = predict_emotions(sample_reviews)
        
        # Display the results
        for idx, result in enumerate(batch_results, start=1):
            print(f"Review {idx}: \"{result['review']}\"")
            print(f"Emotions : {result['emotions']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Error: {e}")
