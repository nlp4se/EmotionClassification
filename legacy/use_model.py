import os
import torch
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_predictor():
    # TODO: Change model path in the following line:
    model_path = "models/dataimbalance_class_weight_bce_with_logits_loss_reduction_mean"
    print(f"Loading model from: {model_path} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    emotions = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]
    
    def predict_batch(reviews, max_labels=3, threshold=0.4):
        inputs = tokenizer(reviews, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        results = []
        with torch.no_grad():
            outputs = model(**inputs)
            probs_batch = torch.sigmoid(outputs.logits)
        
        for i in range(len(reviews)):
            probs = probs_batch[i]
            k = min(max_labels, len(emotions))
            
            top_probs, top_indices = torch.topk(probs, k=k)
            
            review_predictions = []
            for prob, idx in zip(top_probs, top_indices):
                if prob.item() >= threshold:
                    review_predictions.append((emotions[idx], round(prob.item(), 3)))
            
            if not review_predictions:
                top_prob, top_idx = torch.max(probs, dim=0)
                review_predictions.append((emotions[top_idx], round(top_prob.item(), 3)))

            results.append({
                "review": reviews[i],
                "emotions": dict(review_predictions)
            })
            
        return results
        
    return predict_batch

# TODO: Write the reviews in the reviews.txt file, write the correct path
reviews_file = "reviews.txt"

if not os.path.exists(reviews_file):
    print(f"'{reviews_file}' file does not exist!! Please create this file and add your reviews.")
    
with open(reviews_file, "r", encoding="utf-8") as f:
    # Read lines, strip whitespace, and ignore empty lines
    sample_reviews = [line.strip() for line in f.readlines() if line.strip()]
    
    if not sample_reviews:
        print(f"There are no reviews to analyze")
        
    try:
        predict_emotions = load_predictor()
        
        print("\n" + "="*50)
        print("🔍 ANALYZING APP REVIEWS")
        print("="*50 + "\n")
        
        batch_results = predict_emotions(sample_reviews)
        
        csv_filename = "predicted_reviews.csv"
        
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(["Review", "Emotion", "Confidence"])
            
            for idx, result in enumerate(batch_results, start=1):
                print(f"Review {idx}: \"{result['review']}\"")
                print(f"Emotions : {result['emotions']}")
                print("-" * 50)
                
                # A review can have multiple predicted emotions, 
                # we write a row for each one.
                for emotion, confidence in result['emotions'].items():
                    writer.writerow([result['review'], emotion, confidence])
                    
        print(f"✅ Results successfully saved to {csv_filename}")
            
    except Exception as e:
        print(f"Exception {e} raised")
