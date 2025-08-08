#!/usr/bin/env python3
import json
import sys
import os
from rouge import Rouge
import numpy as np
import re

def rouge_score(prediction, ground_truth):
    """Calculate ROUGE-L F1 score between prediction and ground truth"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except:
        return 0.0

def extract_prediction_and_truth_from_api_response(data):
    """Extract prediction and ground truth from OpenAI API response format"""
    try:
        # Get the generated text from the API response
        pred_text = data["result"]["choices"][0]["text"]
        
        # Get the prompt to extract the ground truth
        prompt = data["request"]["prompt"]
        
        # Extract prediction (clean up the generated text)
        pred_text = pred_text.strip()
        # Remove leading newlines and end tokens
        pred_text = pred_text.lstrip('\n').replace('</s>', '').strip()
        # Take only the first line (the summary)
        pred_text = pred_text.split('\n')[0].strip()
        
        # Extract ground truth from prompt
        # Find the pattern: "Summarize the above article in 1 sentence.\n{GROUND_TRUTH}\n\n###"
        truth_match = re.search(r'Summarize the above article in 1 sentence\.\n(.*?)\n\n###', prompt, re.DOTALL)
        if truth_match:
            ground_truth = truth_match.group(1).strip()
        else:
            # Try alternative pattern
            truth_match = re.search(r'Summarize the above article in 1 sentence\.\n(.*?)$', prompt, re.DOTALL | re.MULTILINE)
            if truth_match:
                ground_truth = truth_match.group(1).strip()
            else:
                return None, None
        
        return pred_text, ground_truth
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None, None

def evaluate_file(file_path):
    """Evaluate a JSONL file and return scores"""
    predictions = []
    ground_truths = []
    
    print(f"Evaluating: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                pred, truth = extract_prediction_and_truth_from_api_response(data)
                if pred is not None and truth is not None:
                    predictions.append(pred)
                    ground_truths.append(truth)
                else:
                    print(f"Warning: Could not extract data from line {line_num}")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    if len(predictions) == 0:
        print("No valid predictions found")
        return None
    
    # Calculate scores
    total_score = 0.0
    scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        score = rouge_score(pred, truth)
        scores.append(score)
        total_score += score
    
    avg_score = (total_score / len(predictions)) * 100  # Convert to percentage
    return {
        "average_score": round(avg_score, 2),
        "total_examples": len(predictions),
        "raw_score": total_score / len(predictions),
        "scores": scores
    }

def main():
    # File paths
    base_dir = "/local/ymteng/SqueezeAttention/myResults"
    w_squeeze_file = os.path.join(base_dir, "xsum_wSqueeze.jsonl")
    wo_squeeze_file = os.path.join(base_dir, "xsum_woSqueeze.jsonl")
    
    print("=" * 60)
    print("XSUM Performance Comparison: SqueezeAttention vs Baseline")
    print("=" * 60)
    
    # Evaluate both files
    w_squeeze_results = evaluate_file(w_squeeze_file)
    wo_squeeze_results = evaluate_file(wo_squeeze_file)
    
    if w_squeeze_results is None or wo_squeeze_results is None:
        print("Error: Could not evaluate one or both files")
        return
    
    print(f"\nResults Summary:")
    print(f"{'Method':<25} {'ROUGE-L Score':<15} {'Examples':<10}")
    print("-" * 55)
    print(f"{'With SqueezeAttention':<25} {w_squeeze_results['average_score']:<15} {w_squeeze_results['total_examples']:<10}")
    print(f"{'Without SqueezeAttention':<25} {wo_squeeze_results['average_score']:<15} {wo_squeeze_results['total_examples']:<10}")
    
    # Calculate improvement
    improvement = w_squeeze_results['average_score'] - wo_squeeze_results['average_score']
    if wo_squeeze_results['average_score'] > 0:
        improvement_pct = (improvement / wo_squeeze_results['average_score']) * 100
    else:
        improvement_pct = 0
    
    print(f"\nPerformance Analysis:")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Absolute Improvement':<25} {improvement:+.2f}")
    if wo_squeeze_results['average_score'] > 0:
        print(f"{'Relative Improvement':<25} {improvement_pct:+.2f}%")
    else:
        print(f"{'Relative Improvement':<25} N/A (baseline is 0)")
    
    if improvement > 0.1:  # Small threshold to account for rounding
        print(f"\n✅ SqueezeAttention shows BETTER performance")
    elif improvement < -0.1:
        print(f"\n❌ SqueezeAttention shows WORSE performance")
    else:
        print(f"\n➖ SqueezeAttention shows SIMILAR performance")
    
    # Statistical analysis
    print(f"\nDetailed Statistics:")
    print(f"SqueezeAttention - Mean: {w_squeeze_results['average_score']:.2f}, Std: {np.std(w_squeeze_results['scores']) * 100:.2f}")
    print(f"Baseline - Mean: {wo_squeeze_results['average_score']:.2f}, Std: {np.std(wo_squeeze_results['scores']) * 100:.2f}")
    
    # Show some example predictions
    print(f"\nExample Predictions (first 3):")
    print("-" * 40)
    
    w_squeeze_examples = []
    wo_squeeze_examples = []
    
    # Re-read files to get examples
    with open(w_squeeze_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            pred, truth = extract_prediction_and_truth_from_api_response(data)
            if pred and truth:
                w_squeeze_examples.append((pred, truth))
    
    with open(wo_squeeze_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            pred, truth = extract_prediction_and_truth_from_api_response(data)
            if pred and truth:
                wo_squeeze_examples.append((pred, truth))
    
    for i in range(min(3, len(w_squeeze_examples), len(wo_squeeze_examples))):
        print(f"\nExample {i+1}:")
        print(f"Ground Truth: {w_squeeze_examples[i][1]}")
        print(f"SqueezeAttention: {w_squeeze_examples[i][0]}")
        print(f"Baseline: {wo_squeeze_examples[i][0]}")
        print(f"Scores - Squeeze: {rouge_score(w_squeeze_examples[i][0], w_squeeze_examples[i][1]):.3f}, Baseline: {rouge_score(wo_squeeze_examples[i][0], wo_squeeze_examples[i][1]):.3f}")
    
    # Also check samsum results if available
    print("\n" + "=" * 60)
    print("SAMSUM Performance Comparison (if available)")
    print("=" * 60)
    
    samsum_w_file = os.path.join(base_dir, "samsum_wSqueeze_llama_chat_hf.jsonl")
    samsum_wo_file = os.path.join(base_dir, "samsum_woSqueeze_llama_chat_hf.jsonl")
    
    if os.path.exists(samsum_w_file) and os.path.exists(samsum_wo_file):
        print("SAMSUM files found, evaluating...")
        samsum_w_results = evaluate_file(samsum_w_file)
        samsum_wo_results = evaluate_file(samsum_wo_file)
        
        if samsum_w_results and samsum_wo_results:
            print(f"\nSAMSUM Results:")
            print(f"{'Method':<25} {'ROUGE-L Score':<15} {'Examples':<10}")
            print("-" * 55)
            print(f"{'With SqueezeAttention':<25} {samsum_w_results['average_score']:<15} {samsum_w_results['total_examples']:<10}")
            print(f"{'Without SqueezeAttention':<25} {samsum_wo_results['average_score']:<15} {samsum_wo_results['total_examples']:<10}")
            
            samsum_improvement = samsum_w_results['average_score'] - samsum_wo_results['average_score']
            if samsum_wo_results['average_score'] > 0:
                samsum_improvement_pct = (samsum_improvement / samsum_wo_results['average_score']) * 100
                
                print(f"\nSAMSUM Performance Analysis:")
                print(f"{'Absolute Improvement':<25} {samsum_improvement:+.2f}")
                print(f"{'Relative Improvement':<25} {samsum_improvement_pct:+.2f}%")
    
    # Check if there are JSON files with aggregated results
    print("\n" + "=" * 60)
    print("Pre-computed Results Summary")
    print("=" * 60)
    
    w_json = os.path.join(base_dir, "samsum_wSqueeze_llama_chat_hf.json")
    wo_json = os.path.join(base_dir, "samsum_woSqueeze_llama_chat_hf.json")
    
    if os.path.exists(w_json) and os.path.exists(wo_json):
        with open(w_json, 'r') as f:
            w_data = json.load(f)
        with open(wo_json, 'r') as f:
            wo_data = json.load(f)
        
        print("Pre-computed SAMSUM scores:")
        w_score = w_data.get('samsum', 'N/A')
        wo_score = wo_data.get('samsum', 'N/A')
        print(f"With SqueezeAttention: {w_score}")
        print(f"Without SqueezeAttention: {wo_score}")
        
        if isinstance(w_score, (int, float)) and isinstance(wo_score, (int, float)):
            improvement = w_score - wo_score
            improvement_pct = (improvement / wo_score) * 100 if wo_score > 0 else 0
            print(f"Improvement: {improvement:+.2f} ({improvement_pct:+.2f}%)")

if __name__ == "__main__":
    main() 