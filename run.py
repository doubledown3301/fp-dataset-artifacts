import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import transformers
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

NUM_PREPROCESSING_WORKERS = 2

def print_title(title):
    print("=" * 20)
    print(title)
    print("=" * 20)

def print_footer():
    print("=" * 20)

def check_versions():
    print_title("Version and GPU Information")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:         {torch.version.cuda}")
        print(f"GPU device count:     {torch.cuda.device_count()}")
        print(f"Current GPU device:   {torch.cuda.current_device()}")
        print(f"GPU device name:      {torch.cuda.get_device_name(0)}")
    print_footer()

def analyze_overlap(predictions_file, output_file=None):
    print_title("Lexical Overlap")
    
    # Load predictions
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    print(f"Loaded {len(predictions)} predictions\n")
    
    # Calculate word overlap
    def word_overlap(premise, hypothesis):
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if len(h_words) == 0:
            return 0.0
        return len(p_words & h_words) / len(h_words)
    
    for pred in predictions:
        pred['overlap'] = word_overlap(pred['premise'], pred['hypothesis'])
        pred['correct'] = (pred['label'] == pred['predicted_label'])
    
    df = pd.DataFrame(predictions)
    
    # Overall accuracy
    overall_acc = df['correct'].mean()
    print(f"Overall Accuracy: {overall_acc:.4f} ({df['correct'].sum()}/{len(df)})\n")
    
    # Accuracy by label
    print("Accuracy by Label:")
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    for label in [0, 1, 2]:
        subset = df[df['label'] == label]
        acc = subset['correct'].mean()
        avg_overlap = subset['overlap'].mean()
        print(f"  {label_names[label]:13s}: {acc:.4f} (avg overlap: {avg_overlap:.3f})")
    
    # Accuracy by overlap quartiles
    print("\nAccuracy by Overlap Quartile:")
    df['overlap_quartile'] = pd.qcut(df['overlap'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], duplicates='drop')
    
    quartile_results = []
    for quartile in df['overlap_quartile'].unique():
        subset = df[df['overlap_quartile'] == quartile]
        acc = subset['correct'].mean()
        min_overlap = subset['overlap'].min()
        max_overlap = subset['overlap'].max()
        count = len(subset)
        print(f"  {quartile:11s}: {acc:.4f} ({min_overlap:.2f}-{max_overlap:.2f} overlap, n={count})")
        quartile_results.append({
            'quartile': quartile,
            'accuracy': acc,
            'min_overlap': min_overlap,
            'max_overlap': max_overlap,
            'count': count
        })
    
    # Check for artifact (accuracy gap between high/low overlap)
    low_overlap_acc = df[df['overlap_quartile'] == 'Q1 (Low)']['correct'].mean()
    high_overlap_acc = df[df['overlap_quartile'] == 'Q4 (High)']['correct'].mean()
    gap = high_overlap_acc - low_overlap_acc
    
    print(f"\nOverlap Artifact Analysis:")
    print(f"  High overlap (Q4) accuracy: {high_overlap_acc:.4f}")
    print(f"  Low overlap (Q1) accuracy:  {low_overlap_acc:.4f}")
    print(f"  Accuracy gap:                {gap:+.4f} ({gap*100:+.2f} percentage points)")
    
    if abs(gap) > 0.02:
        print(f"  ⚠️  Significant overlap artifact detected!")
    else:
        print(f"  ✓ Minimal overlap artifact")
    
    print("=" * 20)
    
    # Save results if output file specified
    if output_file:
        results = {
            'overall_accuracy': float(overall_acc),
            'accuracy_by_label': {
                label_names[i]: float(df[df['label'] == i]['correct'].mean())
                for i in range(3)
            },
            'accuracy_by_quartile': [
                {
                    'quartile': str(qr['quartile']),
                    'accuracy': float(qr['accuracy']),
                    'min_overlap': float(qr['min_overlap']),
                    'max_overlap': float(qr['max_overlap']),
                    'count': int(qr['count'])
                }
                for qr in quartile_results
            ],
            'overlap_gap': float(gap),
            'artifact_detected': bool(abs(gap) > 0.02)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return df


def prepare_dataset_nli_hypothesis_only(examples, tokenizer, max_length):
    tokenized = tokenizer(
        examples['hypothesis'],
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    tokenized['label'] = examples['label']
    return tokenized

class DebiasedTrainer(Trainer):
    
    def __init__(self, *args, bias_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_model_moved = False  # Track if we've moved bias model to GPU
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.bias_model is not None:
            # Move bias model to same device as inputs (only do this once)
            if not self.bias_model_moved:
                device = next(model.parameters()).device
                self.bias_model = self.bias_model.to(device)
                self.bias_model_moved = True
            
            # Get bias model predictions
            with torch.no_grad():
                bias_outputs = self.bias_model(**inputs)
                bias_probs = F.softmax(bias_outputs.logits, dim=-1)
                bias_confidence = bias_probs.max(dim=-1)[0]
            
            # Reweight loss: downweight examples where bias model is confident
            # Weight = 1 / (1 + bias_confidence)
            weights = 1.0 / (1.0 + bias_confidence)
            
            # Compute weighted cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            loss = (loss * weights).mean()
        else:
            # Standard loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def evaluate_bias_model(model_path, output_dir, max_length=128):
    """
    Evaluate hypothesis-only bias model on SNLI test set.
    
    Args:
        model_path: Path to trained hypothesis-only model
        output_dir: Directory to save evaluation results and predictions
        max_length: Maximum sequence length for tokenization
    """
    print_title("Evaluating Hypothesis-Only Bias Model")
    
    # Convert to absolute path if it's a relative path
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    # Verify the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Load model and tokenizer
    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        local_files_only=True,
        trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    
    # Load SNLI test set
    print("Loading SNLI test dataset...")
    dataset = datasets.load_dataset("snli")
    test_dataset = dataset["test"].filter(lambda x: x["label"] != -1)
    print(f"Test examples: {len(test_dataset)}")
    
    # Tokenize hypothesis only
    def tokenize_hypothesis_only(examples):
        tokenized = tokenizer(
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        tokenized['label'] = examples['label']
        return tokenized
    
    print("Tokenizing test dataset...")
    tokenized_test = test_dataset.map(
        tokenize_hypothesis_only, 
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=test_dataset.column_names
    )
    
    # Define compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Create Trainer for evaluation
    print("Running evaluation...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Evaluate on test set
    results = trainer.evaluate(tokenized_test)
    
    print("\n" + "=" * 50)
    print("HYPOTHESIS-ONLY BIAS MODEL RESULTS")
    print("=" * 50)
    print(f"Test Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"Test F1:        {results['eval_f1']:.4f}")
    print(f"Test Precision: {results['eval_precision']:.4f}")
    print(f"Test Recall:    {results['eval_recall']:.4f}")
    print("=" * 50)
    
    # Get predictions for ensemble use
    predictions = trainer.predict(tokenized_test)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Per-class performance
    print("\n" + "=" * 50)
    print("PER-CLASS PERFORMANCE")
    print("=" * 50)
    print(classification_report(
        true_labels, 
        pred_labels, 
        target_names=['Entailment', 'Neutral', 'Contradiction'],
        digits=4
    ))
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evaluation metrics
    metrics_file = os.path.join(output_dir, 'bias_model_eval_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save prediction logits (for ensemble debiasing)
    logits_file = os.path.join(output_dir, 'bias_model_predictions.npy')
    np.save(logits_file, predictions.predictions)
    print(f"✓ Prediction logits saved to: {logits_file}")
    
    # Save detailed predictions (optional, for analysis)
    predictions_file = os.path.join(output_dir, 'bias_model_predictions.jsonl')
    with open(predictions_file, 'w') as f:
        for i, example in enumerate(test_dataset):
            pred_data = {
                'hypothesis': example['hypothesis'],
                'label': int(example['label']),
                'predicted_label': int(pred_labels[i]),
                'predicted_scores': predictions.predictions[i].tolist(),
                'correct': int(pred_labels[i]) == int(example['label'])
            }
            f.write(json.dumps(pred_data) + '\n')
    print(f"✓ Detailed predictions saved to: {predictions_file}")
    
    print_footer()
    
    return results

def analyze_hypothesis_bias(predictions_file, output_file=None):
    """
    Analyze hypothesis-only biases including negation words and length patterns.
    
    Args:
        predictions_file: Path to predictions jsonl file
        output_file: Optional path to save analysis results
    """
    print_title("Hypothesis Bias Analysis")
    
    # Load predictions
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    print(f"Loaded {len(predictions)} predictions\n")
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    df['correct'] = df['label'] == df['predicted_label']
    
    overall_acc = df['correct'].mean()
    print(f"Overall Accuracy: {overall_acc:.4f}\n")
    
    # Analyze word patterns
    def analyze_word_pattern(df, word, word_label):
        hypothesis_with_word = df['hypothesis'].str.lower().str.contains(word, na=False, regex=False)
        
        print(f"=== Hypothesis contains '{word}' ===")
        total_with_word = hypothesis_with_word.sum()
        total_without_word = (~hypothesis_with_word).sum()
        print(f"With '{word}': {total_with_word} ({total_with_word/len(df)*100:.1f}%)")
        print(f"Without '{word}': {total_without_word} ({total_without_word/len(df)*100:.1f}%)")
        
        # Distribution of labels when word is present
        subset_with = df[hypothesis_with_word]
        subset_without = df[~hypothesis_with_word]
        
        label_names = ['Entailment', 'Neutral', 'Contradiction']
        
        results = {
            'word': word,
            'total_with_word': int(total_with_word),
            'total_without_word': int(total_without_word),
            'with_word': {},
            'without_word': {}
        }
        
        print(f"\nWith '{word}':")
        for label in [0, 1, 2]:
            count = (subset_with['label'] == label).sum()
            pct = count / len(subset_with) * 100 if len(subset_with) > 0 else 0
            acc = subset_with[subset_with['label'] == label]['correct'].mean() if count > 0 else 0
            print(f"  {label_names[label]:13s}: {count:4d} ({pct:5.1f}%) - Accuracy: {acc:.4f}")
            results['with_word'][label_names[label].lower()] = {
                'count': int(count),
                'percentage': float(pct),
                'accuracy': float(acc) if count > 0 else None
            }
        
        print(f"\nWithout '{word}':")
        for label in [0, 1, 2]:
            count = (subset_without['label'] == label).sum()
            pct = count / len(subset_without) * 100 if len(subset_without) > 0 else 0
            acc = subset_without[subset_without['label'] == label]['correct'].mean() if count > 0 else 0
            print(f"  {label_names[label]:13s}: {count:4d} ({pct:5.1f}%) - Accuracy: {acc:.4f}")
            results['without_word'][label_names[label].lower()] = {
                'count': int(count),
                'percentage': float(pct),
                'accuracy': float(acc) if count > 0 else None
            }
        
        print()
        return results
    
    # Analyze negation words
    print("=" * 60)
    print("NEGATION WORD ANALYSIS")
    print("=" * 60)
    
    negation_results = []
    for word in ['not', 'no ', 'never', "n't"]:
        result = analyze_word_pattern(df, word, 'negation')
        negation_results.append(result)
    
    # Analyze hypothesis length
    print("=" * 60)
    print("HYPOTHESIS LENGTH ANALYSIS")
    print("=" * 60)
    
    df['hyp_length'] = df['hypothesis'].str.split().str.len()
    
    print("\nAverage hypothesis length by label:")
    label_names = ['Entailment', 'Neutral', 'Contradiction']
    length_by_label = {}
    for label in [0, 1, 2]:
        avg_len = df[df['label'] == label]['hyp_length'].mean()
        print(f"  {label_names[label]:13s}: {avg_len:.2f} words")
        length_by_label[label_names[label].lower()] = float(avg_len)
    
    # Accuracy by length quartile
    print("\nAccuracy by hypothesis length quartile:")
    df['length_quartile'] = pd.qcut(df['hyp_length'], q=4, labels=['Q1 (Short)', 'Q2', 'Q3', 'Q4 (Long)'], duplicates='drop')
    
    length_quartile_results = []
    for quartile in df['length_quartile'].unique():
        subset = df[df['length_quartile'] == quartile]
        acc = subset['correct'].mean()
        min_len = subset['hyp_length'].min()
        max_len = subset['hyp_length'].max()
        avg_len = subset['hyp_length'].mean()
        count = len(subset)
        print(f"  {str(quartile):12s}: {acc:.4f} ({min_len}-{max_len} words, avg={avg_len:.1f}, n={count})")
        length_quartile_results.append({
            'quartile': str(quartile),
            'accuracy': float(acc),
            'min_length': int(min_len),
            'max_length': int(max_len),
            'avg_length': float(avg_len),
            'count': int(count)
        })
    
    # Check for length bias
    short_acc = df[df['length_quartile'] == 'Q1 (Short)']['correct'].mean() if 'Q1 (Short)' in df['length_quartile'].values else None
    long_acc = df[df['length_quartile'] == 'Q4 (Long)']['correct'].mean() if 'Q4 (Long)' in df['length_quartile'].values else None
    
    if short_acc is not None and long_acc is not None:
        length_gap = long_acc - short_acc
        print(f"\nLength Bias Analysis:")
        print(f"  Short (Q1) accuracy: {short_acc:.4f}")
        print(f"  Long (Q4) accuracy:  {long_acc:.4f}")
        print(f"  Accuracy gap:        {length_gap:+.4f} ({length_gap*100:+.2f} percentage points)")
        
        if abs(length_gap) > 0.02:
            print(f"  ⚠️  Significant length bias detected!")
        else:
            print(f"  ✓ Minimal length bias")
    
    print("=" * 60)
    
    # Save results if output file specified
    if output_file:
        results = {
            'overall_accuracy': float(overall_acc),
            'negation_analysis': negation_results,
            'length_analysis': {
                'avg_length_by_label': length_by_label,
                'accuracy_by_quartile': length_quartile_results,
                'length_gap': float(length_gap) if short_acc is not None and long_acc is not None else None,
                'length_bias_detected': bool(abs(length_gap) > 0.02) if short_acc is not None and long_acc is not None else None
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print_footer()
    
    return df

def ensemble_debias_evaluate(biased_model_path, bias_model_path, output_dir, alpha=1.0, max_length=128):
    """
    Perform ensemble debiasing by combining biased and bias-only model predictions.
    
    Args:
        biased_model_path: Path to the full (biased) model
        bias_model_path: Path to the hypothesis-only bias model
        output_dir: Directory to save debiased results
        alpha: Weight for bias model logits (default=1.0)
        max_length: Maximum sequence length
    """
    print_title("Ensemble Debiasing Evaluation")
    
    # Load models
    print(f"Loading biased model from: {biased_model_path}")
    biased_model = AutoModelForSequenceClassification.from_pretrained(
        biased_model_path, local_files_only=True
    )
    biased_model.eval()
    
    print(f"Loading bias model from: {bias_model_path}")
    bias_model = AutoModelForSequenceClassification.from_pretrained(
        bias_model_path, local_files_only=True
    )
    bias_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
    
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    biased_model = biased_model.to(device)
    bias_model = bias_model.to(device)
    
    # Load SNLI test set
    print("Loading SNLI test dataset...")
    dataset = datasets.load_dataset("snli")
    test_dataset = dataset["test"].filter(lambda x: x["label"] != -1)
    print(f"Test examples: {len(test_dataset)}")
    
    # Tokenize full premise-hypothesis pairs for biased model
    def tokenize_full(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize hypothesis only for bias model
    def tokenize_hypothesis_only(examples):
        return tokenizer(
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    print("Tokenizing test dataset...")
    tokenized_full = test_dataset.map(
        tokenize_full,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS
    )
    tokenized_hypo = test_dataset.map(
        tokenize_hypothesis_only,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS
    )
    
    # Get predictions from both models
    print("Getting predictions from biased model...")
    biased_logits = []
    with torch.no_grad():
        for i in range(0, len(tokenized_full), 32):  # batch size 32
            batch = tokenized_full[i:i+32]
            inputs = {
                'input_ids': torch.tensor(batch['input_ids']).to(device),
                'attention_mask': torch.tensor(batch['attention_mask']).to(device)
            }
            outputs = biased_model(**inputs)
            biased_logits.append(outputs.logits.cpu().numpy())
    biased_logits = np.vstack(biased_logits)
    
    print("Getting predictions from bias model...")
    bias_logits = []
    with torch.no_grad():
        for i in range(0, len(tokenized_hypo), 32):
            batch = tokenized_hypo[i:i+32]
            inputs = {
                'input_ids': torch.tensor(batch['input_ids']).to(device),
                'attention_mask': torch.tensor(batch['attention_mask']).to(device)
            }
            outputs = bias_model(**inputs)
            bias_logits.append(outputs.logits.cpu().numpy())
    bias_logits = np.vstack(bias_logits)
    
    # Ensemble debiasing: subtract weighted bias logits from biased logits
    print(f"Applying ensemble debiasing (alpha={alpha})...")
    debiased_logits = biased_logits - alpha * bias_logits
    
    # Get predictions
    biased_preds = np.argmax(biased_logits, axis=1)
    bias_preds = np.argmax(bias_logits, axis=1)
    debiased_preds = np.argmax(debiased_logits, axis=1)
    
    # Get true labels
    true_labels = np.array(test_dataset['label'])
    
    # Calculate accuracies
    biased_acc = accuracy_score(true_labels, biased_preds)
    bias_acc = accuracy_score(true_labels, bias_preds)
    debiased_acc = accuracy_score(true_labels, debiased_preds)
    
    # Calculate metrics for debiased model
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, debiased_preds, average='weighted'
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("ENSEMBLE DEBIASING RESULTS")
    print("=" * 60)
    print(f"Bias-only model accuracy:     {bias_acc:.4f}")
    print(f"Biased model accuracy:        {biased_acc:.4f}")
    print(f"Debiased model accuracy:      {debiased_acc:.4f}")
    print(f"\nDebiased model metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print("=" * 60)
    
    # Per-class performance for debiased model
    print("\n" + "=" * 60)
    print("DEBIASED MODEL - PER-CLASS PERFORMANCE")
    print("=" * 60)
    print(classification_report(
        true_labels,
        debiased_preds,
        target_names=['Entailment', 'Neutral', 'Contradiction'],
        digits=4
    ))
    print("=" * 60)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison metrics
    comparison_file = os.path.join(output_dir, 'debiasing_comparison.json')
    comparison_results = {
        'alpha': alpha,
        'bias_only_accuracy': float(bias_acc),
        'biased_accuracy': float(biased_acc),
        'debiased_accuracy': float(debiased_acc),
        'debiased_precision': float(precision),
        'debiased_recall': float(recall),
        'debiased_f1': float(f1),
        'accuracy_drop': float(biased_acc - debiased_acc)
    }
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\n✓ Comparison metrics saved to: {comparison_file}")
    
    # Save debiased logits
    logits_file = os.path.join(output_dir, 'debiased_logits.npy')
    np.save(logits_file, debiased_logits)
    print(f"✓ Debiased logits saved to: {logits_file}")
    
    # Save detailed predictions
    predictions_file = os.path.join(output_dir, 'debiased_predictions.jsonl')
    with open(predictions_file, 'w') as f:
        for i, example in enumerate(test_dataset):
            pred_data = {
                'premise': example['premise'],
                'hypothesis': example['hypothesis'],
                'label': int(example['label']),
                'biased_prediction': int(biased_preds[i]),
                'bias_prediction': int(bias_preds[i]),
                'debiased_prediction': int(debiased_preds[i]),
                'biased_correct': int(biased_preds[i]) == int(example['label']),
                'debiased_correct': int(debiased_preds[i]) == int(example['label'])
            }
            f.write(json.dumps(pred_data) + '\n')
    print(f"✓ Detailed predictions saved to: {predictions_file}")
    
    print_footer()
    
    return comparison_results

def main():
    # Check for version flag early (before parsing all arguments)
    import sys
    if '--check_versions' in sys.argv:
        check_versions()
        return
    
    # Check for ensemble debiasing flag early
    if '--ensemble_debias' in sys.argv:
        # Get required parameters
        if '--biased_model' not in sys.argv or '--bias_model' not in sys.argv:
            print("Error: --ensemble_debias requires --biased_model and --bias_model")
            print("Usage: python run.py --ensemble_debias --biased_model <path> --bias_model <path> [--output_dir <dir>] [--alpha <float>]")
            return
        
        biased_idx = sys.argv.index('--biased_model')
        bias_idx = sys.argv.index('--bias_model')
        
        biased_model_path = sys.argv[biased_idx + 1]
        bias_model_path = sys.argv[bias_idx + 1]
        
        # Get optional parameters
        output_dir = './debiased_model_eval'
        if '--output_dir' in sys.argv:
            output_idx = sys.argv.index('--output_dir')
            if output_idx + 1 < len(sys.argv):
                output_dir = sys.argv[output_idx + 1]
        
        alpha = 1.0
        if '--alpha' in sys.argv:
            alpha_idx = sys.argv.index('--alpha')
            if alpha_idx + 1 < len(sys.argv):
                alpha = float(sys.argv[alpha_idx + 1])
        
        max_length = 128
        if '--max_length' in sys.argv:
            max_len_idx = sys.argv.index('--max_length')
            if max_len_idx + 1 < len(sys.argv):
                max_length = int(sys.argv[max_len_idx + 1])
        
        ensemble_debias_evaluate(biased_model_path, bias_model_path, output_dir, alpha, max_length)
        return
    
    # Check for evaluate bias model flag early
    if '--eval_bias_model' in sys.argv:
        idx = sys.argv.index('--eval_bias_model')
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]
            # Get output directory
            output_dir = './bias_model_eval'
            if '--output_dir' in sys.argv:
                output_idx = sys.argv.index('--output_dir')
                if output_idx + 1 < len(sys.argv):
                    output_dir = sys.argv[output_idx + 1]
            # Get max_length if specified
            max_length = 128
            if '--max_length' in sys.argv:
                max_len_idx = sys.argv.index('--max_length')
                if max_len_idx + 1 < len(sys.argv):
                    max_length = int(sys.argv[max_len_idx + 1])
            
            evaluate_bias_model(model_path, output_dir, max_length)
        else:
            print("Error: --eval_bias_model requires a model path")
            print("Usage: python run.py --eval_bias_model <model_path> [--output_dir <dir>] [--max_length <int>]")
        return
    
    # Check for analyze_overlap flag early
    if '--analyze_overlap' in sys.argv:
        idx = sys.argv.index('--analyze_overlap')
        if idx + 1 < len(sys.argv):
            predictions_file = sys.argv[idx + 1]
            # Check for optional output file
            output_file = None
            if '--analysis_output' in sys.argv:
                output_idx = sys.argv.index('--analysis_output')
                if output_idx + 1 < len(sys.argv):
                    output_file = sys.argv[output_idx + 1]
            analyze_overlap(predictions_file, output_file)
        else:
            print("Error: --analyze_overlap requires a predictions file path")
            print("Usage: python run.py --analyze_overlap <predictions.jsonl> [--analysis_output <output.json>]")
        return
    
    # Check for analyze_hypothesis_bias flag early
    if '--analyze_hypothesis_bias' in sys.argv:
        idx = sys.argv.index('--analyze_hypothesis_bias')
        if idx + 1 < len(sys.argv):
            predictions_file = sys.argv[idx + 1]
            # Check for optional output file
            output_file = None
            if '--analysis_output' in sys.argv:
                output_idx = sys.argv.index('--analysis_output')
                if output_idx + 1 < len(sys.argv):
                    output_file = sys.argv[output_idx + 1]
            analyze_hypothesis_bias(predictions_file, output_file)
        else:
            print("Error: --analyze_hypothesis_bias requires a predictions file path")
            print("Usage: python run.py --analyze_hypothesis_bias <predictions.jsonl> [--analysis_output <output.json>]")
        return
    
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--check_versions', action='store_true',
                      help='Print version information and exit.')
    argp.add_argument('--ensemble_debias', action='store_true',
                      help='Perform ensemble debiasing using biased and bias models.')
    argp.add_argument('--biased_model', type=str, default=None,
                      help='Path to biased (full) model for ensemble debiasing.')
    argp.add_argument('--bias_model', type=str, default=None,
                      help='Path to bias (hypothesis-only) model for ensemble debiasing.')
    argp.add_argument('--alpha', type=float, default=1.0,
                      help='Weight for bias model logits in ensemble debiasing (default=1.0).')
    argp.add_argument('--eval_bias_model', type=str, default=None,
                      help='Evaluate hypothesis-only bias model and save predictions.')
    argp.add_argument('--analyze_overlap', type=str, default=None,
                      help='Analyze lexical overlap in predictions file and exit.')
    argp.add_argument('--analyze_hypothesis_bias', type=str, default=None,
                      help='Analyze hypothesis-only biases (negation, length) in predictions file and exit.')
    argp.add_argument('--analysis_output', type=str, default=None,
                      help='Optional: Save analysis results to JSON file.')
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--hypothesis_only', action='store_true',
                      help='Train on hypothesis only (for bias model).')
    argp.add_argument('--bias_model_path', type=str, default=None,
                      help='Path to bias model for ensemble debiasing.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    bias_model = None
    if args.bias_model_path:
        print(f"Loading bias model from {args.bias_model_path}")
        bias_model = model_class.from_pretrained(args.bias_model_path, **task_kwargs)
        bias_model.eval()
        if hasattr(bias_model, 'electra'):
            for param in bias_model.electra.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        # Move bias model to the same device as the main model
        bias_model = bias_model.to(model.device)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        if args.hypothesis_only:
            # For bias model: only use hypothesis
            prepare_train_dataset = prepare_eval_dataset = \
                lambda exs: prepare_dataset_nli_hypothesis_only(exs, tokenizer, args.max_length)
        else:
            prepare_train_dataset = prepare_eval_dataset = \
                lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
            # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    if args.bias_model_path:
        trainer = DebiasedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_and_store_predictions,
            bias_model=bias_model
        )
    else:
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_and_store_predictions
        )
    
    # Train and/or evaluate
    if training_args.do_train:
        # Check for existing checkpoints to resume from, total life saver so I don't keep rerunning from the beginning
        checkpoint = None
        if os.path.isdir(training_args.output_dir):
            checkpoints = [d for d in os.listdir(training_args.output_dir) 
                          if d.startswith('checkpoint-')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
                print(f"🔄 Resuming from checkpoint: {checkpoint}")
    
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
