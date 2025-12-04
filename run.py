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

NUM_PREPROCESSING_WORKERS = 2


def check_versions():
    """
    Print version information for PyTorch and Transformers, and check CUDA availability.
    """
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:         {torch.version.cuda}")
        print(f"GPU device count:     {torch.cuda.device_count()}")
        print(f"Current GPU device:   {torch.cuda.current_device()}")
        print(f"GPU device name:      {torch.cuda.get_device_name(0)}")
    print("=" * 60)


def analyze_overlap(predictions_file, output_file=None):
    """
    Analyze lexical overlap artifact in predictions.
    Computes word overlap and accuracy by overlap quartiles.
    """
    import pandas as pd
    
    print("=" * 60)
    print("LEXICAL OVERLAP ANALYSIS")
    print("=" * 60)
    
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
    
    print("=" * 60)
    
    # Save results if output file specified
    if output_file:
        results = {
            'overall_accuracy': float(overall_acc),
            'accuracy_by_label': {
                label_names[i]: float(df[df['label'] == i]['correct'].mean())
                for i in range(3)
            },
            'accuracy_by_quartile': quartile_results,
            'overlap_gap': float(gap),
            'artifact_detected': abs(gap) > 0.02
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return df


def prepare_dataset_nli_hypothesis_only(examples, tokenizer, max_length):
    """
    Preprocessing function for hypothesis-only training (bias model).
    Only tokenizes the hypothesis, not the premise.
    """
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

def main():
    # Check for version flag early (before parsing all arguments)
    import sys
    if '--check_versions' in sys.argv:
        check_versions()
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
    argp.add_argument('--analyze_overlap', type=str, default=None,
                      help='Analyze lexical overlap in predictions file and exit.')
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
        trainer.train()
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
