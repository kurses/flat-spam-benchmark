#!/usr/bin/env python3
"""
Multi-Model FLAT Benchmarking Suite

This script trains and evaluates multiple transformer models (BERT, ALBERT, DeBERTa)
using the FLAT (Feature Learning for Adversarial Training) methodology.

Usage:
    python benchmark_suite.py

Output:
    - Model checkpoints: flat_bert-base-uncased.pt, flat_albert-base-v2.pt, flat_distilbert-base-uncased.pt
    - Visualizations: loss_curves.png, performance_comparison.png, confusion_matrix_components.png
    - Report: benchmark_report.csv
"""

import os
import argparse
import glob

# Disable CUDA fusers/JIT to avoid NVRTC builtins issues on cu130
os.environ.setdefault('TORCH_CUDA_FUSER_DISABLE', '1')
os.environ.setdefault('PYTORCH_NVFUSER_DISABLE', '1')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import textattack as ta
from textattack.attack_recipes import PWWSRen2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_results import SuccessfulAttackResult
from textattack import AttackArgs
import kagglehub
import gc
import nltk
"""
Note: Disabling allocator config tweaks to avoid Torch internal asserts
on some versions. We'll rely on reduced batch/seq length for stability.
"""

# Download NLTK dependencies
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH = kagglehub.dataset_download("venky73/spam-mails-dataset") + "/spam_ham_dataset.csv"
MODELS = ['bert-base-uncased', 'albert-base-v2', 'distilbert-base-uncased']
IMG_PATH = 'img/'

# Known models map to help recover from filenames like flat_bert-base-uncased.pt
KNOWN_MODELS = [
    'bert-base-uncased',
    'albert-base-v2',
    'distilbert-base-uncased',
    'microsoft/deberta-base'
]
MAX_SEQ_LENGTH = 256
ATTACK_BATCH_SIZE = 8

# FLAT Hyperparameters
NUM_FLAT_ITERATIONS = 5
NUM_EPOCHS_PER_ITERATION = 2
GAMMA = 0.001
TRAIN_ATTACK_SAMPLES = 200  # Set to -1 for full dataset
MAX_QUERIES_TA = 750

# Evaluation
EVAL_SUBSET_SIZE = 50

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Using device: {DEVICE}\n")

# ============================================================================
# DATASET CLASSES
# ============================================================================

class CustomTextDataset(Dataset):
    """Dataset wrapper for text classification."""
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LENGTH
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class FLATDataset(Dataset):
    """Dataset for FLAT training with original and adversarial examples."""
    def __init__(self, original_dataset, adversarial_texts, replacement_meta, tokenizer):
        self.original_encodings = original_dataset.encodings
        self.original_labels = original_dataset.labels

        if not adversarial_texts:
            print("WARNING: FLATDataset received empty adversarial list.")
            self.adv_encodings = tokenizer(
                ["fallback"] * len(self.original_labels),
                truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH
            )
        else:
            self.adv_encodings = tokenizer(
                adversarial_texts,
                truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH
            )

        self.replacement_meta = replacement_meta

    def __getitem__(self, idx):
        item_ori = {key: torch.tensor(val[idx]) for key, val in self.original_encodings.items()}
        item_ori['labels'] = self.original_labels[idx]

        item_adv = {key: torch.tensor(val[idx]) for key, val in self.adv_encodings.items()}
        item_adv['labels'] = self.original_labels[idx]

        meta = self.replacement_meta[idx]
        return item_ori, item_adv, meta

    def __len__(self):
        return len(self.original_labels)


# ============================================================================
# FLAT MODEL
# ============================================================================

class FLATUniversalModel(nn.Module):
    """Universal FLAT model supporting BERT, ALBERT, and DeBERTa."""
    def __init__(self, model_name, vocab_size, num_labels=2):
        super().__init__()
        self.base_model_wrapper = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.vocab_size = vocab_size
        self.phi = nn.Parameter(torch.rand(vocab_size))

    def forward(self, input_ids, attention_mask, labels=None, mask_input=True, embeddings_override=None):
        # Get base model
        if hasattr(self.base_model_wrapper, 'bert'):
            base_model = self.base_model_wrapper.bert
        elif hasattr(self.base_model_wrapper, 'albert'):
            base_model = self.base_model_wrapper.albert
        elif hasattr(self.base_model_wrapper, 'deberta'):
            base_model = self.base_model_wrapper.deberta
        elif hasattr(self.base_model_wrapper, 'distilbert'):
            base_model = self.base_model_wrapper.distilbert
        else:
            raise ValueError(f"Unsupported model type")

        # Get embeddings (allow override for attribution)
        if embeddings_override is not None:
            embeddings = embeddings_override
        else:
            embeddings = base_model.embeddings(input_ids)

        # Apply mask
        if mask_input:
            importance_scores = self.phi[input_ids]
            W = torch.sigmoid(importance_scores)
            masked_embeddings = embeddings * W.unsqueeze(-1)
        else:
            masked_embeddings = embeddings

        # Forward pass
        output = self.base_model_wrapper(
            inputs_embeds=masked_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )

        return output, self.phi


# ============================================================================
# ADVERSARIAL DATA GENERATOR
# ============================================================================

class AdversarialDataGenerator:
    """Generates adversarial examples using PWWS attack."""
    def __init__(self, model_name, tokenizer, attack_recipe):
        self.tokenizer = tokenizer
        self.attack_recipe = attack_recipe

    def generate(self, model, original_texts, original_labels, num_examples=-1):
        print(f"  Attacking {len(original_texts)} samples...")

        ta_model = model.base_model_wrapper
        model_wrapper = HuggingFaceModelWrapper(ta_model, self.tokenizer)

        attack_dataset = ta.datasets.Dataset(
            [(t, l) for t, l in zip(original_texts, original_labels)]
        )

        attack_args = AttackArgs(
            num_examples=num_examples,
            log_to_csv=None,
            checkpoint_interval=None,
            disable_stdout=True,
            query_budget=MAX_QUERIES_TA,
            parallel=False
        )

        attack = self.attack_recipe.build(model_wrapper)
        attacker = ta.Attacker(attack, attack_dataset, attack_args)

        results = attacker.attack_dataset()

        # Process results
        adversarial_texts = []
        replacement_meta_list = []
        success_count = 0

        results_map = {i: res for i, res in enumerate(results)}

        for i in range(len(original_texts)):
            final_adv_text = original_texts[i]
            meta = []

            if i in results_map and isinstance(results_map[i], SuccessfulAttackResult):
                success_count += 1
                final_adv_text = results_map[i].perturbed_result.attacked_text.text

                original_ids = self.tokenizer.encode(original_texts[i], add_special_tokens=False)
                adv_ids = self.tokenizer.encode(final_adv_text, add_special_tokens=False)

                for idx, (o_id, a_id) in enumerate(zip(original_ids, adv_ids)):
                    if o_id != a_id:
                        meta.append((o_id, a_id))
                        break

            adversarial_texts.append(final_adv_text)
            replacement_meta_list.append(meta)

        print(f"  Attack complete. Successful: {success_count}/{len(original_texts)}")
        return adversarial_texts, replacement_meta_list


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def flat_collate_fn(batch):
    """Custom collate function for FLAT dataset."""
    from torch.utils.data._utils.collate import default_collate
    batch_ori = default_collate([item[0] for item in batch])
    batch_adv = default_collate([item[1] for item in batch])
    batch_meta = [item[2] for item in batch]
    return batch_ori, batch_adv, batch_meta


def calculate_detailed_metrics(model, data_loader, device):
    """Calculate confusion matrix and derived metrics."""
    model.eval()
    TP = TN = FP = FN = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(inputs, masks, mask_input=False)
            predictions = torch.argmax(outputs.logits, dim=-1)

            for pred, label in zip(predictions, labels):
                if pred == 1 and label == 1:
                    TP += 1
                elif pred == 0 and label == 0:
                    TN += 1
                elif pred == 1 and label == 0:
                    FP += 1
                else:
                    FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': precision, 'recall': recall,
        'f1': f1, 'accuracy': accuracy
    }


def calculate_aa_metrics(model, generator, texts, labels, device):
    """Calculate After-Attack Accuracy and Attack Success Rate."""
    model.eval()

    print(f"  Evaluating robustness on {len(texts)} samples...")
    adv_texts, _ = generator.generate(model, texts, labels)

    if not adv_texts:
        return {'aa_accuracy': 0.0, 'attack_success_rate': 0.0}

    adv_encodings = generator.tokenizer(
        adv_texts, truncation=True, padding='max_length',
        max_length=MAX_SEQ_LENGTH, return_tensors='pt'
    )

    adv_inputs = adv_encodings['input_ids'].to(device)
    adv_masks = adv_encodings['attention_mask'].to(device)
    adv_labels = torch.tensor(labels[:len(adv_texts)]).to(device)

    with torch.no_grad():
        outputs, _ = model(adv_inputs, adv_masks, mask_input=False)
        predictions = torch.argmax(outputs.logits, dim=-1)

        aa_correct = (predictions == adv_labels).sum().item()
        aa_total = len(adv_texts)

    aa_accuracy = aa_correct / aa_total
    attack_success_rate = 1.0 - aa_accuracy

    return {
        'aa_accuracy': aa_accuracy,
        'attack_success_rate': attack_success_rate
    }


def calculate_interpretability_metrics(model, tokenizer, original_text, adversarial_text, device, k=10):
    """Calculate Kendall's Tau and Top-k Intersection using embedding gradients.

    - Computes token saliency by backpropagating prediction score w.r.t input embeddings.
    - Filters special/pad tokens; compares rankings on trimmed aligned lengths.
    - Top-k intersection compares token id sets among top-k salient tokens.
    """
    model.eval()

    def saliency_for_text(text):
        enc = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_SEQ_LENGTH, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        emb_layer = model.base_model_wrapper.get_input_embeddings()
        # Detach to create leaf tensor for gradient capture
        embeddings = emb_layer(input_ids).detach().requires_grad_(True)
        embeddings.retain_grad()

        model.zero_grad(set_to_none=True)
        outputs, _ = model(
            input_ids=None,
            attention_mask=attention_mask,
            labels=None,
            mask_input=False,
            embeddings_override=embeddings,
        )
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        score = logits[0, pred]
        score.backward()
        if embeddings.grad is None:
            return input_ids[0].detach().cpu().numpy(), np.array([]), attention_mask[0].detach().cpu().numpy().astype(bool)
        sal = embeddings.grad[0].abs().sum(-1).detach().cpu().numpy()
        ids = input_ids[0].detach().cpu().numpy()
        mask = attention_mask[0].detach().cpu().numpy().astype(bool)
        return ids, sal, mask

    ori_ids, ori_sal, ori_mask = saliency_for_text(original_text)
    adv_ids, adv_sal, adv_mask = saliency_for_text(adversarial_text)

    special_ids = set()
    for tok in [getattr(tokenizer, 'cls_token_id', None), getattr(tokenizer, 'sep_token_id', None), getattr(tokenizer, 'pad_token_id', None)]:
        if tok is not None:
            special_ids.add(tok)

    def filter_valid(ids, sal, mask):
        idxs = [i for i in range(len(ids)) if mask[i] and ids[i] not in special_ids]
        if len(idxs) == 0:
            return np.array([]), np.array([])
        return np.array([ids[i] for i in idxs]), np.array([sal[i] for i in idxs])

    ori_ids_f, ori_sal_f = filter_valid(ori_ids, ori_sal, ori_mask)
    adv_ids_f, adv_sal_f = filter_valid(adv_ids, adv_sal, adv_mask)

    if len(ori_sal_f) == 0 or len(adv_sal_f) == 0:
        return {'kendall_tau': 0.0, 'top_k_intersection': 0.0}

    L = min(len(ori_sal_f), len(adv_sal_f))
    ori_sal_f = ori_sal_f[:L]
    adv_sal_f = adv_sal_f[:L]
    ori_ids_f = ori_ids_f[:L]
    adv_ids_f = adv_ids_f[:L]

    ori_rank = np.argsort(ori_sal_f)[::-1]
    adv_rank = np.argsort(adv_sal_f)[::-1]

    tau, _ = kendalltau(ori_rank, adv_rank)
    tau = 0.0 if np.isnan(tau) else float(tau)

    k_eff = int(min(k, len(ori_rank), len(adv_rank)))
    if k_eff == 0:
        topk_inter = 0.0
    else:
        ori_top_tokens = set(ori_ids_f[ori_rank[:k_eff]].tolist())
        adv_top_tokens = set(adv_ids_f[adv_rank[:k_eff]].tolist())
        topk_inter = float(len(ori_top_tokens & adv_top_tokens)) / float(k_eff)

    return {'kendall_tau': tau, 'top_k_intersection': topk_inter}


# ============================================================================
# MAIN BENCHMARKING FUNCTION
# ============================================================================

def run_benchmarking():
    """Main benchmarking loop for all models."""
    
    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    full_df = pd.read_csv(CSV_PATH)
    texts = full_df['text'].fillna('').tolist()
    labels = full_df['label_num'].tolist()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    dev_texts, test_texts, dev_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    print(f"Train samples: {len(train_texts)}, Dev samples: {len(dev_texts)}\n")

    # Master log
    master_log = {
        'model_results': {},
        'training_curves': {}
    }

    # Iterate through models
    for model_idx, current_model_name in enumerate(MODELS):
        print("\n" + "="*80)
        print(f"BENCHMARKING MODEL {model_idx + 1}/{len(MODELS)}: {current_model_name}")
        print("="*80)

        # Instantiate model
        print(f"\n[1/4] Instantiating {current_model_name}...")
        model_tokenizer = AutoTokenizer.from_pretrained(current_model_name)
        vocab_size = model_tokenizer.vocab_size
        model_short = current_model_name.split('/')[-1]

        train_dataset = CustomTextDataset(train_texts, train_labels, model_tokenizer)
        dev_dataset = CustomTextDataset(dev_texts, dev_labels, model_tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=ATTACK_BATCH_SIZE, shuffle=False)

        model = FLATUniversalModel(current_model_name, vocab_size).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-5)

        # AMP mixed precision scaler (only on CUDA)
        # Disable AMP for DeBERTa to avoid Half overflow in attention masking
        use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        print(f"Model instantiated. Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Initialize training
        print(f"\n[2/4] Starting FLAT training pipeline...")

        adv_generator = AdversarialDataGenerator(current_model_name, model_tokenizer, PWWSRen2019)

        initial_meta = [[] for _ in range(len(train_texts))]
        augmented_dataset = FLATDataset(train_dataset, train_texts, initial_meta, model_tokenizer)
        augmented_data_loader = DataLoader(
            augmented_dataset,
            batch_size=ATTACK_BATCH_SIZE,
            shuffle=True,
            collate_fn=flat_collate_fn
        )

        # Training metrics storage
        training_log = {
            'iteration': [], 'epoch': [],
            'L_total': [], 'L_pred': [], 'L_imp': [],
            'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
            'TP': [], 'TN': [], 'FP': [], 'FN': []
        }

        # FLAT Training Loop
        for iteration in range(NUM_FLAT_ITERATIONS):
            print(f"\n  --- ITERATION {iteration + 1}/{NUM_FLAT_ITERATIONS} ---")

            if iteration > 0:
                adv_texts, meta = adv_generator.generate(
                    model, train_texts, train_labels,
                    num_examples=TRAIN_ATTACK_SAMPLES
                )

                if len(adv_texts) > 0:
                    if len(adv_texts) < len(train_texts):
                        diff = len(train_texts) - len(adv_texts)
                        adv_texts.extend(train_texts[len(adv_texts):])
                        meta.extend([[]] * diff)

                    augmented_dataset = FLATDataset(train_dataset, adv_texts, meta, model_tokenizer)
                    augmented_data_loader = DataLoader(
                        augmented_dataset,
                        batch_size=ATTACK_BATCH_SIZE,
                        shuffle=True,
                        collate_fn=flat_collate_fn
                    )
                    print(f"  Dataset updated with {len([m for m in meta if m])} adversarial examples.")

            # Training epochs
            for epoch in range(NUM_EPOCHS_PER_ITERATION):
                model.train()
                total_loss = total_l_pred = total_l_imp = 0

                for batch_ori, batch_adv, attack_meta in augmented_data_loader:
                    optimizer.zero_grad()

                    ori_input_ids = batch_ori['input_ids'].to(DEVICE)
                    ori_attention_mask = batch_ori['attention_mask'].to(DEVICE)
                    ori_labels = batch_ori['labels'].to(DEVICE)
                    adv_input_ids = batch_adv['input_ids'].to(DEVICE)
                    adv_attention_mask = batch_adv['attention_mask'].to(DEVICE)
                    adv_labels = batch_adv['labels'].to(DEVICE)

                    # Forward pass (AMP disabled for stability)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_ori, phi = model(ori_input_ids, ori_attention_mask, labels=ori_labels)
                        output_adv, _ = model(adv_input_ids, adv_attention_mask, labels=adv_labels)

                    L_imp = 0.0
                    if iteration > 0:
                        for batch_m in attack_meta:
                            for ori_id, adv_id in batch_m:
                                if ori_id < len(phi) and adv_id < len(phi):
                                    L_imp += torch.abs(phi[ori_id] - phi[adv_id])

                    # Compute losses in FP32 to avoid half overflow
                    L_pred = output_ori.loss.float() + output_adv.loss.float()

                    # Ensure L_imp is FP32
                    if isinstance(L_imp, torch.Tensor):
                        L_imp = L_imp.float()
                    else:
                        L_imp = torch.tensor(L_imp, dtype=torch.float32, device=DEVICE)

                    loss_total = L_pred + GAMMA * L_imp

                    loss_total.float().backward()
                    optimizer.step()

                    total_loss += loss_total.item()
                    total_l_pred += L_pred.item()
                    total_l_imp += L_imp.item() if isinstance(L_imp, torch.Tensor) else L_imp

                avg_loss = total_loss / len(augmented_data_loader)
                avg_l_pred = total_l_pred / len(augmented_data_loader)
                avg_l_imp = total_l_imp / len(augmented_data_loader)

                val_metrics = calculate_detailed_metrics(model, dev_loader, DEVICE)

                training_log['iteration'].append(iteration + 1)
                training_log['epoch'].append(epoch + 1)
                training_log['L_total'].append(avg_loss)
                training_log['L_pred'].append(avg_l_pred)
                training_log['L_imp'].append(avg_l_imp)
                training_log['val_accuracy'].append(val_metrics['accuracy'])
                training_log['val_precision'].append(val_metrics['precision'])
                training_log['val_recall'].append(val_metrics['recall'])
                training_log['val_f1'].append(val_metrics['f1'])
                training_log['TP'].append(val_metrics['TP'])
                training_log['TN'].append(val_metrics['TN'])
                training_log['FP'].append(val_metrics['FP'])
                training_log['FN'].append(val_metrics['FN'])

                print(f"    Epoch {epoch+1}: Loss={avg_loss:.4f} (L_pred={avg_l_pred:.4f}, L_imp={avg_l_imp:.4f}) | "
                      f"Val Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")

                # Persist loss curves incrementally to avoid losing on crash
                try:
                    pd.DataFrame(training_log).to_csv(f"training_curves_{model_short}.csv", index=False)
                except Exception as e:
                    print(f"    Warning: could not persist training curves: {e}")

        print(f"\n[3/4] Training complete. Evaluating robustness...")

        # Final evaluation
        aa_metrics = calculate_aa_metrics(
            model, adv_generator,
            dev_texts[:EVAL_SUBSET_SIZE],
            dev_labels[:EVAL_SUBSET_SIZE],
            DEVICE
        )

        interp_metrics = {'kendall_tau': 0.0, 'top_k_intersection': 0.0}
        try:
            adv_texts_interp, _ = adv_generator.generate(
                model, dev_texts[:10], dev_labels[:10], num_examples=10
            )
            if adv_texts_interp:
                for i, adv_t in enumerate(adv_texts_interp):
                    if adv_t != dev_texts[i]:
                        interp_metrics = calculate_interpretability_metrics(
                            model, model_tokenizer,
                            dev_texts[i], adv_t,
                            DEVICE, k=10
                        )
                        break
        except Exception as e:
            print(f"  Warning: Could not calculate interpretability metrics: {e}")

        final_metrics = calculate_detailed_metrics(model, dev_loader, DEVICE)

        # Store results
        master_log['model_results'][current_model_name] = {
            'final_accuracy': final_metrics['accuracy'],
            'final_precision': final_metrics['precision'],
            'final_recall': final_metrics['recall'],
            'final_f1': final_metrics['f1'],
            'TP': final_metrics['TP'],
            'TN': final_metrics['TN'],
            'FP': final_metrics['FP'],
            'FN': final_metrics['FN'],
            'aa_accuracy': aa_metrics['aa_accuracy'],
            'attack_success_rate': aa_metrics['attack_success_rate'],
            'kendall_tau': interp_metrics['kendall_tau'],
            'top_k_intersection': interp_metrics['top_k_intersection']
        }

        master_log['training_curves'][current_model_name] = training_log

        print(f"\n[4/4] Saving model...")
        model_filename = f"flat_{current_model_name.split('/')[-1]}.pt"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as '{model_filename}'")

        print(f"\n{current_model_name} SUMMARY:")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {final_metrics['f1']:.4f}")
        print(f"  AA Accuracy: {aa_metrics['aa_accuracy']:.4f}")
        print(f"  Attack Success Rate: {aa_metrics['attack_success_rate']:.4f}")

        # Cleanup
        print(f"\nCleaning up GPU memory...")
        del model, optimizer, augmented_data_loader, train_dataset, dev_dataset, dev_loader
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleared.")

    print("\n" + "="*80)
    print("ALL MODELS BENCHMARKED SUCCESSFULLY")
    print("="*80)

    return master_log


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_visualizations(master_log):
    """Generate all visualization plots and CSV report."""
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("="*80)

    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 8)

    # Loss curves (only if we actually have training curves)
    if master_log.get('training_curves') and any(len(v.get('L_total', [])) > 0 for v in master_log['training_curves'].values()):
        print("\n[1/4] Plotting raw loss curves...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Training Loss Curves (Raw, No Smoothing)', fontsize=16, fontweight='bold')

        loss_types = ['L_total', 'L_pred', 'L_imp']
        colors = {'bert-base-uncased': 'blue', 'albert-base-v2': 'green', 'microsoft/deberta-base': 'red'}

        for idx, loss_type in enumerate(loss_types):
            ax = axes[idx]
            for model_name, training_log in master_log['training_curves'].items():
                if loss_type in training_log and len(training_log[loss_type]) > 0:
                    x_axis = list(range(1, len(training_log[loss_type]) + 1))
                    model_short = model_name.split('/')[-1]
                    ax.plot(x_axis, training_log[loss_type],
                           label=model_short, color=colors.get(model_name, 'black'),
                           linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(loss_type, fontsize=12)
            ax.set_title(f'{loss_type} Over Time', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(IMG_PATH + 'loss_curves.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {IMG_PATH}loss_curves.png")
        plt.close()
        step_offset = 0
    else:
        print("\n[1/3] No training curves available; skipping loss curves plot.")

    # Performance comparison
    print("\n[2/3] Plotting performance comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    model_names_short = [name.split('/')[-1] for name in master_log['model_results'].keys()]
    accuracies = [results['final_accuracy'] for results in master_log['model_results'].values()]
    aa_accuracies = [results['aa_accuracy'] for results in master_log['model_results'].values()]

    x_pos = np.arange(len(model_names_short))

    # Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, accuracies, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Final Accuracy (Clean Data)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names_short, rotation=15, ha='right')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # Robustness
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, aa_accuracies, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('After-Attack Accuracy', fontsize=12)
    ax2.set_title('Robustness (After-Attack Accuracy)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names_short, rotation=15, ha='right')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(IMG_PATH + 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(F"  Saved: {IMG_PATH}performance_comparison.png")
    plt.close()

    # Confusion matrix components
    print("\n[3/3] Plotting confusion matrix components...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confusion Matrix Components by Model', fontsize=16, fontweight='bold')

    confusion_metrics = ['TP', 'TN', 'FP', 'FN']
    metric_colors = ['green', 'blue', 'orange', 'red']

    for idx, (metric, color) in enumerate(zip(confusion_metrics, metric_colors)):
        ax = axes[idx // 2, idx % 2]
        values = [results[metric] for results in master_log['model_results'].values()]

        bars = ax.bar(x_pos, values, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names_short, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(IMG_PATH + 'confusion_matrix_components.png', dpi=300, bbox_inches='tight')
    print(F"  Saved: {IMG_PATH}confusion_matrix_components.png")
    plt.close()

    # Export CSV
    print("\nExporting benchmark report...")
    report_data = []
    for model_name, results in master_log['model_results'].items():
        report_data.append({
            'Model': model_name,
            'Accuracy': results['final_accuracy'],
            'Precision': results['final_precision'],
            'Recall': results['final_recall'],
            'F1-Score': results['final_f1'],
            'TP': results['TP'],
            'TN': results['TN'],
            'FP': results['FP'],
            'FN': results['FN'],
            'AA_Accuracy': results['aa_accuracy'],
            'Attack_Success_Rate': results['attack_success_rate'],
            'Kendall_Tau': results['kendall_tau'],
            'Top_K_Intersection': results['top_k_intersection']
        })

    benchmark_df = pd.DataFrame(report_data)
    benchmark_df.to_csv('benchmark_report.csv', index=False)
    print("  Saved: benchmark_report.csv")

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(benchmark_df.to_string(index=False))

    print("\n" + "="*80)
    print("BENCHMARKING SUITE COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print(f"  - {IMG_PATH}loss_curves.png")
    print(f"  - {IMG_PATH}performance_comparison.png")
    print(f"  - {IMG_PATH}confusion_matrix_components.png")
    print(f"  - {IMG_PATH}benchmark_report.csv")
    for model_name in MODELS:
        print(f"  - {IMG_PATH}flat_{model_name.split('/')[-1]}.pt")


# ==========================================================================
# EVALUATION-ONLY (RECOVERY) PATH
# ==========================================================================

def evaluate_saved_models(skip_robustness=False, models=None):
    """Evaluate existing checkpoints and regenerate report/plots without retraining.

    - Detects `flat_*.pt` in the CWD and maps them to known model IDs.
    - Recomputes clean metrics, AA metrics (unless skipped), and report.
    """

    print("="*80)
    print("RECOVERY MODE: EVALUATING SAVED CHECKPOINTS")
    print("="*80)

    # Build mapping from short name -> full model id
    known_map = {m.split('/')[-1]: m for m in KNOWN_MODELS}

    # If explicit list of models is provided, prefer that; else discover from files
    candidates = []
    if models is not None:
        for m in models:
            short = m.split('/')[-1]
            ckpt = f"flat_{short}.pt"
            if os.path.exists(ckpt):
                candidates.append((short, m, ckpt))
            else:
                print(f"  Warning: checkpoint not found for {m} -> {ckpt}")
    else:
        for path in glob.glob('flat_*.pt'):
            short = os.path.basename(path)[5:-3]  # strip 'flat_' and '.pt'
            full = known_map.get(short)
            if full is None:
                print(f"  Skipping unknown checkpoint '{path}' (no mapping to full model id)")
                continue
            candidates.append((short, full, path))

    if not candidates:
        raise RuntimeError("No usable checkpoints found. Nothing to evaluate.")

    # Load data (same split to maintain comparability)
    full_df = pd.read_csv(CSV_PATH)
    texts = full_df['text'].fillna('').tolist()
    labels = full_df['label_num'].tolist()

    _, temp_texts, _, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    dev_texts, _, dev_labels, _ = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    master_log = {
        'model_results': {},
        'training_curves': {}  # will remain empty in recovery mode
    }

    for short, full_id, ckpt in candidates:
        print("\n" + "-"*80)
        print(f"Evaluating checkpoint: {ckpt} (model: {full_id})")
        print("-"*80)

        tokenizer = AutoTokenizer.from_pretrained(full_id)
        model = FLATUniversalModel(full_id, tokenizer.vocab_size).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        dev_dataset = CustomTextDataset(dev_texts, dev_labels, tokenizer)
        dev_loader = DataLoader(dev_dataset, batch_size=ATTACK_BATCH_SIZE, shuffle=False)

        # Clean metrics
        final_metrics = calculate_detailed_metrics(model, dev_loader, DEVICE)

        # Robustness
        if skip_robustness:
            aa_metrics = {'aa_accuracy': 0.0, 'attack_success_rate': 0.0}
        else:
            adv_generator = AdversarialDataGenerator(full_id, tokenizer, PWWSRen2019)
            aa_metrics = calculate_aa_metrics(
                model, adv_generator,
                dev_texts[:EVAL_SUBSET_SIZE],
                dev_labels[:EVAL_SUBSET_SIZE],
                DEVICE
            )

        # Interpretability (best-effort)
        interp_metrics = {'kendall_tau': 0.0, 'top_k_intersection': 0.0}
        try:
            if not skip_robustness:
                adv_generator = AdversarialDataGenerator(full_id, tokenizer, PWWSRen2019)
                adv_texts_interp, _ = adv_generator.generate(
                    model, dev_texts[:10], dev_labels[:10], num_examples=10
                )
                if adv_texts_interp:
                    for i, adv_t in enumerate(adv_texts_interp):
                        if adv_t != dev_texts[i]:
                            interp_metrics = calculate_interpretability_metrics(
                                model, tokenizer,
                                dev_texts[i], adv_t,
                                DEVICE, k=10
                            )
                            break
        except Exception as e:
            print(f"  Warning: interpretability metrics failed: {e}")

        master_log['model_results'][full_id] = {
            'final_accuracy': final_metrics['accuracy'],
            'final_precision': final_metrics['precision'],
            'final_recall': final_metrics['recall'],
            'final_f1': final_metrics['f1'],
            'TP': final_metrics['TP'],
            'TN': final_metrics['TN'],
            'FP': final_metrics['FP'],
            'FN': final_metrics['FN'],
            'aa_accuracy': aa_metrics['aa_accuracy'],
            'attack_success_rate': aa_metrics['attack_success_rate'],
            'kendall_tau': interp_metrics['kendall_tau'],
            'top_k_intersection': interp_metrics['top_k_intersection']
        }

        print(f"  Clean Accuracy: {final_metrics['accuracy']:.4f}")
        if not skip_robustness:
            print(f"  AA Accuracy: {aa_metrics['aa_accuracy']:.4f}")

    # Visualizations and CSV
    generate_visualizations(master_log)

    print("\n✓ Recovery evaluation complete.")
    return master_log


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Model FLAT Benchmarking Suite')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate saved checkpoints only (no training)')
    parser.add_argument('--skip-robustness', action='store_true', help='Skip adversarial robustness eval to save time')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model ids to train/eval (defaults to MODELS constant)')
    args = parser.parse_args()

    selected_models = MODELS if args.models is None else [m.strip() for m in args.models.split(',') if m.strip()]

    if args.eval_only:
        print("="*80)
        print("MULTI-MODEL FLAT BENCHMARKING SUITE (EVAL-ONLY)")
        print("="*80)
        if args.models is None:
            print("\nModels to evaluate: auto-detected from flat_*.pt")
        else:
            print(f"\nModels to evaluate: {', '.join(selected_models)}")
        print(f"Evaluation subset size: {EVAL_SUBSET_SIZE}")
        print("\nNote: Loss curves will be skipped in recovery mode if logs are unavailable.\n")
        eval_models = None if args.models is None else selected_models
        evaluate_saved_models(skip_robustness=args.skip_robustness, models=eval_models)
    else:
        print("="*80)
        print("MULTI-MODEL FLAT BENCHMARKING SUITE")
        print("="*80)
        print(f"\nModels to benchmark: {', '.join(selected_models)}")
        print(f"FLAT iterations: {NUM_FLAT_ITERATIONS}")
        print(f"Epochs per iteration: {NUM_EPOCHS_PER_ITERATION}")
        print(f"Attack samples per iteration: {TRAIN_ATTACK_SAMPLES}")
        print(f"Evaluation subset size: {EVAL_SUBSET_SIZE}")
        print(f"\nEstimated runtime: ~24-36 hours on GPU\n")

        # Override MODELS if provided via CLI
        MODELS = selected_models

        # Run benchmarking
        master_log = run_benchmarking()

        # Generate visualizations
        generate_visualizations(master_log)

        print("\n✓ All tasks completed successfully!")
