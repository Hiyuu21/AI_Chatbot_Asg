import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import time
import json
import matplotlib.pyplot as plt

from transformer import TransformerModel, Tokenizer
from preprocess import load_and_clean_data, group_aware_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report


"""
Dataset Class
"""

class FAQDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # PyTorch needs to know exactly how many items are in the dataset
        return len(self.data_list)

    def __getitem__(self, idx):
        text, target_id = self.data_list[idx]
        
        token_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        x = torch.tensor(token_ids)
        y = torch.tensor(target_id)
        
        return x, y


"""
Plot train vs test performance trend lines

Draws loss and accuracy curves side by side. 
The gap between the train and test lines is exactly what makes overfitting visible: train loss
keeps dropping / train accuracy keeps climbing while test loss flattens
out or starts climbing again - that widening gap is the overfit signal.
"""

def plot_training_curves(history, save_path="transformer_training_curves.png"):
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss subplot ---
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", marker="o")
    ax.plot(epochs, history["test_loss"], label="Test Loss", color="tab:orange", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Test Loss")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Accuracy subplot ---
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Accuracy", color="tab:blue", marker="o")
    ax.plot(epochs, history["test_acc"], label="Test Accuracy", color="tab:orange", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Train vs Test Accuracy")
    ax.set_ylim(0,100)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Training Performance Trend (Overfit Check)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"-- Saved train/test performance chart to '{save_path}'.")

"""
Shared inference loop
"""

def run_inference(model, loader, device, criterion=None):
    """
    Runs 'model' over every batch in 'loader' with gradients disabled.

    criterion: if provided, also accumulates loss (used during training-time
    checkpoint evaluation). If None, total_loss is returned as None (used by
    the final diagnostic pass, which only needs predictions/targets).
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0 if criterion is not None else None

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            if criterion is not None:
                total_loss += criterion(logits, targets).item()

            predicted_classes = torch.argmax(logits, dim=-1)
            all_preds.extend(predicted_classes.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return all_preds, all_targets, total_loss


"""
Transformer model train
"""

def train_transformer(eval_every=10, patience=3):
    """
    eval_every: how often (in epochs) to run the test-set evaluation and
    record a point on the train/test curves. Set to 1 for a fine-grained
    view of exactly when train/test start to diverge (roughly doubles
    training time, since it adds a full test-set forward pass every
    epoch instead of every `eval_every` epochs). Default of 10 matches
    the original behaviour.

    patience: number of consecutive eval checkpoints (spaced eval_every
    epochs apart) allowed without test loss improving, before training
    stops early. The best-performing checkpoint (lowest test loss) is
    kept and used for the final model, diagnostics, and saved weights -
    not necessarily the last epoch. Set to a large number (e.g. 9999)
    to effectively disable early stopping and always run all EPOCHS.
    """

    CONFIG = {
        "max_length": 20,
        "batch_size": 16,
        "d_model": 64,
        "ff_hidden_dim": 128,
        "num_heads": 8,
        "num_blocks": 2,
        "embedder_dropout": 0.3,
        "ffn_dropout": 0.3,
        "classifier_dropout": 0.3,
        "epochs": 200,
        "learning_rate": 0.0003,
        "weight_decay": 1e-2,
        "label_smoothing": 0.1,
        "lr_scheduler_step_size": 30,
        "lr_scheduler_gamma": 0.5,
    }
    MAX_LENGTH = CONFIG["max_length"]
    BATCH_SIZE = CONFIG["batch_size"]
    EPOCHS = CONFIG["epochs"]

    raw_data, intent_to_response = load_and_clean_data("faq_idk7.xlsx")
    if not raw_data: return

    with open("transformer_responses.json", "w", encoding="utf-8") as f:
        json.dump(intent_to_response, f, indent=4)

    raw_train, raw_test, fallback_intents = group_aware_split(raw_data, test_size=0.20, seed=42)

    # Extract all unique intents and sort them alphabetically for consistency
    unique_intents = sorted(list(set([row[1] for row in raw_data])))

    intent_to_id = {intent:idx for idx, intent in enumerate(unique_intents)}
    id_to_intent = {idx:intent for idx, intent in enumerate(unique_intents)}

    num_intents = len(unique_intents)
    print(f"-- Successfully mapped {num_intents} unique intents.")

    train_sentences = [row[0] for row in raw_train]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(train_sentences)

    train_mapped = [(row[0], intent_to_id[row[1]]) for row in raw_train]
    test_mapped = [(row[0], intent_to_id[row[1]]) for row in raw_test]

    train_dataset = FAQDataset(train_mapped,tokenizer,MAX_LENGTH)
    test_dataset = FAQDataset(test_mapped,tokenizer,MAX_LENGTH)

    # The dataloader will automatically groups the data into batches of 16 words
    # shuffle=True will ensure the model doesn't memorize the order of the dataset
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)


    with open("transformer_vocab.json",'w', encoding="utf-8") as f:
        json.dump({
            "vocab":tokenizer.vocab,
            "num_intents":num_intents,
            "intent_to_id":intent_to_id,
            "id_to_intent":id_to_intent
        }, f, indent=4)
    print("-- Vocabulary and intent mappings saved as 'transformer_vocab.json'.")

    with open("transformer_config.json", 'w', encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4)
    print("-- Architecture/training hyperparameters saved as 'transformer_config.json'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-- Using computing device: {device}")

    model = TransformerModel(
        vocab_size=len(tokenizer.vocab) + 50,
        num_intents=num_intents,
        d_model=CONFIG["d_model"],
        max_length=MAX_LENGTH,
        ff_hidden_dim=CONFIG["ff_hidden_dim"],
        num_heads=CONFIG["num_heads"],
        num_blocks=CONFIG["num_blocks"],
        embedder_dropout=CONFIG["embedder_dropout"],
        ffn_dropout=CONFIG["ffn_dropout"],
        classifier_dropout=CONFIG["classifier_dropout"],
    )

    if os.path.exists("trained_transformer.pth"):
        print("-- Found existing train path. Loading previous knowledge.\n")
        print('='*60 + "\n")
        model.load_state_dict(torch.load("trained_transformer.pth", weights_only=True))
    else:
        print("-- No previous knowledge found. Starting from scratch.\n")
        print('='*60 + "\n")

    model.to(device)  

    # Setup the Grader and Adjuster
    # Add Label Smoothing to prevent overconfident target memorization
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # Use AdamW with stronger weight decay (1e-2 instead of 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    model.train()

    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG["lr_scheduler_step_size"], gamma=CONFIG["lr_scheduler_gamma"]
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    best_test_loss = float("inf")
    best_epoch = None
    best_model_state = None
    checkpoints_without_improvement = 0
    stopped_early = False

    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0

        train_preds = []
        train_targets = []
        
        # We now iterate through the train_loader instead of the raw data list.
        # batch_inputs shape: [16, 20] (16 sentences, 20 tokens each)
        # batch_targets shape: [16] (16 correct intent IDs)
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Clear old gradients
            optimizer.zero_grad()
            
            # Forward Pass: Ask the model to predict the whole batch at once
            predictions = model(batch_inputs)
            
            # Calculate the error for the whole batch
            loss = criterion(predictions, batch_targets)
            
            # Backward Pass: Calculate how to fix the weights
            loss.backward()
            
            # Apply the fixes
            optimizer.step()
            total_loss += loss.item()

            # Save predictions and targets for accuracy calculation
            predicted_classes = torch.argmax(predictions, dim=-1)
            train_preds.extend(predicted_classes.detach().cpu().numpy())
            train_targets.extend(batch_targets.cpu().numpy())

        scheduler.step()
            
 
        if (epoch + 1) % eval_every == 0:
            # We divide by the length of the loader (number of batches), not the raw data
            average_loss = total_loss / len(train_loader)

            # Calculate Train Metrics
            train_correct = sum([1 for p, t in zip(train_preds, train_targets) if p==t])
            train_accuracy = (train_correct / len(raw_train)) * 100
            
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                train_targets, train_preds, average='weighted', zero_division=0
            )

            all_preds, all_targets, test_loss = run_inference(model, test_loader, device, criterion=criterion)
            
            test_correct = sum([1 for p, t in zip(all_preds, all_targets) if p==t])
            test_accuracy = test_correct / len(raw_test) * 100

            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average='weighted', zero_division=0
            )

            avg_test_loss = test_loss / len(test_loader)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(average_loss)
            history["test_loss"].append(avg_test_loss)
            history["train_acc"].append(train_accuracy)
            history["test_acc"].append(test_accuracy)

            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"TRAIN -> Loss: {average_loss:.4f} | Acc: {train_accuracy:6.2f}% | Prec: {train_precision:.4%} | Rec: {train_recall:.4%} | F1: {train_f1:.4%}")
            print(f"TEST  -> Loss: {avg_test_loss:.4f} | Acc: {test_accuracy:6.2f}% | Prec: {test_precision:.4%} | Rec: {test_recall:.4%} | F1: {test_f1:.4%}")

            # Early stopping: track the checkpoint with the lowest test loss seen so far. 
            # Test loss (not accuracy) is used because it keeps falling even after accuracy plateaus, 
            # making it a more sensitive early signal of the model starting to overfit.

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                checkpoints_without_improvement = 0
                print(f"   -> New best test loss ({best_test_loss:.4f}). Checkpoint saved.")
            else:
                checkpoints_without_improvement += 1
                print(f"   -> No improvement for {checkpoints_without_improvement}/{patience} checkpoint(s).")

            print('-'*100 + "\n")
            model.train()

            if checkpoints_without_improvement >= patience:
                print(f"-- Early stopping: no test loss improvement for {patience} "
                      f"consecutive checkpoints. Best was epoch {best_epoch} "
                      f"(test loss {best_test_loss:.4f}).\n")
                stopped_early = True
                break

    end_time = time.time()
    print(f"--- Transformer Training Time: {(end_time - start_time) / 60:.4f} minutes ---")
        
    print("\nTraining Complete." + (" (stopped early)" if stopped_early else ""))

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"-- Restored best checkpoint: epoch {best_epoch}, test loss {best_test_loss:.4f}.")
        print(f"-- (Final epoch reached: {epoch + 1}/{EPOCHS} — the saved model is the "
              f"best checkpoint, not necessarily the last epoch trained.)")
    else:
        print("-- WARNING: no checkpoint was recorded (eval_every may be larger than "
              "EPOCHS) — saving the final epoch's weights instead.")

    torch.save(model.state_dict(), "trained_transformer.pth")
    print("Saved as 'trained_transformer.pth'.")

    LOG_DIR = "training_logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    plot_training_curves(history, save_path=os.path.join(LOG_DIR, "transformer_training_curves.png"))

    # Final diagnostic: is high test accuracy real, or leakage from the random-fallback intents 
    # (single SeedGroup, near-duplicate paraphrases split across train/test)? 
    # Break down accuracy by split-method.
    print("\n-- Running final diagnostic evaluation...")
    final_preds, final_targets, _ = run_inference(model, test_loader, device)


    test_intent_strings = [row[1] for row in raw_test]

    group_protected_correct, group_protected_total = 0, 0
    fallback_correct, fallback_total = 0, 0

    for pred, target, intent_str in zip(final_preds, final_targets, test_intent_strings):
        correct = int(pred == target)
        if intent_str in fallback_intents:
            fallback_correct += correct
            fallback_total += 1
        else:
            group_protected_correct += correct
            group_protected_total += 1

    print("\n-- Test accuracy by split-method:")
    if group_protected_total:
        gp_acc = group_protected_correct / group_protected_total * 100
        print(f"   Group-protected intents: "
              f"{gp_acc:.2f}% ({group_protected_correct}/{group_protected_total})")
    else:
        print("   Group-protected intents: none in this dataset.")
    if fallback_total:
        fb_acc = fallback_correct / fallback_total * 100
        print(f"   Random-fallback intents (single SeedGroup, leakage risk): "
              f"{fb_acc:.2f}% ({fallback_correct}/{fallback_total})")
    else:
        print("   Random-fallback intents: none.")

    if group_protected_total and fallback_total:
        gap = fb_acc - gp_acc
        print(f"   Gap: {gap:+.2f} points. A large positive gap suggests the fallback "
              f"intents' accuracy is inflated by near-duplicate leakage rather than "
              f"genuine generalisation.")

    # Full per-intent breakdown, saved to file (84 classes is too much for console)
    target_names = [id_to_intent[i] for i in range(num_intents)]
    report = classification_report(
        final_targets, final_preds, labels=list(range(num_intents)),
        target_names=target_names, zero_division=0, digits=4
    )
    with open(os.path.join(LOG_DIR, "transformer_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
    print("-- Full per-intent precision/recall/F1 saved to 'transformer_classification_report.txt'.")

if __name__ == "__main__":
    # eval_every=10 (default): faster, coarser curve (10 points over 100 epochs)
    # eval_every=1: slower (~2x), but shows exactly which epoch train/test start to diverge
    # patience=3: stop after 3 eval checkpoints (30 epochs at eval_every=10) with no
    #   test loss improvement, and keep the best checkpoint rather than the last epoch.
    #   Increase for a more patient search, or set very high (e.g. 9999) to disable.
    train_transformer(eval_every=10, patience=3)