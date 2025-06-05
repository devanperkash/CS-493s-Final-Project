from dataloader import load_hellaswag, load_gsm8k
from distill_utils import get_device, count_parameters
from models import TeacherModel, StudentModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import datetime
import time

def train_student(logits_distillation, dataset, dataset_examples, num_epochs, generation_length):
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()

    device = get_device()
    print(f"Running on: {device}")

    log_file = f"logs/{dataset}_{"logit" if logits_distillation else "label"}_distill_log.txt"

    with open(log_file, 'w') as file:
        file.write(f"Starting... {datetime.datetime.now()}\nt1: {t1}")

    # Load data
    if dataset == "hellaswag":
        subset_train, subset_val = load_hellaswag(dataset_examples)
    else:
        subset_train, subset_val = load_gsm8k(dataset_examples)
    
    train_loader = DataLoader(subset_train, batch_size=8, shuffle=True)
    val_loader   = DataLoader(subset_val, batch_size=8, shuffle=False)

    # Get models set up
    teacher_model = TeacherModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    teacher_model.model.to(device)
    teacher_model.model.eval()

    student_model = StudentModel(teacher_model.tokenizer).to(device)
    student_model.train()

    print('Vocab length:', len(teacher_model.tokenizer))
    print("Size of student model (# of params):")
    count_parameters(student_model)

    # Prep for training
    T = 1.0  # distillation temperature; can raise to 2.0 or 4.0 for “softer” teacher probs
    loss_func = nn.KLDivLoss(reduction="batchmean")
    ce_loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    # Track average loss per epoch for plotting later
    epoch_losses = []

    # Run models
    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in train_loader:
            if logits_distillation:
                # a) Teacher forward (no grads):
                with torch.no_grad():
                    t_seqs, t_logits = teacher_model.get_teacher_y(batch, max_length=generation_length)
                    t_logits = t_logits.to(device)

                # b) Student forward (grad OK):
                s_seqs, s_logits = student_model.get_student_y_hat(batch, max_length=generation_length)
                s_logits = s_logits.to(device)

                # c) Flatten and compute distillation loss:
                B, L, V = t_logits.size()
                t_flat = t_logits.view(B * L, V)  # (B*L, V)
                s_flat = s_logits.view(B * L, V)  # (B*L, V)

                teacher_probs    = torch.softmax(t_flat / T, dim=-1).detach()
                student_logprobs = torch.log_softmax(s_flat / T, dim=-1)
                loss = loss_func(student_logprobs, teacher_probs) * (T * T)
            else:
                # a) Teacher generates token ids (no grads):
                with torch.no_grad():
                    t_seqs, _ = teacher_model.get_teacher_y(batch, max_length=generation_length)
                    t_seqs = t_seqs.to(device)

                # b) Student forward (grad OK):
                s_seqs, s_logits = student_model.get_student_y_hat(batch, max_length=generation_length)
                s_logits = s_logits.to(device)

                # c) Compute cross-entropy loss:
                # s_logits: (B, L, V), t_seqs: (B, L+input_len)
                # We want to align the generated tokens (after input prompt)
                # For simplicity, align on the last L tokens
                # (Assumes input prompt is same for both)
                B, L, V = s_logits.size()
                # Use the last L tokens from t_seqs as targets
                targets = t_seqs[:, -L:]
                loss = ce_loss_func(s_logits.view(B * L, V), targets.reshape(B * L))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch}   Avg {'KL' if logits_distillation else 'CE'} loss: {avg_loss:.5f}")

        #
        # e) Example: generate one batch from `val_loader`
        #
        student_model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            teacher_out = teacher_model.generate_text(sample_batch[0:3], max_length=generation_length)
            student_out = student_model.generate_text(sample_batch[0:3], max_length=generation_length)
        
        with open(log_file, 'a', encoding="utf-8") as file:
            file.write(f"\nEpoch {epoch}   Avg {'KL' if logits_distillation else 'CE'} loss: {avg_loss:.5f}")
            file.write("\n---------------------")
            file.write(f"\nTeacher says (first example): {teacher_out[0]}")
            file.write(f"\nStudent says (first example): {student_out[0]}")
            file.write(f"\nTeacher says (second example): {teacher_out[1]}")
            file.write(f"\nStudent says (second example): {student_out[1]}")
            file.write(f"\nTeacher says (third example): {teacher_out[2]}")
            file.write(f"\nStudent says (third example): {student_out[2]}")
            file.write("\n---------------------")
        file.close()

        student_model.train()
    
    # ── Now plot epoch_losses ──
    epochs = list(range(1, len(epoch_losses) + 1))
    plt.figure()
    plt.plot(epochs, epoch_losses, marker="o", linestyle="-")
    plt.title(f"Training Loss Over Epochs: ({dataset}: {'LOGITS' if logits_distillation else 'LABELS'})")
    plt.xlabel("Epoch")
    plt.ylabel(f"Average {'KL' if logits_distillation else 'CE'} Loss")
    plt.grid(True)
    plt.savefig(f"plots/{dataset}_{"logit" if logits_distillation else "label"}_loss.png")

    # After all epochs are finished:
    save_path = f"saved_models/{dataset}_{"logit" if logits_distillation else "label"}_student.pt"
    torch.save(student_model.state_dict(), save_path)
    print(f"Saved student checkpoint to {save_path}")

    t2 = time.time()
    max_mem = torch.cuda.max_memory_allocated()
    max_mem = round(max_mem / (1024 ** 3), 2)

    with open(log_file, 'a') as file:
        file.write(f"\nt2: {t2}")
        file.write(f"\nDistillation took {t2-t1} seconds")
        file.write(f"\nMaximum memory allocated: {max_mem} GB")

    file.close()


if __name__ == "__main__":
    train_student(False, "hellaswag", 4, 10, 10) # label distillation
    train_student(False, "gsm8k", 4, 10, 10) # label distillation
    train_student(True, "hellaswag", 4, 10, 10) # logits distillation
    train_student(True, "gsm8k", 4, 10, 10) # logits distillation

