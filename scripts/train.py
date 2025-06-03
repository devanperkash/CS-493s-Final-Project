from dataloader import load_hellaswag, load_gsm8k
from distill_utils import get_device, count_parameters
from models import TeacherModel, StudentModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_student(logits_distillation:bool = False):
    device = get_device()
    print(f"Running on: {device}")

    # Load data
    subset_train, subset_val = load_gsm8k(3)
    train_loader = DataLoader(subset_train, batch_size=8, shuffle=True)
    val_loader   = DataLoader(subset_val,   batch_size=8, shuffle=False)

    # Get models set up
    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base")
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
    for epoch in range(5):
        total_loss = 0.0

        for batch in train_loader:
            if logits_distillation:
                # a) Teacher forward (no grads):
                with torch.no_grad():
                    t_seqs, t_logits = teacher_model.get_teacher_y(batch, max_length=50)
                    t_logits = t_logits.to(device)

                # b) Student forward (grad OK):
                s_seqs, s_logits = student_model.get_student_y_hat(batch, max_length=50)
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
                    t_seqs, _ = teacher_model.get_teacher_y(batch, max_length=50)
                    t_seqs = t_seqs.to(device)

                # b) Student forward (grad OK):
                s_seqs, s_logits = student_model.get_student_y_hat(batch, max_length=50)
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
            teacher_out = teacher_model.generate_text(sample_batch, max_length=50)
            student_out = student_model.generate_text(sample_batch, max_length=50)

        print("Teacher says (first example):", teacher_out[0])
        print("Student says (first example):", student_out[0])
        print("---")

        student_model.train()
    
    # ── Now plot epoch_losses ──
    epochs = list(range(1, len(epoch_losses) + 1))
    plt.figure()
    plt.plot(epochs, epoch_losses, marker="o", linestyle="-")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(f"Average {'KL' if logits_distillation else 'CE'} Loss")
    plt.grid(True)
    plt.show()

    # After all epochs are finished:
    save_path = "student_model.pt"
    torch.save(student_model.state_dict(), save_path)
    print(f"Saved student checkpoint to {save_path}")

def test():
    subset_train, subset_val = load_gsm8k(2000) # Modify number of samples as needed

    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base")
    student_model = StudentModel(teacher_model.tokenizer)
    z = teacher_model.generate_text(["What is your name?", "What is 9+9?"])
    print(z)



if __name__ == "__main__":
    train_student(logits_distillation=True) # logits distillation
    # train_student(logits_distillation=False) # label distillation
    # test()

