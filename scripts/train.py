from distill_utils import get_device
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

class TeacherModel():
    def __init__(self, model_str : str):
        teacher_model_id = model_str
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)    
        self.model = AutoModelForCausalLM.from_pretrained(teacher_model_id)
        self.model.eval()

    def generate(self, input_text, max_length=50):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class MiniDeepSeek(nn.Module):
    def __init__(self, vocab_size, teacher_tokenizer, emb_dim=512, n_heads=8, n_layers=4, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # self.pos_emb = nn.Embedding(max_seq_len, emb_dim)

        self.tokenizer = teacher_tokenizer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.output_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        logits = self.output_head(x)
        return logits
    
    def generate(self, input_text, max_length=50):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        generated_ids = input_ids
        for _ in range(max_length):
            outputs = self.forward(generated_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    device = get_device()
    print(f"Running on: {device}")

    teacher_model = TeacherModel("deepseek-ai/deepseek-coder-1.3b-base")
    vocab_size = len(teacher_model.tokenizer)
    model = MiniDeepSeek(vocab_size, teacher_model.tokenizer)
    sout = model.generate("2+2=", max_length=10)
    tout = teacher_model.generate("2+2=")
    
    print(tout)
    print(sout)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

if __name__ == "__main__":
    main()

