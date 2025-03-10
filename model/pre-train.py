# ----------------------------------
# Pre-training by MLM
# ----------------------------------


# Some imports
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os
import torch
from native_sparse_attention_pytorch import SparseAttention

# Custom RoBERTa with SparseAttention
class RobertaForMaskedLMWithSparseAttention(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace the standard attention with SparseAttention in all layers
        for i, layer in enumerate(self.roberta.encoder.layer):
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            
            # Create sparse attention module
            sparse_attn = SparseAttention(
                dim=hidden_size,
                dim_head=hidden_size // num_heads,
                heads=num_heads,
                sliding_window_size=4,
                compress_block_size=8,
                selection_block_size=8,
                num_selected_blocks=4
            )
            
            # Create a forward hook to replace the attention operation
            def create_forward_hook(sparse_attention):
                def hook(module, input, output):
                    hidden_states = input[0]
                    attention_mask = input[1] if len(input) > 1 else None
                    
                    # Apply sparse attention
                    attended = sparse_attention(hidden_states)
                    
                    # Return tuple to match the expected output format
                    return (attended, None)  # None for attention weights
                return hook
            
            # Register the hook
            layer.attention.self.register_forward_hook(create_forward_hook(sparse_attn))

# Initialise the tokenizer of antibody
tokenizer = RobertaTokenizerFast.from_pretrained(
    "tokenizer"
)

# Initialise the data collator, which is necessary for batching
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
text_datasets = {
    "train": ['assets/mlm-train.txt'],
    "eval": ['assets/mlm-val.txt'],
    "test": ['assets/mlm-test.txt']
}

dataset = load_dataset("text", data_files=text_datasets)
tokenized_dataset = dataset.map(
    lambda z: tokenizer(
        z["text"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    num_proc=1,
    batched=True,
    remove_columns=["text"],
)

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    For MLM, the primary metrics are loss and perplexity.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    
    # Mask out padding tokens
    mask = labels != -100
    labels_filtered = labels[mask]
    predictions_filtered = predictions[mask]
    
    # Calculate accuracy
    accuracy = (predictions_filtered == labels_filtered).mean()
    
    # Calculate perplexity
    loss = eval_pred.loss
    perplexity = float(torch.exp(torch.tensor(loss)))
    
    return {
        "accuracy": float(accuracy),
        "perplexity": perplexity,
        "loss": loss
    }

NanoBERTa_config = {
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_size": 768,
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 64,
    "max_steps": 225000,
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}
# Initialise the model
model_config = RobertaConfig(
    vocab_size=NanoBERTa_config.get("vocab_size"),
    hidden_size=NanoBERTa_config.get("hidden_size"),
    max_position_embeddings=NanoBERTa_config.get("max_position_embeddings"),
    num_hidden_layers=NanoBERTa_config.get("num_hidden_layers", 12),
    num_attention_heads=NanoBERTa_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLMWithSparseAttention(model_config)
# construct training arguments
args = TrainingArguments(
    output_dir="test_nsa",
    overwrite_output_dir=True,
    per_device_train_batch_size=NanoBERTa_config.get("batch_size", 32),
    per_device_eval_batch_size=NanoBERTa_config.get("batch_size", 32),
    max_steps=225000,
    save_steps=2500,
    logging_steps=2500,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    learning_rate=1e-4,
    gradient_accumulation_steps=NanoBERTa_config.get("gradient_accumulation_steps", 1),
    fp16=True,
    eval_strategy="steps",
    seed=42,  # uses a default seed(42)
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    compute_metrics=compute_metrics
)

trainer.train()
# Predict MLM performance on the test dataset
# result = trainer.predict(tokenized_dataset['test'])
# print(result)
