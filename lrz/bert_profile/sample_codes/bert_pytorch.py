from datasets import load_dataset
import torch
imdb = load_dataset("imdb")

print(imdb["test"][0])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

print(model)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# prof = torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
#         record_shapes=True,
#         profile_memory=False,
#         with_stack=True,
#         use_cuda=False)

# prof.start()


# with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True
# ) as prof:
#     for step, batch_data in enumerate(train_loader):
#         if step >= (1 + 1 + 3) * 2:
#             break
#         train(batch_data)
#         prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

# with torch.backends.mkldnn.verbose(1):
trainer.train()