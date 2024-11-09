# This file will handle model creation, training, and saving the model.
import torch
from transformers import (
    ViTFeatureExtractor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)
from accelerate import Accelerator
from data_loader import prepare_data
from transformers import TrainingArguments


def get_feature_extractor(model_name_or_path="google/vit-base-patch16-224-in21k"):
    return ViTFeatureExtractor.from_pretrained(model_name_or_path)


def get_model(
    num_labels, model_name_or_path="google/vit-base-patch16-224-in21k", labels=None
):
    return ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
    )


def train_model(
    train_dataset,
    eval_dataset,
    feature_extractor,
    model,
    output_dir="./vit-base-beans-demo-v5",
):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    return trainer
