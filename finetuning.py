import os

from peft import LoraConfig
from tinycss2 import tokenizer
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

from common.dataset import create_dataset, generate_splits, load_data
from common.constants import INSTRUCTION_TEMPLATE, RESPONSE_TEMPLATE
from config import DEVICE, MODEL_ID


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 token=os.environ["HF_TOKEN"],
                                                 quantization_config=bnb_config,
                                                 device_map={"": DEVICE})
    return model, tokenizer


if __name__ == '__main__':
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    collator = DataCollatorForCompletionOnlyLM(instruction_template=INSTRUCTION_TEMPLATE,
                                               response_template=RESPONSE_TEMPLATE,
                                               tokenizer=tokenizer,
                                               mlm=False)

    model, tokenizer = load_model_and_tokenizer()

    data_path = "/kaggle/input/mbti-type/mbti_1.csv"
    data = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = generate_splits(data)

    ds_train = create_dataset(X_train, y_train)
    ds_val = create_dataset(X_val, y_val)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        dataset_text_field="text",
        args=transformers.TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            evaluation_strategy='steps',
            eval_steps=0.1,
            warmup_steps=2,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1000,
            output_dir="outputs",
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_steps=0.1,
            gradient_checkpointing=False
        ),
        peft_config=lora_config,

    )
    trainer.train()

    trainer.save_model("model")
