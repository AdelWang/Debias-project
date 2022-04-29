#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch.nn.functional as F

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from model__init__ import BertClassifier

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer_self,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertModel,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from model__init__ import finetune_layer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    weight_mse: float = field(
        default=1,
        metadata={
            "help": "The weight of the mse loss."
        },
    )
    weight_cl: float = field(
        default=1,
        metadata={
            "help": "The weight of the contrastive loss."
        },
    )
    temp: float = field(
        default=0.1,
        metadata={
            "help": "The temperature of the contrastive learning."
        },
    )
    learning_rate_classifier: float = field(
        default=None,
        metadata={
            "help": "The learning_rate of the classifier if use a different optimizer."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_patience: int = field(
        default=10,
        metadata={
            "help": "The max patience of training."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    eval_metric: Optional[str] = field(
        default='acc', metadata={"help": "The metric you choose to evaluate the model"}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    extra_model_path: str = field(
        default = None, metadata={"help": "Path to extra model."}
    )
    use_classifier_pred: bool = field(
        default = True,
        metadata={"help": "Choose to default the classifier or not"}
    )
    use_extra_model: bool = field(
        default = False,
        metadata={"help": "Choose to default the extra model or not"}
    )
    use_attention: bool = field(
        default = False,
        metadata={"help": "Use the attention matrix to predict or not."}
    )
    use_predictor: bool = field(
        default = False,
        metadata={"help": "Use the overlap to predict the label."}
    )
    sep: bool = field(
        default=False,
        metadata={"help": "Choose to seperate the bert model and the finetune layer"}
    )
    cl_type: str = field(
        default = None, metadata={"help": "Using the sentence / token level contrastive learning."}
    )
    activation: str = field(
        default = None,
        metadata={"help": "Choose the activation function of the classifier."}
    )
    layer: str = field(
        default = None,
        metadata={"help": "Choose the activation function of the classifier."}
    )
    
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    smoother: bool = field(
        default=False,
        metadata={
            "help":"Using smoother mechanism based on overlap."
        },
    )
    
WEIGHTS = 0.

def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

class InputFeatures(object):
    def __init__(self, choices_features):
        input_ids, input_mask, segment_ids, position_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'token_type_ids': segment_ids,
            'attention_mask': input_mask,
            'position_ids': position_ids
        }
        
class classifier_pred(nn.Module):
    def __init__(self, config, activation=nn.Sigmoid(),scale=2):
        super(classifier_pred, self).__init__()
        self.fc_pred_hidden_1 = nn.Linear(config.hidden_size, config.hidden_size * scale)
        self.fc_pred_hidden_2 = nn.Linear(config.hidden_size * scale, config.hidden_size)
        self.fc_pred = nn.Linear(config.hidden_size, 1)
        self.activation_hidden = nn.Tanh()
        self.activation = activation
        #self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_hidden_states, y=None, loss_fn=nn.MSELoss(reduction='mean')):
        if y is not None:
            y = y.reshape(-1,1)
        feature = input_hidden_states
        #last_hidden = input_hidden_states[-1]
        #feature = torch.mean(last_hidden,axis= 1)
        h_hidden = self.fc_pred_hidden_1(self.dropouts(feature))
        pred_hidden = self.activation_hidden(h_hidden)
        h_hidden = self.fc_pred_hidden_2(self.dropouts(pred_hidden))
        pred_hidden = self.activation_hidden(h_hidden)
        h = self.fc_pred(self.dropouts(pred_hidden))
        predict = self.activation(h)
        if loss_fn is not None:
            loss = loss_fn(predict.float(), y.float())
            return predict , loss 
        return predict
        
class classifier_pred_attention(nn.Module):
    def __init__(self, max_seq_length = 128, scale=2, activation=nn.Sigmoid()):
        super(classifier_pred_attention, self).__init__()
        self.weights_layer = nn.Parameter(torch.rand(12, 1))
        self.fc_pred_hidden_1 = nn.Linear(12 * max_seq_length, scale * 12 * max_seq_length)
        self.fc_pred_hidden_2 = nn.Linear(scale * 12 * max_seq_length, 12 * max_seq_length)
        self.fc_pred = nn.Linear(12 * max_seq_length, 1)
        self.BN_1 = nn.BatchNorm1d(scale * 12 * max_seq_length)
        self.BN_2 = nn.BatchNorm1d(12 * max_seq_length)
        self.activation_hidden = nn.Tanh()
        self.activation = activation
        self.max_seq_length = max_seq_length
        self.dropouts = nn.Dropout(0.1)

    def forward(self, input_attentions, y=None, loss_fn=nn.MSELoss(reduction='mean')):
        if y is not None:
            y = y.reshape(-1,1)
        atten_sample = input_attentions[0]
        batch_size = len(atten_sample)
        seq_length_batch = atten_sample.size(-1)
        pad_length = self.max_seq_length - seq_length_batch
        pool_layer = nn.MaxPool2d((1, seq_length_batch))
        logits = 0.
        atten = F.softmax(self.weights_layer, dim=0) 
        for i in range(len(input_attentions)):
            pooled_attentions = pool_layer(input_attentions[i]).sum(dim=-1)
            feature = F.pad(pooled_attentions, pad = (0, pad_length), mode = 'constant').reshape(batch_size, -1) #drop head to the feature space
            h_hidden = self.fc_pred_hidden_1(self.dropouts(feature))
            pred_hidden = self.BN_1(h_hidden)
            pred_hidden = self.activation_hidden(h_hidden)
            h_hidden = self.fc_pred_hidden_2(self.dropouts(pred_hidden))
            pred_hidden = self.BN_2(h_hidden)
            pred_hidden = self.activation_hidden(h_hidden)
            h = self.fc_pred(pred_hidden)
            predict = self.activation(h)
            logits += atten[i] * predict
        loss = loss_fn(logits.float(), y.float())
        if y is not None:
            return logits , loss 
        return logits 

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("/home/newdisk/wkh/code/project/datasets/datasets/glue", data_args.task_name, cache_dir=model_args.cache_dir)
        print(raw_datasets)
        self_data = False
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        self_data = False
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        self_data = True
        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    use_cl_sentence = True if model_args.cl_type == "sentence" else False
    use_cl_token = True if model_args.cl_type == "token" else False
    smoother_type = "overlap" if model_args.smoother else None
    if model_args.use_attention:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            output_hidden_states = use_cl_token,
            output_attentions = model_args.use_attention,
        )
    elif model_args.use_predictor:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            output_hidden_states = False,
            output_attentions = False,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            output_hidden_states = model_args.use_classifier_pred,
            output_attentions = use_cl_token,
            problem_type = smoother_type,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if not model_args.sep:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        finetune_cls = None
    else:
        model = BertModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        ).cuda()
        
        finetune_cls = finetune_layer(config=config).cuda()
    if model_args.use_extra_model:
        model_extra = AutoModelForSequenceClassification.from_pretrained(
            model_args.extra_model_path if model_args.extra_model_path else model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_extra.cuda()
        extra_model = True
        if model_args.extra_model_path:
            freeze_model = True
        else:
            freeze_model = False
    else:
        freeze_model = False
        model_extra = None
        extra_model = False
    if data_args.task_name is None and model_args.use_classifier_pred:
        if model_args.use_attention:
            classifier = classifier_pred_attention().cuda()
        elif model_args.use_predictor:
            classifier = predictor().cuda()
        else:
            if model_args.activation == "relu":
                activation_fn = nn.ReLU()
                classifier = classifier_pred(config=config,activation=activation_fn).cuda()
            else:
                classifier = classifier_pred(config=config).cuda()
        use_classifier = True
    else: 
        classifier = None
        use_classifier = False
    # Preprocessing the raw_datasets
    '''
    if classifier is not None:
        classifier_path = os.path.join("./classifier/", "1e-3.bin")
        state_dict_classifier = torch.load(classifier_path)
        classifier.load_state_dict(state_dict_classifier)
    '''
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        name_label_overlap = ["label","overlap"]
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name not in name_label_overlap] 
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    '''
    max_first_length = config.max_first_length
    max_second_length = config.max_second_length
    '''

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key]) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    def preprocess_function_deformer(examples):
        choices_features = []
        text_1 = examples[sentence1_key]
        token_1 = ["CLS"] + text_1 + ["SEP"]
        text_ids_1 = tokenizer.convert_tokens_to_ids(token_1)
        padding_length_1 = max_first_length - len(text_ids_1) + 2
        text_ids_1 += ([0] * padding_length_1)
        attention_mask_1 = ([1] * len(text_ids_1))
        token_type_ids_1 = ([0] * len(text_ids_1))
        position_ids_1 = torch.arange(max_first_length)
        choices_features.append((text_ids_1, attention_mask_1, token_type_ids_1, position_ids_1))
        result_1 = InputFeatures(choices_features)
        result_1["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        if sentence2_key is not None:
            choices_features_2 = []
            text_2 = examples[sentence2_key]
            token_2 = text_2 + ["SEP"]
            text_ids_2 = tokenizer.convert_tokens_to_ids(token_2)
            padding_length_2 = max_second_length - len(text_ids_2) + 1
            text_ids_2 += ([0] * padding_length_2)
            attention_mask_2 = ([0] * len(text_ids_2))
            token_type_ids_2 = ([1] * len(text_ids_2))
            position_ids_2 = torch.arange(max_second_length)
            choices_features_2.append((text_ids_2, attention_mask_2, token_type_ids_2, position_ids_2))
            result_2 = InputFeatures(choices_features_2)
        return result_1 if sentence2_key is None else result_1, result_2
        
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("/home/newdisk/wkh/code/project/datasets/metrics/glue", data_args.task_name)
    else:
        metric = load_metric("/home/newdisk/wkh/code/project/datasets/metrics/accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            result = pearson_and_spearman(preds=preds, labels=p.label_ids)
            return result
        else:
            result = metric.compute(predictions=preds, references=p.label_ids)
            return result

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    if "heuristics" in data_args.validation_file:
        bias_type = "hans"
        print("****************** hans verified ******************")
    else:
        bias_type = None
    # Initialize our Trainer
    trainer = Trainer_self(
        model=model,
        classifier = classifier,
        model_extra = model_extra,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        use_classifier = use_classifier,
        extra_model = extra_model,
        weight_mse = data_args.weight_mse,
        freeze_model = freeze_model,
        eval_metric = data_args.eval_metric,
        max_patience = data_args.max_patience,
        learning_rate_classifier = data_args.learning_rate_classifier,
        is_regression = is_regression,
        use_attention = model_args.use_attention,
        use_predictor = model_args.use_predictor,
        bias_type = bias_type,
        CL_type = model_args.cl_type,
        weight_cl = data_args.weight_cl,
        temp = data_args.temp,
        layer = model_args.layer,
        finetune_layer = finetune_cls,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate_on_dev(eval_dataset=eval_dataset,is_regression=is_regression)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
