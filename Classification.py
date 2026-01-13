from __future__ import absolute_import, division, print_function
import random
import pandas as pd
import torch
from torch.nn import MSELoss, CrossEntropyLoss, Dropout
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import trange
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from finbert.utils import *
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Config(object):
    """The configuration class for training (classification task)."""
    def __init__(self,
                 data_dir,
                 bert_model,
                 model_dir,
                 max_seq_length=64,
                 train_batch_size=32,
                 eval_batch_size=32,
                 learning_rate=2e-5,
                 num_train_epochs=2,
                 warm_up_proportion=0.1,
                 no_cuda=False,
                 do_lower_case=True,
                 seed=42,
                 local_rank=-1,
                 gradient_accumulation_steps=1,
                 fp16=False,
                 output_mode='classification',
                 discriminate=True,
                 gradual_unfreeze=True,
                 encoder_no=12,
                 base_model='yiyanghkust/finbert-pretrain',  # 公开FinBERT预训练模型
                 dropout_prob=0.1):
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.local_rank = local_rank
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warm_up_proportion = warm_up_proportion
        self.no_cuda = no_cuda
        self.seed = seed
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_mode = output_mode
        self.fp16 = fp16
        self.discriminate = discriminate
        self.gradual_unfreeze = gradual_unfreeze
        self.encoder_no = encoder_no
        self.base_model = base_model
        self.dropout_prob = dropout_prob

class FinBert(object):
    """Main class for PLTA-FinBERT (classification task)."""
    def __init__(self, config):
        self.config = config

    def prepare_model(self, label_list):
        self.processors = {"finsent": FinSentProcessor}
        self.num_labels_task = {'finsent': 2}

        # Device configuration
        if self.config.local_rank == -1 or self.config.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.config.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.n_gpu = 1
            torch.distributed.init_process_group(backend='nccl')
        
        logger.info(f"device: {self.device}, n_gpu: {self.n_gpu}, distributed training: {bool(self.config.local_rank != -1)}, 16-bits training: {self.config.fp16}")

        if self.config.gradient_accumulation_steps < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps: {self.config.gradient_accumulation_steps}, should be >= 1")
        
        self.config.train_batch_size = self.config.train_batch_size // self.config.gradient_accumulation_steps

        # Seed initialization
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.config.seed)

        # Model directory setup
        if os.path.exists(self.config.model_dir) and os.listdir(self.config.model_dir):
            raise ValueError(f"Output directory {self.config.model_dir} already exists and is not empty.")
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

        self.processor = self.processors['finsent']()
        self.num_labels = len(label_list)
        self.label_list = label_list
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, do_lower_case=self.config.do_lower_case)
        self.config.bert_model.dropout = Dropout(self.config.dropout_prob)

    def get_data(self, phase):
        self.num_train_optimization_steps = None
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps) * self.config.num_train_epochs
        
        if phase == 'train':
            train = pd.read_csv(os.path.join(self.config.data_dir, 'train.csv'), sep=',', index_col=False)
            class_weights = [train.shape[0] / train[train.label == label].shape[0] for label in self.label_list]
            self.class_weights = torch.tensor(class_weights).to(self.device)
        
        return examples

    def create_the_model(self):
        model = self.config.bert_model.to(self.device)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        lr = self.config.learning_rate
        dft_rate = 1.2

        if self.config.discriminate:
            encoder_params = []
            for i in range(12):
                encoder_decay = {
                    'params': [p for n, p in model.bert.encoder.layer[i].named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr / (dft_rate ** (12 - i))
                }
                encoder_nodecay = {
                    'params': [p for n, p in model.bert.encoder.layer[i].named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr / (dft_rate ** (12 - i))
                }
                encoder_params.append(encoder_decay)
                encoder_params.append(encoder_nodecay)
            
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in model.bert.pooler.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr},
                {'params': [p for n, p in model.bert.pooler.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr},
                {'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr}
            ] + encoder_params
        else:
            param_optimizer = list(model.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.config.warm_up_proportion)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, correct_bias=False)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_train_optimization_steps
        )
        return model

    def get_loader(self, examples, phase):
        features = convert_examples_to_features(examples, self.label_list,
                                                self.config.max_seq_length,
                                                self.tokenizer,
                                                self.config.output_mode)
        
        logger.info(f"***** Loading {phase} data *****")
        logger.info(f"  Num examples = {len(examples)}")
        logger.info(f"  Batch size = {self.config.train_batch_size}")
        logger.info(f"  Num steps = {self.num_train_optimization_steps}")

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        
        if self.config.output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        elif self.config.output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        
        try:
            all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
        except:
            all_agree_ids = torch.tensor([0 for _ in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_agree_ids)
        sampler = RandomSampler(data) if phase == 'train' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.config.train_batch_size)
        return dataloader

    def add_noise(self, input_ids, attention_mask):
        """Add low-probability random token replacement (conventional NLP data augmentation)."""
        noise_mask = (torch.rand_like(input_ids, dtype=torch.float) < 0.02).to(input_ids.device)
        noise_mask = noise_mask & (attention_mask == 1)  # Only perturb non-padding tokens
        random_ids = torch.randint(100, self.tokenizer.vocab_size, size=input_ids.shape, dtype=torch.long).to(input_ids.device)
        noisy_input_ids = torch.where(noise_mask, random_ids, input_ids)
        return noisy_input_ids

    def random_delete(self, input_ids, attention_mask):
        """Random token deletion (prob=0.02)."""
        delete_mask = (torch.rand_like(input_ids, dtype=torch.float) < 0.02).to(input_ids.device)
        delete_mask = delete_mask & (attention_mask == 1)
        input_ids = torch.where(delete_mask, torch.tensor(0).to(input_ids.device), input_ids)
        return input_ids

    def random_insert(self, input_ids, attention_mask):
        """Random token insertion (prob=0.02)."""
        insert_mask = (torch.rand_like(input_ids, dtype=torch.float) < 0.02).to(input_ids.device)
        insert_mask = insert_mask & (attention_mask == 1)
        random_ids = torch.randint(100, self.tokenizer.vocab_size, size=input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids = torch.where(insert_mask, random_ids, input_ids)
        return input_ids

    def get_pseudo_label(self, model, input_ids, attention_mask, token_type_ids, num_predictions=10):
        """Generate pseudo-label via majority voting of multi-perturbation predictions."""
        predictions = []
        for _ in range(num_predictions):
            noisy_input_ids = self.add_noise(input_ids, attention_mask)
            noisy_input_ids = self.random_delete(noisy_input_ids, attention_mask)
            noisy_input_ids = self.random_insert(noisy_input_ids, attention_mask)
            with torch.no_grad():
                logits = model(noisy_input_ids, attention_mask, token_type_ids)[0]
                pred = torch.argmax(logits, dim=1).item()
                predictions.append(pred)
        
        pseudo_label = max(set(predictions), key=predictions.count)
        confidence = predictions.count(pseudo_label) / num_predictions
        return pseudo_label, confidence

    def entropy_regularization(self, logits):
        """Entropy regularization to mitigate pseudo-label noise."""
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return torch.mean(entropy)

    def self_supervised_learning(self, test_examples, label_list, tokenizer, num_predictions=10):
        """Test-time self-supervised learning with pseudo-labels."""
        model = self.config.bert_model
        model.to(self.device)

        # Optimizer setup (discriminative fine-tuning)
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        lr = self.config.learning_rate
        dft_rate = 1.2

        if self.config.discriminate:
            encoder_params = []
            for i in range(12):
                encoder_decay = {
                    'params': [p for n, p in model.bert.encoder.layer[i].named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01, 'lr': lr / (dft_rate ** (12 - i))
                }
                encoder_nodecay = {
                    'params': [p for n, p in model.bert.encoder.layer[i].named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0, 'lr': lr / (dft_rate ** (12 - i))
                }
                encoder_params.append(encoder_decay)
                encoder_params.append(encoder_nodecay)
            
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in model.bert.pooler.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr},
                {'params': [p for n, p in model.bert.pooler.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr},
                {'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr}
            ] + encoder_params
        else:
            param_optimizer = list(model.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(test_examples))
        loss_fn_ce = CrossEntropyLoss()
        loss_fn_entropy = self.entropy_regularization

        accuracies = []
        losses = []
        pseudo_label_qualities = []

        for i, example in enumerate(tqdm(test_examples, desc="Test-time adaptation")):
            # Convert example to model input
            features = convert_examples_to_features([example], label_list, self.config.max_seq_length, tokenizer)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(self.device)

            # Generate pseudo-label (confidence threshold = 0.8)
            model.eval()
            pseudo_label, confidence = self.get_pseudo_label(model, all_input_ids, all_attention_mask, all_token_type_ids, num_predictions)
            if confidence < 0.8:
                # Skip low-confidence pseudo-labels
                accuracies.append(accuracies[-1] if accuracies else 0.0)
                losses.append(losses[-1] if losses else 0.0)
                pseudo_label_qualities.append(0)
                continue

            # Calculate loss and update model
            true_label = features[0].label_id
            model.train()
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            loss_ce = loss_fn_ce(logits, torch.tensor([pseudo_label]).to(self.device))
            loss_entropy = loss_fn_entropy(logits)
            loss = loss_ce + 0.05 * loss_entropy  # Entropy weight = 0.05

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate after update
            model.eval()
            accuracy, loss_test = self.evaluate(model, test_examples, label_list, tokenizer)
            accuracies.append(accuracy)
            losses.append(loss_test)

            # Track pseudo-label quality
            pseudo_label_quality = 1 if pseudo_label == true_label else 0
            pseudo_label_qualities.append(pseudo_label_quality)

            logger.info(f"Sample {i+1}/{len(test_examples)}: True Label = {true_label}, Pseudo Label = {pseudo_label}, Confidence = {confidence:.2f}, Accuracy = {accuracy:.4f}")

            # Save progress every 100 samples
            if (i + 1) % 100 == 0 or (i + 1) == len(test_examples):
                self.save_training_progress(accuracies, losses, pseudo_label_qualities)

        return accuracies, losses, pseudo_label_qualities

    def evaluate(self, model, examples, label_list, tokenizer):
        """Evaluate classification accuracy and loss."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        loss_fn = CrossEntropyLoss()

        features = convert_examples_to_features(examples, label_list, self.config.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(self.device)
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            loss = loss_fn(logits, all_labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == all_labels).sum().item()
            total += all_labels.size(0)

        accuracy = correct / total
        return accuracy, total_loss / total

    def save_training_progress(self, accuracies, losses, pseudo_label_qualities):
        """Save training curves and metrics."""
        plot_dir = os.path.join(self.config.model_dir, 'training_plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Plot curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(accuracies, label='Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(losses, label='Loss', color='orange')
        plt.title('Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(pseudo_label_qualities, label='Pseudo-label Quality', color='green')
        plt.title('Pseudo-label Quality Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Correct Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save metrics as numpy arrays
        np.save(os.path.join(plot_dir, 'accuracies.npy'), np.array(accuracies))
        np.save(os.path.join(plot_dir, 'losses.npy'), np.array(losses))
        np.save(os.path.join(plot_dir, 'pseudo_label_qualities.npy'), np.array(pseudo_label_qualities))

def predict(text, model, tokenizer, write_to_csv=False, path=None, use_gpu=False, batch_size=5):
    """Predict sentiment of input text (classification mode)."""
    model.eval()
    sentences = sent_tokenize(text) if isinstance(text, str) else []
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}
    result = pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])

    for batch in chunks(sentences, batch_size):
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]
        features = convert_examples_to_features(examples, label_list, 64, tokenizer)
        
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            logits = softmax(np.array(logits.cpu()))
            sentiment_score = pd.Series(logits[:, 0] - logits[:, 1])
            predictions = np.squeeze(np.argmax(logits, axis=1))

            batch_result = {
                'sentence': batch,
                'logit': list(logits),
                'prediction': predictions,
                'sentiment_score': sentiment_score
            }
            batch_result = pd.DataFrame(batch_result)
            result = pd.concat([result, batch_result], ignore_index=True)

    result['prediction'] = result['prediction'].apply(lambda x: label_dict[x] if isinstance(x, int) else label_dict[x.item()])
    if write_to_csv and path:
        result.to_csv(path, sep=',', index=False)
    return result

# ------------------------------
# Run classification task
# ------------------------------
if __name__ == "__main__":
    # Configuration
    data_dir = "./data/classification"  # Path to train.csv/test.csv
    model_dir = "./saved_model/classification"  # Path to save model
    bert_model = BertForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-pretrain', num_labels=3  # 3 classes: positive/negative/neutral
    )

    config = Config(
        data_dir=data_dir,
        bert_model=bert_model,
        model_dir=model_dir,
        max_seq_length=64,
        train_batch_size=32,
        eval_batch_size=32,
        learning_rate=5e-6,  # Classification task LR (from paper Table 3)
        num_train_epochs=2,
        warm_up_proportion=0.1,
        no_cuda=False,
        do_lower_case=True,
        seed=42,
        output_mode='classification',
        discriminate=True,
        gradual_unfreeze=True,
        encoder_no=12,
        base_model='yiyanghkust/finbert-pretrain',
        dropout_prob=0.1
    )

    # Initialize model
    finbert = FinBert(config)
    label_list = ['positive', 'negative', 'neutral']
    finbert.prepare_model(label_list)

    # Load test data (test.csv in data_dir)
    test_examples = finbert.get_data('test')
    logger.info(f"Loaded {len(test_examples)} test samples")

    # Run test-time self-supervised learning
    accuracies, losses, pseudo_label_qualities = finbert.self_supervised_learning(
        test_examples=test_examples,
        label_list=label_list,
        tokenizer=finbert.tokenizer,
        num_predictions=10  # Number of perturbations per sample (from paper Table 3)
    )

    # Save final results
    finbert.save_training_progress(accuracies, losses, pseudo_label_qualities)
    torch.save(bert_model.state_dict(), os.path.join(model_dir, "plta-finbert-classification.pt"))
    logger.info("Training completed! Model saved to {}".format(os.path.join(model_dir, "plta-finbert-classification.pt")))