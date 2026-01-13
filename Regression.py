from __future__ import absolute_import, division, print_function
import random
import pandas as pd
import torch
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from finbert.utils import *
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Config(object):
    """The configuration class for training (regression task)."""
    def __init__(self,
                 data_dir,
                 bert_model,
                 model_dir,
                 learning_rate,
                 max_seq_length=64,
                 train_batch_size=32,
                 eval_batch_size=32,
                 num_train_epochs=3.0,
                 warm_up_proportion=0.1,
                 no_cuda=False,
                 do_lower_case=True,
                 seed=42,
                 local_rank=-1,
                 gradient_accumulation_steps=1,
                 fp16=False,
                 output_mode='regression',
                 discriminate=True,
                 gradual_unfreeze=True,
                 encoder_no=12,
                 base_model='yiyanghkust/finbert-pretrain'):  # 公开FinBERT预训练模型
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

class FinBert(object):
    """Main class for PLTA-FinBERT (regression task)."""
    def __init__(self, config):
        self.config = config

    def prepare_model(self):
        self.processors = {"finsent": FinSentProcessor}

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, do_lower_case=self.config.do_lower_case)

    def get_data(self, phase):
        """Load data for regression task (no label_list required)."""
        self.num_train_optimization_steps = None
        examples = self.processor.get_examples(self.config.data_dir, phase)
        self.num_train_optimization_steps = int(
            len(examples) / self.config.train_batch_size / self.config.gradient_accumulation_steps) * self.config.num_train_epochs
        return examples

    def create_the_model(self):
        """Create model and optimizer for regression."""
        model = self.config.bert_model.to(self.device)
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

        self.num_warmup_steps = int(float(self.num_train_optimization_steps) * self.config.warm_up_proportion)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, correct_bias=False)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_train_optimization_steps
        )
        return model

    def get_loader(self, examples, phase):
        """Create data loader for regression task."""
        try:
            features = convert_examples_to_features(
                examples, None, self.config.max_seq_length, self.tokenizer, self.config.output_mode
            )

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float32)

            if hasattr(features[0], 'agree'):
                all_agree_ids = torch.tensor([f.agree for f in features], dtype=torch.long)
                data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_agree_ids)
            else:
                data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

            sampler = RandomSampler(data) if phase == 'train' else SequentialSampler(data)
            batch_size = self.config.train_batch_size if phase == 'train' else self.config.eval_batch_size
            return DataLoader(data, sampler=sampler, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Failed to create data loader: {str(e)}")
            raise

    def add_noise(self, input_ids, attention_mask, noise_prob=0.02):
        """Add low-probability random token replacement (conventional NLP data augmentation)."""
        noise_mask = (torch.rand_like(input_ids, dtype=torch.float) < noise_prob).to(input_ids.device)
        noise_mask = noise_mask & (attention_mask == 1)  # Only perturb non-padding tokens
        random_ids = torch.randint(100, self.tokenizer.vocab_size, size=input_ids.shape, dtype=torch.long).to(input_ids.device)
        noisy_input_ids = torch.where(noise_mask, random_ids, input_ids)
        return noisy_input_ids

    def get_pseudo_label(self, model, input_ids, attention_mask, token_type_ids, num_predictions=10, noise_prob=0.02):
        """Generate pseudo-label via mean of multi-perturbation predictions (regression)."""
        predictions = []
        model.eval()
        with torch.no_grad():
            for _ in range(num_predictions):
                noisy_input_ids = self.add_noise(input_ids, attention_mask, noise_prob)
                outputs = model(noisy_input_ids, attention_mask, token_type_ids)
                pred = outputs[0].squeeze().item()
                predictions.append(pred)
        
        pseudo_label = np.mean(predictions)
        std = np.std(predictions)
        return pseudo_label, std

    def evaluate(self, model, examples, tokenizer=None):
        """Evaluate regression metrics (MSE, MAE, R²)."""
        model.eval()
        tokenizer = tokenizer or self.tokenizer
        all_preds = []
        all_labels = []
        loss_fn = MSELoss()
        total_loss = 0.0

        features = convert_examples_to_features(examples, None, self.config.max_seq_length, tokenizer, 'regression')
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(self.device)
        all_labels_tensor = torch.tensor([f.label_id for f in features], dtype=torch.float).to(self.device)

        with torch.no_grad():
            outputs = model(all_input_ids, all_attention_mask, all_token_type_ids)
            predictions = outputs[0].squeeze().cpu().numpy()
            labels = all_labels_tensor.cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(labels)

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        loss = loss_fn(torch.tensor(all_preds), torch.tensor(all_labels))
        total_loss += loss.item()

        return mse, mae, r2, total_loss

    def save_training_progress(self, mses, maes, r2s, losses):
        """Save regression training curves and metrics."""
        plot_dir = os.path.join(self.config.model_dir, 'training_plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Smooth curves for visualization
        def smooth(scalars, weight=0.6):
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return np.array(smoothed)

        smooth_mse = smooth(mses)
        smooth_mae = smooth(maes)
        smooth_r2 = smooth(r2s)
        smooth_loss = smooth(losses)

        # Plot curves
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(smooth_mse, color='blue', label='MSE')
        plt.plot(smooth_mae, color='green', label='MAE')
        plt.title('MSE and MAE Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(smooth_r2, color='purple', label='R² Score')
        plt.title('R² Score Curve')
        plt.xlabel('Iteration')
        plt.ylabel('R² Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(smooth_loss, color='orange', label='Loss')
        plt.title('Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save metrics
        np.save(os.path.join(plot_dir, 'mses.npy'), np.array(mses))
        np.save(os.path.join(plot_dir, 'maes.npy'), np.array(maes))
        np.save(os.path.join(plot_dir, 'r2s.npy'), np.array(r2s))
        np.save(os.path.join(plot_dir, 'losses.npy'), np.array(losses))

        # Save key metrics to text file
        min_mse_idx = np.argmin(mses)
        max_r2_idx = np.argmax(r2s)
        with open(os.path.join(self.config.model_dir, 'performance_metrics.txt'), 'w') as f:
            f.write(f"Min MSE: {mses[min_mse_idx]:.4f}\n")
            f.write(f"MAE at Min MSE: {maes[min_mse_idx]:.4f}\n")
            f.write(f"R² at Min MSE: {r2s[min_mse_idx]:.4f}\n")
            f.write(f"Max R²: {max(r2s):.4f}\n")
            f.write(f"MSE at Max R²: {mses[max_r2_idx]:.4f}\n")
            f.write(f"AGF: {sum([1 for mse in mses if mse < mses[0]]) / len(mses):.4f}\n")

    def self_supervised_learning(self, test_examples, tokenizer, noise_level=0.02, num_predictions=10, std_threshold=0.1):
        """Test-time self-supervised learning for regression task."""
        model = self.config.bert_model.to(self.device)
        self.mses = []
        self.maes = []
        self.r2s = []
        self.losses = []

        # Optimizer setup
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        lr = self.config.learning_rate
        dft_rate = 1.2

        if self.config.discriminate:
            encoder_params = []
            for i in range(12):
                encoder_params += [
                    {'params': [p for n, p in model.bert.encoder.layer[i].named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': 0.01, 'lr': lr / (dft_rate ** (12 - i))},
                    {'params': [p for n, p in model.bert.encoder.layer[i].named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0, 'lr': lr / (dft_rate ** (12 - i))}
                ]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01, 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in model.bert.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr / (dft_rate ** 13)},
                {'params': [p for n, p in model.bert.pooler.named_parameters()],
                 'weight_decay': 0.01, 'lr': lr},
                {'params': [p for n, p in model.classifier.named_parameters()],
                 'weight_decay': 0.01, 'lr': lr}
            ] + encoder_params
        else:
            param_optimizer = list(model.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(test_examples))
        loss_fn = MSELoss()
        examples_for_eval = test_examples.copy()

        # Initial evaluation
        initial_mse, initial_mae, initial_r2, initial_loss = self.evaluate(model, examples_for_eval, tokenizer)
        self.mses.append(initial_mse)
        self.maes.append(initial_mae)
        self.r2s.append(initial_r2)
        self.losses.append(initial_loss)
        logger.info(f"Initial evaluation - MSE: {initial_mse:.4f}, MAE: {initial_mae:.4f}, R²: {initial_r2:.4f}")

        for i, example in enumerate(tqdm(test_examples, desc="Test-time adaptation")):
            # Convert example to model input
            features = convert_examples_to_features([example], None, self.config.max_seq_length, tokenizer, 'regression')
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device)
            token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(self.device)

            # Generate pseudo-label (filter by std threshold)
            pseudo_label, std = self.get_pseudo_label(model, input_ids, attention_mask, token_type_ids, num_predictions, noise_level)
            if std > std_threshold:
                # Skip unstable pseudo-labels
                self.mses.append(self.mses[-1])
                self.maes.append(self.maes[-1])
                self.r2s.append(self.r2s[-1])
                self.losses.append(self.losses[-1])
                logger.info(f"Sample {i+1}/{len(test_examples)} skipped - Std: {std:.4f} > Threshold: {std_threshold}")
                continue

            # Update model
            true_label = features[0].label_id
            model.train()
            outputs = model(input_ids, attention_mask, token_type_ids)[0].squeeze()
            loss = loss_fn(outputs, torch.tensor(pseudo_label, dtype=torch.float32).to(self.device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Evaluate after update
            eval_set = examples_for_eval[:i] + examples_for_eval[i+1:] if i < len(examples_for_eval)-1 else examples_for_eval[:i]
            current_mse, current_mae, current_r2, current_loss = self.evaluate(model, eval_set, tokenizer)
            self.mses.append(current_mse)
            self.maes.append(current_mae)
            self.r2s.append(current_r2)
            self.losses.append(current_loss)

            logger.info(f"Sample {i+1}/{len(test_examples)}: True={true_label:.4f}, Pseudo={pseudo_label:.4f}, Std={std:.4f}, MSE={current_mse:.4f}, R²={current_r2:.4f}")

            # Save progress every 100 samples
            if (i + 1) % 100 == 0 or (i + 1) == len(test_examples):
                self.save_training_progress(self.mses, self.maes, self.r2s, self.losses)

        # Save final model
        torch.save(model.state_dict(), os.path.join(self.config.model_dir, "plta-finbert-regression.pt"))
        logger.info("Training completed! Model saved to {}".format(os.path.join(self.config.model_dir, "plta-finbert-regression.pt")))
        return self.mses, self.maes, self.r2s, self.losses

def predict(text, model, tokenizer, write_to_csv=False, path=None, use_gpu=False, max_length=64):
    """Predict sentiment score (regression mode)."""
    model.eval()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        sentences = sent_tokenize(text) if isinstance(text, str) else []
        if not sentences:
            logger.warning("No valid sentences to process")
            return pd.DataFrame(columns=['sentence', 'prediction'])

        input_ids = []
        attention_masks = []
        token_type_ids = []

        for sentence in sentences:
            encoded_dict = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_type_ids.append(encoded_dict['token_type_ids'] if 'token_type_ids' in encoded_dict else torch.tensor([[0]*max_length]))

        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_mask = torch.cat(attention_masks, dim=0).to(device)
        token_type_ids = torch.cat(token_type_ids, dim=0).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            if isinstance(outputs, torch.Tensor):
                predictions = outputs.squeeze().cpu().numpy()
            elif hasattr(outputs, 'logits'):
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs[0].squeeze().cpu().numpy()

            if isinstance(predictions, np.ndarray) and predictions.ndim == 0:
                predictions = np.array([predictions.item()])
            elif not isinstance(predictions, np.ndarray):
                predictions = np.array([predictions])

            result = pd.DataFrame({
                'sentence': sentences[:len(predictions)],
                'prediction': predictions
            })

        if write_to_csv and path:
            result.to_csv(path, index=False)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return pd.DataFrame(columns=['sentence', 'prediction'])

# ------------------------------
# Run regression task
# ------------------------------
if __name__ == "__main__":
    # Configuration
    data_dir = "./data/regression"  # Path to train.csv/test.csv
    model_dir = "./saved_model/regression"  # Path to save model
    bert_model = BertForSequenceClassification.from_pretrained(
        'yiyanghkust/finbert-pretrain', num_labels=1  # Regression task (1 output)
    )

    config = Config(
        data_dir=data_dir,
        bert_model=bert_model,
        model_dir=model_dir,
        learning_rate=48e-7,  # Regression task LR (from paper Table 3)
        max_seq_length=64,
        train_batch_size=16,
        eval_batch_size=16,
        num_train_epochs=3.0,
        output_mode='regression',
        base_model='yiyanghkust/finbert-pretrain'
    )

    # Initialize model
    finbert = FinBert(config)
    finbert.prepare_model()

    # Load test data (test.csv in data_dir)
    test_examples = finbert.get_data('test')
    logger.info(f"Loaded {len(test_examples)} test samples")

    # Run test-time self-supervised learning
    mses, maes, r2s, losses = finbert.self_supervised_learning(
        test_examples=test_examples,
        tokenizer=finbert.tokenizer,
        noise_level=0.02,  # Noise probability (from paper Table 3)
        num_predictions=10,  # Number of perturbations per sample
        std_threshold=0.1  # Regression std threshold (from paper Table 3)
    )

    # Save final results
    finbert.save_training_progress(mses, maes, r2s, losses)