import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random
import os
import pandas as pd
import copy
import gc
import torch.utils.checkpoint as cp


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.set_float32_matmul_precision("high")

seq_len = 32

########################################
# Tokenizer
########################################

class RawPairDataset(torch.utils.data.Dataset):
    def __init__(self, query_target_pairs):
            self.pairs = query_target_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        if isinstance(sample, dict):
            return sample['query'], sample['target']
        return sample  # assume it's already a tuple

# Global tokenizer reference
global_tokenizer = None
seq_len_for_collate = seq_len

def init_collate_globals(tokenizer, seq_len):
    global global_tokenizer, seq_len_for_collate
    global_tokenizer = tokenizer
    seq_len_for_collate = seq_len



class TokenizerWrapper:
    def __init__(self, tokenizer, seq_len=seq_len, add_bos=True, add_eos=True, pad_to_max=True, shift_decoder=False, device="cuda"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_to_max = pad_to_max
        self.shift_decoder = shift_decoder
        self.device = device

        self.bos_token = tokenizer.bos_token or "<BOS>"
        self.eos_token = tokenizer.eos_token or "<EOS>"
        self.pad_token_id = tokenizer.pad_token_id or 0

    def format(self, text):
        if isinstance(text, list):
            return [self.format(t) for t in text]
        return f"{self.bos_token} {text} {self.eos_token}" if self.add_bos and self.add_eos else text

    def encode(self, text_batch, truncate=True):
        if isinstance(text_batch[0], str):
            text_batch = self.format(text_batch)

        encoded = [self.tokenizer.encode(t, add_special_tokens=False) for t in text_batch]
        result = []
        for tokens in encoded:
            if truncate and len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len - 1] + [self.tokenizer.eos_token_id]
            result.append(tokens)
        return result if not self.pad_to_max else torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, device=self.device) for seq in result],
            batch_first=True,
            padding_value=self.pad_token_id
        )

    def encode_shifted_pair(self, text_batch):
        """Returns (decoder_input_ids, labels), both padded"""
        full = self.encode(text_batch)  # [B, T]
        decoder_input = full[:, :-1]
        labels = full[:, 1:]
        return decoder_input, labels



########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################

from transformers import PreTrainedTokenizerFast

# 🔹 Change this to the actual path where your BPE tokenizer files are stored
tokenizer_path = r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest-main"  

# 🔹 Load a BPE tokenizer from local files
base_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

print(f"✅ Loaded custom BPE tokenizer from: {tokenizer_path}")
print(f"📏 Vocabulary size: {base_tokenizer.vocab_size}")

# Wrap it with the hierarchical tokenizer
tokenizer = base_tokenizer


########################################
# 2. Data Extraction
########################################

def extract_data(json_data):
    """Extracts training data from JSON file and tokenizes it."""
    input_ids_list = []
    target_ids_list = []

    for item in json_data:
        conversations = item.get("conversations", [])

        if not isinstance(conversations, list) or len(conversations) < 2:
            print(f"⚠️ Skipping entry with no valid conversation: {item}")
            continue

        for i in range(len(conversations) - 1):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            # Ensure we only process valid user-assistant exchanges
            if user_turn.get("from") in ["user", "human"] and assistant_turn.get("from") in ["assistant", "gpt"]:
                query = user_turn.get("value", "").strip()
                target = assistant_turn.get("value", "").strip()

                # 🔹 Ensure valid text exists before tokenizing
                if not query or not target:
                    print(f"⚠️ Skipping empty user/assistant exchange: {user_turn} -> {assistant_turn}")
                    continue  

                input_ids = tokenizer.tokenize(query)
                target_ids = tokenizer.tokenize(target)

                # 🔹 Ensure tokenized output isn't empty
                if not input_ids or not target_ids:
                    print(f"⚠️ Skipping invalid tokenized entry: {query} -> {input_ids}")
                    continue

                input_ids_list.append(input_ids)
                target_ids_list.append(target_ids)
    

    return list(zip(input_ids_list, target_ids_list))  # Ensure format is (input, target)

def load_dataset(dataset_path):

            dataset_files = os.listdir(dataset_path)
            query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(dataset_path, file)
                if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                            else:
                                data = json.load(f)
                                query_target_pairs.extend(extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]

                elif file.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                else:
                    print("errpr")
            if not query_target_pairs:
                print("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            text_data = []
            for query, target in query_target_pairs:
                text_data.append(f"User: {query}\nAssistant: {target}")

            logging.info(f"Loaded dataset with {len(query_target_pairs)} query/target pairs.")
            return query_target_pairs


def extract_query_target_pairs( data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

        return query_target_pairs

def tokenize_data(query_target_pairs):

        # Select training mode
        input_ids_list = []  # Initialize for unchunked dataset
        labels_list = []  # Initialize for unchunked dataset

        for query, target in query_target_pairs:
                        input_ids, labels = _generate_training_pairs(query, target)

                        if input_ids and labels:
                            input_ids_list.append(input_ids)  # Store for training
                            labels_list.append(labels)  # Store for training
                            #print (input_ids)
                            #print(labels)
        return input_ids_list, labels_list


def _generate_training_pairs(query, target):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)

        input_ids = [tokenizer.bos_token_id] + query_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

        return input_ids, labels

def prepare_batch(input_ids, labels, seq_len):
                pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths

                #input_ids = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in input_ids]
                #labels = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in labels]

                # ✅ Compute correct padding lengths
                #input_ids = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in input_ids]
                #labels = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in labels]
                
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in input_ids
                ]
                logging.info("input ids torched to tensor")
                print(input_ids)
                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels
                ]
                logging.info("labels torched to tensor")
                print(labels)
                # Stack tensors
                input_ids = torch.stack(input_ids).to(device)
                labels = torch.stack(labels).to(device)
                data = torch.utils.data.TensorDataset(input_ids, labels)
                return data


########################################
# 3. Dataset and Collate Function
########################################

def collate_fn(batch):
    global global_tokenizer, seq_len_for_collate

    BOS = global_tokenizer.bos_token or "<BOS>"
    EOS = global_tokenizer.eos_token or "<EOS>"
    PAD_ID = global_tokenizer.pad_token_id or 0  # Fallback if pad_token not set

    def encode_and_fix(texts):
        fixed_seqs = []
        for t in texts:
            tokens = global_tokenizer.encode(BOS + " " + t + " " + EOS, add_special_tokens=False)
            if len(tokens) > seq_len_for_collate:
                tokens = tokens[:seq_len_for_collate - 1] + [global_tokenizer.eos_token_id]  # truncate and force EOS
            padded = tokens + [PAD_ID] * (seq_len_for_collate - len(tokens))
            fixed_seqs.append(padded)
        return torch.tensor(fixed_seqs, dtype=torch.long)

    if isinstance(batch[0], str):
        input_ids = encode_and_fix(batch)
        return input_ids, input_ids

    elif isinstance(batch[0], tuple):
        queries, targets = zip(*batch)
        input_ids = encode_and_fix(queries)
        target_ids = encode_and_fix(targets)
        return input_ids, target_ids


##############################################
# Positional Encoding (Standard Sin/Cos Version)
##############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=seq_len, device=device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        
        self.pe = torch.zeros(max_len, d_model)
        self.position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)
        self.pe = self.pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        #self.register_buffer('pe', self.pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        # x: (batch, seq_len, d_model)
        x = x.to(self.device) + self.pe[:, :seq_len].to(self.device)
        return self.dropout(x)

########################################
#Base Model
########################################

class GeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=10):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = None
        best_loss = float('inf')
        n=0
        loss = 0
        if architecture == "Reasoning Model LNS":

            output = self.model(inputs, decoder_input)

        else:
            output = self.model(inputs, target_labels)          
                
        output = output.reshape(-1, output.shape[-1])
        logging.debug(f"output reshaped Shape: {output.shape}")
        target_labels_reshaped = target_labels.reshape(-1)
        logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
        loss = loss_fn(output, target_labels_reshaped)
        best_loss = loss
        print(f"Original model iteration {n}, Loss: {loss.item()}")
        best_model = self.model
        for model in self.population:
            loss = 0
            if architecture == "Reasoning Model LNS":

                output = model(inputs, decoder_input)

            else:
                output = model(inputs, target_labels)          
                
            output = output.reshape(-1, output.shape[-1])
            logging.debug(f"output reshaped Shape: {output.shape}")
            target_labels_reshaped = target_labels.reshape(-1)
            logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
            loss = loss_fn(output, target_labels_reshaped)
            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss.item()}")
                    best_model = model
            
            else:
                loss = 0

                if architecture == "Reasoning Model LNS":

                    output = model(inputs, decoder_input)

                else:
                    output = model(inputs, target_labels)
                # Flatten logits and targets:
                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                target_labels_reshaped = target_labels.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
                loss = loss_fn(output, target_labels_reshaped)
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss.item()}")
                        best_model = model
        return best_model

    def evolve(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        self.model = self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
        self.population = [copy.deepcopy(self.model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]

        return self.dropout(x + pe)

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, sinusoidal_emb):
    return (x * sinusoidal_emb.cos()) + (rotate_half(x) * sinusoidal_emb.sin())

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)[None, :, :]  # [1, seq_len, dim]
        return apply_rotary(x, emb)

    
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, device, tokenizer=base_tokenizer):
        super().__init__()
        self.embed_size = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = RotaryPositionalEmbedding(embedding_dim)
        #self.pos_encoder = DynamicPositionalEncoding(embedding_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)
        self.decoder = nn.TransformerDecoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.decoder_layers = nn.TransformerDecoder(decoder_layer=self.decoder, num_layers=num_layers)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=seq_length, shift_decoder=False, device=device)
        self.tokenizer = tokenizer
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def generate_mask(self, src, tgt):
        # Padding mask: (batch_size, seq_len) with True for padding tokens
        src_pad_mask = (src == 0)  # Shape: [batch, src_len]
        tgt_pad_mask = (tgt == 0)  # Shape: [batch, tgt_len]

        # Causal mask for decoder (no peeking into the future)
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)  # Shape: [tgt_len, tgt_len]

        return src_pad_mask, tgt_pad_mask, causal_mask

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def encode_src(self, src):
        src_pad_mask = (src == self.tokenizer.pad_token_id)
        src_emb = self.token_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        return self.encoder_layers(src_emb, src_key_padding_mask=src_pad_mask)

    def decode_tgt(self, tgt_ids, memory):
        if tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        tgt_pad_mask = (tgt_ids == self.tokenizer.pad_token_id)
        causal_mask = self.generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)

        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)

        def layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory,
                tgt_mask=inputs[1],
                tgt_key_padding_mask=inputs[2],
                memory_key_padding_mask=None
            )
        output = cp.checkpoint(layer_fn, tgt_emb, causal_mask, tgt_pad_mask)

        return self.fc_out(output)

    def forward(self, src, tgt_ids=None, mode='eval'):

        if isinstance(src[0], str):
            src = self.tokenizer_wrapper.encode(src)
        if tgt_ids is not None and isinstance(tgt_ids[0], str):
            tgt_ids= self.tokenizer_wrapper.encode(tgt_ids)
        elif tgt_ids is not None and mode == 'train':
            tgt_ids = tgt_ids
            #tgt_ids = tgt_ids[:, 1:]
        #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt_ids is not None and tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        src_pad_mask, tgt_pad_mask, causal_mask = self.generate_mask(src, tgt_ids if tgt_ids is not None else src)
        #print(f"📏 src_pad_mask: {src_pad_mask.shape}")
        #print(f"📏 tgt_pad_mask: {tgt_pad_mask.shape}")
        #print(f"📏 causal_mask: {causal_mask.shape}")

        src_emb = self.token_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        def layer_fn(*inputs):
            return self.encoder_layers(
                inputs[0], 
                src_key_padding_mask=inputs[1]
            )
        memory = cp.checkpoint(layer_fn, src_emb, src_pad_mask)
            
        if tgt_ids is None:
            tgt_ids = src[:, :1]  # dummy start

        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        #print(f"💡 Embeddings: src {src_emb.shape}, tgt {tgt_emb.shape}")

        def decoder_layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory,
                tgt_mask=inputs[1],
                tgt_key_padding_mask=inputs[2],
                memory_key_padding_mask=inputs[3]
            )
        output = cp.checkpoint(decoder_layer_fn, tgt_emb, causal_mask, tgt_pad_mask, src_pad_mask)

        return self.fc_out(output)


class SymbolicTokenPredictor(nn.Module):
    def __init__(self, vocab_size, seq_length, tokenizer, embed_dim=128, num_terms=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.scalar_layer = nn.Linear(embed_dim, embed_dim)
        funcs = [
            "sin", "cos", "tan", "csc", "sec", "cot",
            "arcsin", "arccos", "arctan",
            "exp", "log",
            "poly1", "poly2", "poly3"
            ]
        self.terms = nn.ModuleList([SymbolicTerm(func_type=random.choice(funcs)) for _ in range(num_terms)])
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=seq_length, shift_decoder=False, device=device)
        self.tokenizer = tokenizer

    def forward(self, input_ids, target):
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids.to(device))       # [B, T, D]
        x = self.scalar_layer(x)            # [B, T, D]

        out = []
        for t in range(x.size(1)):          # autoregressive loop
            xt = x[:, t]                    # [B, D]
            xt_transformed = xt
            for term in self.terms:
                tarm, _ = term(xt_transformed.to(device), target.to(device))
                xt_transformed = xt_transformed + tarm
            out.append(xt_transformed)  # [B, V]

        logits = torch.stack(out, dim=1)    # [B, T, V]

        return logits

# Simulate a single Term in the additive series
class Term(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(1))
        self.weight = nn.Parameter(torch.randn(1))
        self.route = nn.Parameter(torch.randn(3))  # forward, backward, sideways

    def forward(self, target, neighbor=None):
        route_weights = F.softmax(self.route, dim=0)  # ensure they sum to 1
        fwd = route_weights[0] * self.weight * self.value
        bwd = -route_weights[1] * self.weight * self.value
        side = route_weights[2] * neighbor.weight * neighbor.value if neighbor else 0.0
        return fwd + bwd + side

class SymbolicTerm(nn.Module):
    def __init__(self, func_type="sin"):
        super().__init__()
        self.func_type = func_type
        self.a = nn.Parameter(torch.randn(1))  # scale
        self.b = nn.Parameter(torch.randn(1))  # freq/multiplier
        self.c = nn.Parameter(torch.randn(1))  # phase/offset
        self.weight = nn.Parameter(torch.randn(1))
        self.route = nn.Parameter(torch.randn(3))  # fwd/bwd/side
        self.goal_gen = GoalGenerator()

    def safe_input(self, x):
        return torch.clamp(x, 1e-6, 1e6)  # for log, tan, etc.

    def compute(self, x):
        x = self.safe_input(x)
        t = self.func_type
        if t == "sin":
            return self.a * torch.sin(self.b * x + self.c)
        elif t == "cos":
            return self.a * torch.cos(self.b * x + self.c)
        elif t == "tan":
            return self.a * torch.tan(self.b * x + self.c)
        elif t == "csc":
            return self.a / torch.sin(self.b * x + self.c)
        elif t == "sec":
            return self.a / torch.cos(self.b * x + self.c)
        elif t == "cot":
            return self.a / torch.tan(self.b * x + self.c)
        elif t == "arcsin":
            return self.a * torch.arcsin(torch.clamp(self.b * x + self.c, -1 + 1e-3, 1 - 1e-3))
        elif t == "arccos":
            return self.a * torch.arccos(torch.clamp(self.b * x + self.c, -1 + 1e-3, 1 - 1e-3))
        elif t == "arctan":
            return self.a * torch.arctan(self.b * x + self.c)
        elif t == "exp":
            return self.a * torch.exp(torch.clamp(self.b * x + self.c, max=10))
        elif t == "log":
            return self.a * torch.log(torch.clamp(self.b * x + self.c, min=1e-3))
        elif t == "poly1":
            return self.a * x + self.b
        elif t == "poly2":
            return self.a * x**2 + self.b * x + self.c
        elif t == "poly3":
            return self.a * x**3 + self.b * x**2 + self.c
        else:
            return self.a * x  # fallback

    def forward(self, x, target, neighbor=None):
        value = self.compute(x)
        target = self.goal_gen.update_goal(target)

        route_weights = F.softmax(self.route, dim=0)
        fwd = route_weights[0] * self.weight * value
        bwd = -route_weights[1] * self.weight * value
        side = route_weights[2] * neighbor.compute(x) if neighbor else 0.0
        total = fwd + bwd + side

        return total, target


# Controller that generates evolving target values
class GoalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def update_goal(self, step):
        self.phase += 0.1
        return torch.tensor([[torch.sin(torch.tensor(self.phase))]])  # shape: [1, 1]

# Series that aggregates active terms
class ScalarLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = nn.Linear(1, 1)

    def forward(self, x):
        return self.scalar(x)

def analyze_model(model, step):
    func_counts = {}
    coef_stats = {"a": [], "b": [], "c": []}
    routing_entropy = []

    for term in model.terms:
        func = term.func_type
        func_counts[func] = func_counts.get(func, 0) + 1

        coef_stats["a"].append(term.a.item())
        coef_stats["b"].append(term.b.item())
        coef_stats["c"].append(term.c.item())

        route_probs = F.softmax(term.route, dim=0)
        entropy = -(route_probs * torch.log(route_probs + 1e-8)).sum().item()
        routing_entropy.append(entropy)

    print(f"\n[Diagnostics at step {step}]")
    print("Function distribution:", func_counts)
    print("Mean a/b/c:", {k: round(sum(v) / len(v), 4) for k, v in coef_stats.items()})
    print("Avg routing entropy:", round(sum(routing_entropy) / len(routing_entropy), 4))

class SeriesSimulator(nn.Module):
    def __init__(self, num_terms):
        super().__init__()
        self.scalar_layer = ScalarLayer()
        funcs = [
            "sin", "cos", "tan", "csc", "sec", "cot",
            "arcsin", "arccos", "arctan",
            "exp", "log",
            "poly1", "poly2", "poly3"
        ]

        self.terms = nn.ModuleList([SymbolicTerm(func_type=random.choice(funcs)) for _ in range(num_terms)])
        self.goal_gen = GoalGenerator()

    def forward(self, step):
        x = torch.tensor([[step / 100.0]])
        scalar_out = self.scalar_layer(x)

        # Combine base scalar with symbolic input
        x_symbolic = x + scalar_out

        target = self.goal_gen.update_goal(step)
        total = 0.0
        for i, term in enumerate(self.terms):
            neighbor = self.terms[i + 1] if i + 1 < len(self.terms) else None
            total += term(x_symbolic, neighbor)

        loss = F.smooth_l1_loss(total, target)
        return total, target, loss


    
########################################
# 5. Training Loop
########################################

def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output


def build_custom_validation_batch(tokenizer, seq_len=seq_len, device=device, batch_size=1):
    query_strings = [
        "1. What is 17 + 35?",
        "2. Solve for x: 2x + 5 = 13",
        "3. What is the derivative of x^2?",
        "4. What is the integral of x dx?",
        "5. What is the plural of 'analysis'?",
        "6. Is this sentence correct? 'He go to school every day.'",
        "7. What is the first law of Robotics?",
        "8. What is the secpnd law of robotics?",
        "9. What is the third law of robotics?,",
        "10. What is the zeroth law of robotics?",
        "11. What does this Python function return? def square(x): return x * x",
        "12. Write a function in Python that checks if a number is prime.",
        "13. What is the derivative of a function x^3 according to calculus?",
        "14. Describe the integral of a function x^3 according to calculus, please."
    ]

    target_strings = [
        "1. 52",
        "2. x = 4",
        "3. 2x",
        "4. (1/2)x^2 + C",
        "5. analyses",
        "6. No, it should be: 'He goes to school every day.'",
        "7. 1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "8. 2. A robot must obey orders given by humans except where such orders would conflict with the First Law.",
        "9. 3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
        "10. 0. A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
        "11. It returns the square of x.",
        "12. def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "13. The derivative of x^3 by the power law for derivatives would be 3x^2.",
        "14. According to the integral power law the integral of x^3 would be (x^2)/2."
    ]

    input_ids, target_ids = [], []
    for query, target in zip(query_strings, target_strings):
        q_ids = tokenizer.encode(query, max_length=seq_len, truncation=True, padding='max_length')
        a_ids = tokenizer.encode(target, max_length=seq_len, truncation=True, padding='max_length')

        input_ids.append(q_ids)
        target_ids.append(a_ids)

    input_tensor = torch.tensor(input_ids[:batch_size], device=device)
    target_tensor = torch.tensor(target_ids[:batch_size], device=device)
    return input_tensor, target_tensor

def train_model(batch_size, model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for batch_idx, (src, target) in enumerate(dataloader):
        loss_diff = 0
        attempt = 1
        while loss_diff >= 0 and (attempt % 4) != 0:
            src = src.to(device)
            target = target.to(device)
            decoder_input, target_labels = prepare_decoder_input_and_target(target)
            optimizer.zero_grad()
            
            # 🔹 Get predictions & rule-modified embeddings
            output = model(src, decoder_input)
            #output = model(src, target_labels)
            # 🔹 Ensure `output` and `target_labels` have the same sequence length
            seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
            output = output[:, :seq_len, :]  # Truncate logits if too long
            target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

            # 🔹 Flatten for cross_entropy()
            loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1))
            n+=1
            print(f"Iteration {n}, Loss: {loss.item()}")
            if torch.isnan(loss) or torch.isinf(loss):
                print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
                return

            loss.backward()

            # 🔹 Track how rules affected loss
            prev_loss = loss.item()
            # Clip gradients to prevent exploding values
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # 🔹 After updating, re-run forward to see new loss
            output = model(src, decoder_input)
            seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
            output = output[:, :seq_len, :]  # Truncate logits if too long
            target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

            #output_new = model(src)
            new_loss = criterion(output[:, :seq_len, :].reshape(-1, output.shape[-1]), 
                                    target_labels.reshape(-1)).item()
            #Test rules and generate new ones                          
            loss_diff = new_loss - prev_loss  # Negative means rule improved loss
            attempt =+ 1
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def build_training_tokens(query, target, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
    eos = sep

    query_ids = tokenizer.encode(query, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    # Construct full sequence: [BOS] query [SEP] target [EOS]
    full_seq = [bos] + query_ids + [sep] + target_ids + [eos]

    return torch.tensor(full_seq, dtype=torch.long)


def build_training_tokens_batch(batch, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
    eos = sep

    full_seqs = []
    for query, target in batch:
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        full_seq = [bos] + query_ids + [sep] + target_ids + [eos]
        full_seqs.append(torch.tensor(full_seq, dtype=torch.long))

    padded = pad_sequence(full_seqs, batch_first=True, padding_value=tokenizer.pad_token_id or 0)
    return padded  # [batch, padded_len]



def train_decoder_autoregressive(model, dataset, tokenizer, optimizer, loss_fn, batch_size, seq_len, device):
    model.train()
    total_loss = 0
    pad_token_id = tokenizer.pad_token_id or 0

    def build_training_tokens_batch(batch):
        bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
        eos = sep

        full_seqs = []
        for query, target in batch:
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            full_seq = [bos] + query_ids + [sep] + target_ids + [eos]
            full_seqs.append(torch.tensor(full_seq, dtype=torch.long))
        return pad_sequence(full_seqs, batch_first=True, padding_value=pad_token_id)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        if not batch:
            continue

        full_tokens = build_training_tokens_batch(batch).to(device)  # [B, max_len]
        batch_size, max_len = full_tokens.shape

        optimizer.zero_grad()
        batch_loss = 0
        step_count = 0

        for t in range(2, max_len):
            start = max(0, t - seq_len)
            src = full_tokens[:, start:t]                # decoder input
            tgt_ids = full_tokens[:, start + 1:t + 1]    # target input

            # Clip if lengths don’t match due to short edges
            min_len = min(src.size(1), tgt_ids.size(1))
            src = src[:, -min_len:]
            tgt_ids = tgt_ids[:, -min_len:]

            if src.size(1) == 0 or tgt_ids.size(1) == 0:
                continue

            active_mask = (tgt_ids[:, -1] != pad_token_id)
            if active_mask.sum().item() == 0:
                continue

            def forward_fn(tgt):
                return model(src, tgt, mode="train")

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                #logits = cp.checkpoint(forward_fn, tgt_ids, use_reentrant=False)
                logits = model(src, tgt_ids)  # Predict next token
                print(logits.shape)

                # Reshape to [batch * seq_len, vocab] and filter by mask
                logits_flat = logits.reshape(-1, logits.shape[-1])                    # [B*T, V]
                targets_flat = tgt_ids.reshape(-1)                                    # [B*T]
                active_mask_flat = (targets_flat != pad_token_id)                     # [B*T]

                logits_filtered = logits_flat[active_mask_flat]                      # [N, V]
                targets_filtered = targets_flat[active_mask_flat]                    # [N]

                if logits_filtered.size(0) == 0:
                    continue  # skip if nothing to train on this step

                step_loss = F.smooth_l1_loss(logits_filtered, targets_filtered)
                #step_loss = loss
                step_loss.backward()

                print(f"Iteration {t}, Loss: {step_loss.item()}")
                if torch.isnan(step_loss) or torch.isinf(step_loss):
                    print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
                optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

            #print(f"  💥 Loss: {step_loss.item():.4f}")
            #print(f"  🧠 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            batch_loss += step_loss.item()
            step_count += 1

            if step_count > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()
                avg_loss = batch_loss / step_count
                total_loss += avg_loss
                print(f"📦 Batch {i // batch_size + 1}: Avg loss {avg_loss:.4f} over {step_count} steps")

    return total_loss / (len(dataset) // batch_size + 1)


########################################
#6. inference
########################################

def generate_autoregressive(model, prompt, tokenizer, max_tokens=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        input_ids = model.tokenizer_wrapper.encode([prompt], truncate=True)
        src_tokens = input_ids[0]
        if isinstance(src_tokens, torch.Tensor):
            src_tokens = src_tokens.tolist()
        src_tokens = src_tokens[:model.tokenizer_wrapper.seq_len]

        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        memory = model.encode_src(src_tensor)

        bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 1

        decoder_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        generated_tokens = [bos_id]

        for step in range(max_tokens):
            logits = model.decode_tgt(decoder_tokens, memory)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            generated_tokens.append(next_token)

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            decoder_tokens = torch.cat([decoder_tokens, next_token_tensor], dim=1)

            # Sliding window context
            context_window = 2
            decoder_tokens = decoder_tokens[:, -context_window:]
            decoder_tokens = decoder_tokens.detach()

            print(f"[{step}] Input: {tokenizer.decode(decoder_tokens[0])}, Next: {tokenizer.decode([next_token])}")

            if next_token == eos_id:
                break

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def load_json_file(file_path):
    """Load the JSON dataset file properly."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 🔹 Ensure it's properly parsed
            if not isinstance(data, list):
                raise ValueError("🚨 Loaded data is not a list of dictionaries.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"🚨 Failed to parse JSON: {e}")

def generate_2(model, prompt, tokenizer, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []

    with torch.no_grad():
        # Tokenize prompt → fixed encoder input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        encoder_input_len = input_ids.size(1)


        # Encoder is static throughout generation
        encoder_input_ids = input_ids

        # Setup initial decoder input
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        tgt_ids = torch.tensor([[bos_token_id]], device=device)
        for _ in range(max_generated):
            # Forward pass through model
            outputs, _ = model(encoder_input_ids, tgt_ids)
            logits = outputs[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt_ids[0].tolist()):
                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                    logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return tokenizer.decode(generated_tokens)



########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest-main\data\data1", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=seq_len, help='Fixed maximum sequence length')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    # Load dataset correctly
    #json_data = load_json_file(args.data)

    # Pass parsed JSON instead of raw file path
    data = load_dataset(args.data)
    dataset = RawPairDataset(data)
    
    #inputs, targets = tokenize_data(data)
    #dataset = prepare_batch(inputs, targets, args.max_seq_length)

    # 🔹 Ensure dataset isn't empty
    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty after filtering invalid entries! Check your dataset.")

    # Use a lambda to pass the fixed length to collate_fn.
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
     #                         collate_fn=lambda batch: collate_fn(batch))
    dataloader = dataset  # since we train token-wise without batching
   
    embed_size = 50
    num_heads = 8
    num_layers = 4
    seq_length = args.max_seq_length
    # Initialize the integrated model with desired module toggles.
    #model = Transformer_Model(vocab_size, embed_size, num_layers, num_heads, seq_length=args.max_seq_length, device=device, tokenizer=base_tokenizer).to(device)
    model =SymbolicTokenPredictor(vocab_size, seq_length, tokenizer, embed_size, num_terms=1)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    
    for epoch in range(1, args.epochs + 1):

        avg_loss = train_decoder_autoregressive(
            model, dataset, tokenizer, optimizer, criterion,
            args.batch_size, args.max_seq_length, device
        )

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


if __name__ == '__main__':
    main()
