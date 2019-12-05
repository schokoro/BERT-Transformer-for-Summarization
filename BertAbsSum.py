import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
import json
import time
import torch.nn.functional as F
from preprocess import LCSTSProcessor, DataLoader, DataProcessor
from model import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from preprocess import convert_examples_to_features
from tqdm import tqdm, trange
from transformer import Constants


model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def cal_loss(dratf_logits, refine_logits, ground):
    ground = ground[:, 1:]
    draft_loss = F.cross_entropy(dratf_logits, ground, ignore_index=Constants.PAD)
    refine_loss = F.cross_entropy(refine_logits, ground, ignore_index=Constants.PAD)
    return draft_loss + refine_loss

class ARGS(object):
    data_dir = 'data/processed_data'
    bert_model = 'bert-base-uncased'
    output_dir = 'output'
    GPU_index = 0
    learning_rate = 5e-5
    num_train_epochs = 3
    warmup_proportion = 0.1
    max_src_len = 130
    max_tgt_len = 30
    train_batch_size = 32
    decoder_config = None
    print_every = 100

args = ARGS()

if torch.cuda.is_available():
    device = torch.device('cuda', args.GPU_index)
else:
    device = torch.device('cpu')
logger.info(f'Using device:{device}')

print(device)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
model_path = os.path.join(args.output_dir, time.strftime('model_%m-%d-%H:%M:%S', time.localtime()))
os.mkdir(model_path)
logger.info(f'Saving model to {model_path}.')

if args.decoder_config is not None:
    with open(args.decoder_config, 'r') as f:
        decoder_config = json.load(f)
else:
    with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
        bert_config = json.load(f)
        decoder_config = {}
        decoder_config['len_max_seq'] = args.max_tgt_len
        decoder_config['vocab_size'] = bert_config['vocab_size']
        decoder_config['n_layers'] = 8
        decoder_config['num_head'] = 12
        decoder_config['d_k'] = 64
        decoder_config['d_v'] = 64
        decoder_config['d_model'] = bert_config['hidden_size']
        decoder_config['d_inner'] = decoder_config['d_model']

# data preprocess
processor = LCSTSProcessor()
tokenizer = BertTokenizer.from_pretrained(args.bert_model)
logger.info('Loading train examples...')
train_examples = processor.get_examples('data/processed_data')
num_train_optimization_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs
logger.info('Converting train examples to features...')
features = convert_examples_to_features(train_examples, args.max_src_len, args.max_tgt_len, tokenizer)
example = train_examples[0]
example_feature = features[0]
logger.info("*** Example ***")
logger.info("guid: %s" % (example.guid))
logger.info("src text: %s" % example.src)
logger.info("src_ids: %s" % " ".join([str(x) for x in example_feature.src_ids]))
logger.info("src_mask: %s" % " ".join([str(x) for x in example_feature.src_mask]))
logger.info("tgt text: %s" % example.tgt)
logger.info("tgt_ids: %s" % " ".join([str(x) for x in example_feature.tgt_ids]))
logger.info("tgt_mask: %s" % " ".join([str(x) for x in example_feature.tgt_mask]))
logger.info('Building dataloader...')
all_src_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
all_src_mask = torch.tensor([f.src_mask for f in features], dtype=torch.long)
all_tgt_ids = torch.tensor([f.tgt_ids for f in features], dtype=torch.long)
all_tgt_mask = torch.tensor([f.tgt_mask for f in features], dtype=torch.long)
train_data = TensorDataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


model = BertAbsSum(args.bert_model, decoder_config, device)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args.learning_rate,
                     warmup=0.1,
                     t_total=num_train_optimization_steps)
print(decoder_config['n_head'])


logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args.train_batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)
model.train()
global_step = 0
for i in trange(int(args.num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        draft_logits, refine_logits = model(*batch)
        loss = cal_loss(draft_logits, refine_logits, batch[2])
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += batch[0].size(0)
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
    if step % args.print_every == 0:
        logger.info(f'Epoch {i}, step {step}, loss {loss.item()}.')
    torch.save(model.state_dict(), os.join(model_path, 'BertAbsSum.bin'))
    logger.info(f'Epoch {i} finished. Model saved.')

