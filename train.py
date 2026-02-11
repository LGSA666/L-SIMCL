import os
os.environ['NCCL_P2P_DISABLE'] = '1'
import random
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import datasets
from tqdm import tqdm
import argparse
from eval import evaluate

import utils
from models.model import LSIMCLModel
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
import swanlab

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--data', type=str, default='WebOfScience')
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--swanlab', default=False, action='store_true')
    parser.add_argument('--arch', type=str, default='bert-base-uncased')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)
    # LSIMCL Parameters
    parser.add_argument('--lambda_1', type=float, default=0.1, help='Sample-level CL weight')
    parser.add_argument('--lambda_2', type=float, default=0.1, help='Label-level CL weight')
    parser.add_argument('--temp', type=float, default=0.1, help='Contrastive Learning Temperature')
    parser.add_argument('--heads', type=int, default=4, help='Number of Label-Aware Attention Heads')
    parser.add_argument('--dropout_cl_rate', type=float, default=0.1, help='Dropout rate (Sample-level CL)')
    
    # Loss Function Parameters
    parser.add_argument('--loss', type=str, default='zlpr', choices=['zlpr', 'bce', 'hbm'],
                       help='Classification Loss Function: zlpr (default), bce, or hbm')
    parser.add_argument('--hbm_alpha', type=float, default=1.0, help='HBM loss alpha')
    parser.add_argument('--hbm_margin', type=float, default=0.1, help='HBM loss margin')
    parser.add_argument('--hbm_loss_mode', type=str, default='unit', choices=['unit', 'hierarchy'],
                       help='HBM Loss Mode')
    
    # Inference and Training
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification Threshold during Inference')
    parser.add_argument('--epoch', type=int, default=100, help='Max Training Epochs')

    return parser

class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
    def __call__(self, score, best_score, name):
        abs_name = os.path.abspath(name)
        save_dir = os.path.dirname(abs_name)
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if os.path.exists(abs_name):
            try:
                os.remove(abs_name)
            except OSError:
                pass
        
        torch.save({'param': model_to_save.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args, 'best_score': best_score}, abs_name)

if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    print(args)
    utils.seed_torch(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name), exist_ok=True)

    if args.swanlab:
        swanlab.init(config=args, project='L-SIMCL')
    logger = utils.init_logger(os.path.join('checkpoints', args.name, 'run.log'))
    logger.info(args)
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v: num_class = v
    num_class += 1
    
    # Construct path_list (Edge List for Graph)
    # Filter edges connected to -1 (root parent)
    path_list = [(i, v) for v, i in value2slot.items() if v != -1 and i != -1]
    for i in range(num_class):
        if i not in value2slot: value2slot[i] = -1

    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth
    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    label_depths = [depth_dict[i] for i in range(num_class)]

    # Data Preprocessing
    if os.path.exists(os.path.join(data_path, 'LSIMCL_processed')):
        dataset = datasets.load_from_disk(os.path.join(data_path, 'LSIMCL_processed'))
    else:
        dataset = datasets.load_dataset('json',
                                        data_files={'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                    'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                    'test': 'data/{}/{}_test.json'.format(args.data, args.data)})
        def data_map_function(batch, tokenizer):
            new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
            for l, t in zip(batch['label'], batch['token']):

                label_vec = [0] * num_class
                
                # Set true label position to 1
                for i in l:
                    if i < num_class: # Ensure index within bounds
                        label_vec[i] = 1
                
                new_batch['labels'].append(label_vec)
                
                tokens = tokenizer(t, truncation=True, max_length=512, padding='max_length')
                new_batch['input_ids'].append(tokens['input_ids'])
                new_batch['attention_mask'].append(tokens['attention_mask'])
                new_batch['token_type_ids'].append(tokens.get('token_type_ids', [0] * 512))
            return new_batch
            
        dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
        dataset.save_to_disk(os.path.join(data_path, 'LSIMCL_processed'))

    dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
    dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
    dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])

    if args.low_res:
        if os.path.exists(os.path.join(data_path, 'low.json')):
            index = json.load(open(os.path.join(data_path, 'low.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(data_path, 'low.json'), 'w'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
    
    # Initialize Model
    model = LSIMCLModel.from_pretrained(
        args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
        graph_type=args.graph, data_path=data_path, depth2label=depth2label,
        lambda_1=args.lambda_1, lambda_2=args.lambda_2, temp=args.temp,
        loss_type=args.loss, hbm_alpha=args.hbm_alpha, hbm_margin=args.hbm_margin,
        hbm_loss_mode=args.hbm_loss_mode,
        label_depths=label_depths,
        heads=args.heads,
        dropout_cl_rate=args.dropout_cl_rate
    )
    logger.info(model)
    logger.info(f"Total params: {sum(param.numel() for param in model.parameters()) / 1000000.0}M. ")
    
    model.to('cuda')
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    update_step = 0
    loss = 0
    cls_loss_sum = 0
    contrastive_loss_sum = 0
    sample_loss_sum = 0
    
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.makedirs(os.path.join('checkpoints', args.name), exist_ok=True)

    for epoch in range(args.epoch):
        logger.info("------------ epoch {} ------------".format(epoch + 1))
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break
        model.train()
        with tqdm(train) as p_bar:
            for batch in p_bar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch, label_depths=label_depths)

                loss_val = output['loss'] if isinstance(output, dict) else output.loss
                if loss_val.dim() > 0:
                    loss_val = loss_val.mean()
                
                loss_to_backward = loss_val / args.update
                loss_to_backward.backward()
                loss += loss_val.item()
                
                if isinstance(output, dict):
                    cls_l = output.get('cls_loss', None)
                    con_l = output.get('contrastive_loss', None)
                    ins_l = output.get('sample_loss', None)
                else:
                    cls_l = getattr(output, 'cls_loss', None)
                    con_l = getattr(output, 'contrastive_loss', None)
                    ins_l = getattr(output, 'sample_loss', None)

                if cls_l is not None:
                    if cls_l.dim() > 0: cls_l = cls_l.mean()
                    cls_loss_sum += cls_l.item()
                if con_l is not None:
                    if con_l.dim() > 0: con_l = con_l.mean()
                    contrastive_loss_sum += con_l.item()
                if ins_l is not None:
                    if ins_l.dim() > 0: ins_l = ins_l.mean()
                    sample_loss_sum += ins_l.item()
                
                update_step += 1
                if update_step % args.update == 0:
                    avg_loss = loss / args.update
                    avg_cls_loss = cls_loss_sum / args.update
                    avg_contrastive_loss = contrastive_loss_sum / args.update
                    avg_sample_loss = sample_loss_sum / args.update
                    
                    if args.swanlab:
                        swanlab.log({
                            'loss': avg_loss,
                            'cls_loss': avg_cls_loss,
                            'contrastive_loss': avg_contrastive_loss,
                            'sample_loss': avg_sample_loss,
                        })
                    p_bar.set_description('loss:{:.4f} cls:{:.4f} con_l:{:.4f} con_i:{:.4f}'.format(
                        avg_loss, avg_cls_loss, avg_contrastive_loss, avg_sample_loss))
                    
                    # Gradient clipping for numerical stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    loss = 0
                    cls_loss_sum = 0
                    contrastive_loss_sum = 0
                    sample_loss_sum = 0
                    update_step = 0
        
        # Validation
        model.eval()
        pred, gold = [], []
        with torch.no_grad(), tqdm(dev) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                actual_model = model.module if hasattr(model, 'module') else model
                output_ids, logits = actual_model.generate(batch['input_ids'], depth2label=depth2label, threshold=args.threshold)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set(out))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1: gold[-1].append(i)
        
        scores = evaluate(pred, gold, label_dict)
        macro_f1, micro_f1 = scores['macro_f1'], scores['micro_f1']
        logger.info(' macro: {:.4f}, micro: {:.4f}'.format(macro_f1, micro_f1))
        print('macro', macro_f1, 'micro', micro_f1)
        
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            logger.info(f"New best macro F1: {best_score_macro:.4f}. Checkpoint saved.")
            early_stop_count = 0
        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            logger.info(f"New best micro F1: {best_score_micro:.4f}. Checkpoint saved.")
            early_stop_count = 0
        
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))

        if args.swanlab:
            swanlab.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

    # Test
    test = DataLoader(dataset['test'], batch_size=8, shuffle=False)
    
    def test_function(extra):
        rel_path = os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra))
        abs_path = os.path.abspath(rel_path)
        
        checkpoint = torch.load(abs_path, map_location='cpu')

        logger.info(f'Test load checkpoint: {checkpoint.keys()}')  # log keys instead of full content for brevity
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['param'])
        else:
            model.load_state_dict(checkpoint['param'])
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                actual_model = model.module if hasattr(model, 'module') else model
                output_ids, logits = actual_model.generate(batch['input_ids'], depth2label=depth2label, threshold=args.threshold)
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        # 1. Standard Metrics
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']

        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)

    test_function('_macro')
    test_function('_micro')