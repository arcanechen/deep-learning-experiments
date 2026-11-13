"""Small FP32 learning runs. Reads existing TRAIN/VAL data; never evaluates test.

Run from the project root. Outputs go to tutorials/runs/<unique-name>/.
This teaching script does not implement exact checkpoint resume; study the
production weather trainer for that feature. A small-run score is not a paper result.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'weather'))
from models import LeNet
from data import preprocess
from weather_model import WeatherWindows, WeatherForecaster


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_practice(task, train_limit, val_limit):
    """Fixed subset membership; changing the training seed does not change the split."""
    if task == 'cnn':
        raw = FashionMNIST(str(ROOT/'data'), train=True, download=False)
        path = ROOT/'data/splits/fashion_mnist_seed2026.npz'
        with np.load(path) as f:
            train_ids, val_ids = f['train'], f['val']
        assert not np.intersect1d(train_ids, val_ids).size
        rng = np.random.default_rng(2026)
        train_ids = rng.permutation(train_ids)[:train_limit]
        val_ids = rng.permutation(val_ids)[:val_limit]
        make = lambda ids: TensorDataset(raw.data[ids].unsqueeze(1), raw.targets[ids])
        return make(train_ids), make(val_ids), {'split_sha256':sha(path),
            'train_ids':train_ids.tolist(),'val_ids':val_ids.tolist(),'target_std':None}
    path = ROOT/'weather/data/prepared.npz'
    meta = json.loads((ROOT/'weather/data/preprocessing.json').read_text(encoding='utf-8'))
    assert sha(path) == meta['prepared_sha256']
    with np.load(path) as f:
        values = f['values']
        train_all, val_all = f['train_starts'], f['val_starts']
    select = lambda a,n: a[np.linspace(0,len(a)-1,min(n,len(a)),dtype=int)]
    train_ids, val_ids = select(train_all,train_limit), select(val_all,val_limit)
    assert train_ids.max()+336 <= meta['splits']['train']['end_index_exclusive']
    assert val_ids.min() >= meta['splits']['val']['start_index']
    make = lambda ids: WeatherWindows(values,ids,meta['target_index'])
    return make(train_ids), make(val_ids), {'prepared_sha256':meta['prepared_sha256'],
        'train_window_starts':train_ids.tolist(),'val_window_starts':val_ids.tolist(),
        'features':meta['features'],'target_index':meta['target_index'],
        'target_mean':meta['mean']['T (degC)'],'target_std':meta['std_population']['T (degC)']}


def run_epoch(model, loader, task, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sum_loss, count, correct = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss() if task == 'cnn' else nn.MSELoss()
    with torch.set_grad_enabled(training):
        for raw_x, raw_y in loader:
            x = preprocess(raw_x,device) if task == 'cnn' else raw_x.to(device)
            y = raw_y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if task == 'cnn':
                assert prediction.shape == (len(y),10) and y.dtype == torch.long
            else:
                assert prediction.shape == y.shape == (len(y),168)
            loss = criterion(prediction,y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss: inspect data, learning rate and gradients')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                optimizer.step()
            units = len(y) if task == 'cnn' else y.numel()
            sum_loss += loss.detach().item()*units
            count += units
            if task == 'cnn':
                correct += (prediction.argmax(1)==y).sum().item()
    return {'loss':sum_loss/count, 'accuracy':correct/count if task=='cnn' else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['cnn','rnn','gru','lstm'],default='cnn')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--train-limit',type=int,default=512)
    p.add_argument('--val-limit',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--one-batch-steps',type=int,default=0)
    args=p.parse_args()
    if min(args.epochs,args.train_limit,args.val_limit,args.batch_size,args.lr)<=0 or args.one_batch_steps<0:
        p.error('Sizes, epochs and learning rate must be positive; steps cannot be negative')
    if args.device=='cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable: use --device cpu or inspect the environment')
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    train,val,info=load_practice(args.task,args.train_limit,args.val_limit)
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator,num_workers=0)
    validation=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=LeNet() if args.task=='cnn' else WeatherForecaster(args.task)
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    folder=ROOT/'tutorials/runs'/f'{args.task}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    folder.mkdir(parents=True,exist_ok=False)
    config={**vars(args),'torch':str(torch.__version__),'data':info,'script_sha256':sha(__file__),
            'purpose':'learning_only_not_formal_results','test_evaluated':False}
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    raw_x,y=next(iter(loader));x=preprocess(raw_x,device) if args.task=='cnn' else raw_x.to(device)
    print('RUN_DIR=',folder,flush=True)
    print('INPUT',tuple(x.shape),x.dtype,x.device,'TARGET',tuple(y.shape),y.dtype,flush=True)
    with torch.no_grad():
        print('OUTPUT',tuple(model(x).shape),'PARAMETERS',sum(p.numel() for p in model.parameters()),flush=True)
    if args.one_batch_steps:
        # Each pass through a one-element list performs exactly one optimizer update.
        loader=[(raw_x,y)]
        validation=[(raw_x,y)]
        rounds=args.one_batch_steps
        label='same_training_batch'
    else:
        rounds=args.epochs
        label='validation_subset'
    best=float('inf');history=[]
    for epoch in range(1,rounds+1):
        train_metrics=run_epoch(model,loader,args.task,device,optimizer)
        val_metrics=run_epoch(model,validation,args.task,device)
        record={'epoch_or_step':epoch,'train':train_metrics,'measurement':val_metrics,'measurement_scope':label}
        if args.task!='cnn':
            record['measurement_rmse_degC']=val_metrics['loss']**.5*info['target_std']
        history.append(record)
        if val_metrics['loss']<best:
            best=val_metrics['loss']
            torch.save({'model':model.state_dict(),'config':config,'epoch_or_step':epoch,'measurement_scope':label},folder/'best.pt')
        print(json.dumps(record,ensure_ascii=False),flush=True)
    (folder/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    # Reload saved best weights and generate an inspectable prediction batch.
    ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model']);model.eval()
    bx,by=next(iter(validation));bx=preprocess(bx,device) if args.task=='cnn' else bx.to(device)
    with torch.inference_mode():
        output=model(bx).cpu().numpy()
    np.savez_compressed(folder/'example_prediction.npz',prediction=output,target=by.numpy())
    print('Practice finished. No test metric was computed. Full resume is not implemented here.',flush=True)


if __name__=='__main__':
    main()
