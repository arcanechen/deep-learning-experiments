"""Evaluate the frozen nine-run plan and naive baselines on identical test origins."""
import json
import csv
import numpy as np
from train_weather import ROOT, DATA, torch, WeatherWindows, WeatherForecaster, DataLoader, sha, atomic, config_for, setup, lock


def metrics(predicted, actual):
    error = predicted.astype(np.float64) - actual.astype(np.float64)
    return {'mae_degC': float(np.abs(error).mean()), 'rmse_degC': float(np.sqrt(np.square(error).mean())),
            'horizon_mae_degC': np.abs(error).mean(axis=0).tolist(),
            'horizon_rmse_degC': np.sqrt(np.square(error).mean(axis=0)).tolist(),
            'day_mae_degC': np.abs(error).reshape(-1,7,24).mean(axis=(0,2)).tolist(),
            'day_rmse_degC': np.sqrt(np.square(error).reshape(-1,7,24).mean(axis=(0,2))).tolist()}


def main():
    plan = json.loads((ROOT/'experiment_plan.json').read_text(encoding='utf-8'))
    meta = json.loads((DATA/'preprocessing.json').read_text(encoding='utf-8'))
    assert sha(DATA/'prepared.npz') == meta['prepared_sha256']
    device = torch.device('cuda')
    # Check the whole queue BEFORE making any test predictions.
    checkpoints = []
    for seed in plan['seeds']:
        for cell in plan['models']:
            path = ROOT/'runs'/f'{cell}_s{seed}'/'last.pt'
            if not path.exists():
                raise RuntimeError(f'Train the complete plan before test evaluation: missing {path}')
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert state['config'] == config_for(plan, cell, seed, meta, device), f'Configuration changed: {path}'
            assert state['epoch'] == plan['epochs'] and len(state['history']) == plan['epochs'], f'Incomplete: {path}'
            best_index = int(np.argmin([r['val_mse'] for r in state['history']]))
            assert state['best_epoch'] == best_index+1
            assert state['best_val_mse'] == state['history'][best_index]['val_mse']
            checkpoints.append((cell,seed,state,sha(path)))
    setup(42)
    data = dict(np.load(DATA/'prepared.npz'))
    target = meta['target_index']
    mean, std = meta['mean']['T (degC)'], meta['std_population']['T (degC)']
    starts = data['test_starts']
    target_indices = starts[:,None] + 168 + np.arange(168)
    actual = (data['values'][target_indices,target].astype(np.float64)*std + mean).astype(np.float32)
    loader = DataLoader(WeatherWindows(data['values'],starts,target),batch_size=128,shuffle=False,num_workers=0)
    out = ROOT/'reports'/'evaluation'
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'targets.npz', actual_degC=actual, window_starts=starts,
                        forecast_origin_seconds=data['time_seconds'][starts+167])
    results = {}
    baseline_predictions = {
        'last_temperature': np.repeat(data['values'][starts+167,target,None],168,axis=1),
        'repeat_last_24_hours': np.tile(data['values'][starts[:,None]+144+np.arange(24),target],(1,7)),
        'repeat_last_168_hours': data['values'][starts[:,None]+np.arange(168),target]}
    for name, p in baseline_predictions.items():
        pred = (p.astype(np.float64)*std+mean).astype(np.float32)
        results[name] = metrics(pred,actual)
        np.savez_compressed(out/(name+'.npz'),predicted_degC=pred,window_starts=starts)
    for cell,seed,state,checkpoint_sha in checkpoints:
        model = WeatherForecaster(cell,len(meta['features']),plan['hidden_size'],plan['horizon_hours']).to(device)
        model.load_state_dict(state['best_model']); model.eval()
        predictions = []
        with torch.inference_mode():
            for x,y in loader:
                predictions.append(model(x.to(device)).cpu().numpy())
        pred = (np.concatenate(predictions).astype(np.float64)*std+mean).astype(np.float32)
        assert pred.shape == actual.shape and np.isfinite(pred).all()
        name = f'{cell}_s{seed}'
        results[name] = {**metrics(pred,actual), 'best_epoch':state['best_epoch'],
                         'best_val_rmse_degC':float(np.sqrt(state['best_val_mse'])*std),
                         'parameters':sum(p.numel() for p in model.parameters()),
                         'train_val_seconds':sum(r['train_val_seconds'] for r in state['history']),
                         'checkpoint_sha256':checkpoint_sha}
        np.savez_compressed(out/(name+'.npz'), predicted_degC=pred,window_starts=starts)
        print(name, 'MAE', results[name]['mae_degC'], 'RMSE',results[name]['rmse_degC'],flush=True)
    # Re-read saved predictions and verify independently, including baseline alignment.
    for name,result in results.items():
        saved = np.load(out/(name+'.npz'))
        assert np.array_equal(saved['window_starts'], starts)
        errors = (saved['predicted_degC'].astype(np.float64)-actual).ravel()
        assert abs(np.abs(errors).sum()/len(errors)-result['mae_degC']) < 1e-12
        assert abs(np.sqrt(np.dot(errors,errors)/len(errors))-result['rmse_degC']) < 1e-10
    atomic(out/'metrics.json', {'origins':len(starts),'horizon_hours':168,'prediction_pairs':actual.size,
           'prepared_sha256':meta['prepared_sha256'], 'evaluation_source_sha256':sha(__file__), 'results':results})
    lines = ['# 气温预测实验：独立测试结果', '',
             '9 次训练均完成后统一测试，以最低验证 MSE 选取权重。所有模型与基线使用相同的 6269 个滚动预测起点，每个预测 168 小时，共 1053192 个起点—预测步配对。MAE、RMSE 单位均为 °C，越低越好。', '',
             '| 方法 | MAE（°C） | RMSE（°C） |', '|---|---:|---:|']
    for name in baseline_predictions:
        r=results[name]; lines.append(f'| {name} | {r["mae_degC"]:.3f} | {r["rmse_degC"]:.3f} |')
    for cell in plan['models']:
        ms = np.array([[results[f'{cell}_s{s}'][k] for k in ['mae_degC','rmse_degC']] for s in plan['seeds']])
        m,sd = ms.mean(axis=0),ms.std(axis=0,ddof=1)
        lines.append(f'| {cell.upper()}（三种子均值 ± 样本标准差） | {m[0]:.3f} ± {sd[0]:.3f} | {m[1]:.3f} ± {sd[1]:.3f} |')
    lines += ['', '标准差描述随机初始化和训练顺序带来的波动，不是置信区间。滚动窗口重叠，同一真实小时可能对应多个预测起点；上述指标按起点—预测步配对等权统计，不能当作 1053192 个独立观测。测试段不足一年，结论不代表所有年份与季节。', '',
              '| 运行 | 验证最优轮次 | 验证 RMSE | 测试 MAE | 测试 RMSE | 训练+验证秒数 |', '|---|---:|---:|---:|---:|---:|']
    for cell,seed,_,_ in checkpoints:
        name=f'{cell}_s{seed}';r=results[name]
        lines.append(f'| {name} | {r["best_epoch"]} | {r["best_val_rmse_degC"]:.3f} | {r["mae_degC"]:.3f} | {r["rmse_degC"]:.3f} | {r["train_val_seconds"]:.1f} |')
    lines += ['', '计时只包含每轮训练与验证，不含存盘、下载与进程启动；同隐藏维度不代表同参数量。详细逐小时与逐天误差见 metrics.json，逐窗口预测见同目录 NPZ 文件。', '',
              '![Validation curves](validation_curves.png)', '', '![Horizon errors](horizon_rmse.png)', '',
              '![First test forecast](first_forecast.png)', '', '示例固定为测试集第一个可用起点、seed42，没有按误差挑选好看的案例。']
    (out/'结果汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for cell,seed,state,_ in checkpoints:
        if seed==42:
            ax.plot(range(1,21),[np.sqrt(r['val_mse'])*std for r in state['history']],label=cell.upper())
    ax.set(xlabel='Epoch',ylabel='Validation RMSE (°C)',title='Seed 42 validation curves');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for name in baseline_predictions:
        ax.plot(range(1,169),results[name]['horizon_rmse_degC'],linestyle='--',alpha=.65,label=name)
    for cell in plan['models']:
        curve=np.mean([results[f'{cell}_s{s}']['horizon_rmse_degC'] for s in plan['seeds']],axis=0)
        ax.plot(range(1,169),curve,label=cell.upper()+' (mean of 3 RMSEs)')
    ax.set(xlabel='Forecast horizon (hours)',ylabel='RMSE (°C)',title='Test error by forecast horizon');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'horizon_rmse.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(range(1,169),actual[0],color='black',label='Observed')
    for cell in plan['models']:
        pred=np.load(out/f'{cell}_s42.npz')['predicted_degC']
        ax.plot(range(1,169),pred[0],label=cell.upper())
    ax.set(xlabel='Forecast horizon (hours)',ylabel='Temperature (°C)',title='First eligible test origin, seed 42');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'first_forecast.png',dpi=150);plt.close(fig)
    atomic(out/'verification.json', {'verified_model_runs':9,'verified_baselines':3,'predictions_reloaded_and_metrics_recomputed':True,
           'all_best_epochs_checked':True,'all_checkpoints_complete_and_config_matched':True,'plot_visual_review':'pending'})


if __name__ == '__main__':
    with lock(ROOT/'runs'/'queue.lock'):
        main()
