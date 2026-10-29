import torch
import os
import sys
from torch import nn
from tqdm import trange

# ---------------------------------------------------------------------------
# wandb is optional — use --use-wandb flag (default: False)
# ---------------------------------------------------------------------------
try:
    import wandb as _real_wandb
    _WANDB_REAL = True
except ImportError:
    _real_wandb = None
    _WANDB_REAL = False


class _NoOpWandb:
    """Drop-in no-op replacement for wandb when disabled or not installed."""
    def init(self, **kwargs): pass
    def log(self, *a, **kw): pass


wandb = _NoOpWandb()  # default to no-op; overridden below if enabled


from data import prepare_data, return_dataloader, split_dataset_by_fold
from models import declare_model
from engine.trainer import train, test
from config import args, MODELS, metrics_classification, metrics_regression


def main():
    # ---- Device selection (no hard CPU exit) --------------------------------
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available() and args.device != 'cpu':
            print(f"[WARN] CUDA not available. Falling back to CPU.")

    # ---- Optional wandb ------------------------------------------------------
    global wandb
    use_wandb = getattr(args, 'use_wandb', False)
    if use_wandb and _WANDB_REAL:
        wandb = _real_wandb
    else:
        wandb = _NoOpWandb()

    torch.manual_seed(42)

    # ---- Data path based on mode --------------------------------------------
    if args.exp_mode == '5_fold_val':
        data = args.data
    elif args.exp_mode in ['pure_train', 'train_wo_valid']:
        data = args.train_data
    elif args.exp_mode == 'pure_test':
        data = args.test_data
    elif args.exp_mode == 'train_and_test':
        data = [args.train_data, args.test_data]
    else:
        sys.exit('Invalid exp_mode specified')

    # ---- Model list ----------------------------------------------------------
    if args.model == 'ensdti':
        models = 'ensdti'
    elif args.model == 'default':
        models = MODELS
    elif args.model.startswith('ablation'):
        modelsToRemove = args.model.replace("ablation_", "").split("_")
        models = [m for m in MODELS if m not in modelsToRemove]
    elif '_' in args.model:
        models = args.model.split('_')
    else:
        models = [args.model]

    # ---- Prepare data -------------------------------------------------------
    data_obj = prepare_data(data, models, args)

    # ---- Per-model training / testing ---------------------------------------
    for m in models:
        try:
            model = declare_model(m, args.task, data_obj if m in ['dpdta', 'cpi'] else None).to(device)
        except (ImportError, NotImplementedError) as e:
            print(f"[SKIP] {m}: {e}")
            continue

        if args.task == 'classification':
            loss_fn = nn.BCELoss()
            args.metrics = metrics_classification
        else:
            loss_fn = nn.MSELoss()
            args.metrics = metrics_regression

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.exp_mode == 'pure_test':
            test_loader = return_dataloader(data_obj[m], args.batch, shuffle=False)
            model.load_state_dict(torch.load(
                f'{os.getcwd()}/dataset/{args.data}/{m}/{args.model_mode}/{args.custom}/final_model_{args.custom}.pt',
                map_location=device))
            result, perf = test('Test', test_loader, model, loss_fn, args)
        else:
            if args.exp_mode == '5_fold_val':
                datasets, all_idx = data_obj
                for fold in range(5):
                    train_loader, valid_loader, test_loader = split_dataset_by_fold(
                        datasets[m], all_idx, fold, args.batch)
                    wandb.init(project='ensdti', name=f'{m}_fold{fold}', config=args, reinit=True)
                    train(train_loader, model, loss_fn, optimizer, args, valid_loader=valid_loader)
                    result, perf = test('Test', test_loader, model, loss_fn, args)
            elif args.exp_mode in ['pure_train', 'train_and_test']:
                train_datasets, valid_datasets = data_obj[:2]
                train_loader = return_dataloader(train_datasets[m], args.batch, shuffle=True)
                valid_loader = return_dataloader(valid_datasets[m], args.batch, shuffle=False)
                wandb.init(project='ensdti', name=f'{m}_train', config=args, reinit=True)
                train(train_loader, model, loss_fn, optimizer, args, valid_loader=valid_loader)
                if args.exp_mode == 'train_and_test':
                    test_loader = return_dataloader(data_obj[2][m], args.batch, shuffle=False)
                    result, perf = test('Test', test_loader, model, loss_fn, args)

        # ---- Save results --------------------------------------------------
        if args.save_result:
            out_path = f'{os.getcwd()}/dataset/{args.data}/{m}/{args.exp_mode}/{args.custom}'
            os.makedirs(out_path, exist_ok=True)
            result.to_csv(f'{out_path}/test_result.csv', index=False)
        if args.save_perf:
            perf.to_csv(f'{out_path}/test_perf.csv', index=False)


if __name__ == '__main__':
    main()
