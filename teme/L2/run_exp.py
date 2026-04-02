def run_exp(config_path):
    import os
    from convtasnet import ConvTasNet, SI_SNR_PIT
    from hw_utils import prep_dataset, LibriMixDataset
    import lighter
    import torch
    from torch.utils.data import DataLoader
    import toml

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    config_dir, _ = os.path.split(config_path)
    working_dir = config.get('working_dir', config_dir)

    os.chdir(working_dir)

    prep_dataset(config['dataset']['root'])
    
    model = ConvTasNet(**config['model'])
    
    model.compile(
        torch.optim.Adam(model.parameters(), **config['optimizer']),
        SI_SNR_PIT(),
        metrics=[],
        device=device,
    )
    
    model_path = config['model_path']
    
    train_dataset = LibriMixDataset(
        subset="train",
        typ='clean',
        **config['dataset'],
    )
    
    val_dataset = LibriMixDataset(
        subset="val",
        typ='clean',
        **config['dataset'],
    )
    
    train_loader = DataLoader(train_dataset, **config['dataloader'])
    val_loader   = DataLoader(  val_dataset, **config['dataloader'])
    
    chkpoint_path = config['chkpoint_path']
    log_path      = config['log_path']
    
    model.fit(
        train_loader,
        validation_loader=val_loader,
        callbacks=[
            lighter.callbacks.CSVLogger(log_path),
            lighter.callbacks.History(),
            lighter.callbacks.Checkpoint(
                chkpoint_path,
                save_best_only=True
            ),
        ],
        **config['fit'],
    )

    # Save the final best...
    model.save(model_path)


if __name__ == '__main__':
    import os
    print(__file__)
    file_dir, _ = os.path.split(__file__)
    if not (os.path.abspath(os.getcwd()) is os.path.abspath(file_dir)):
        os.chdir(file_dir)

    configs = [
        # 'experiments/arch1/conf.toml',
        'experiments/arch2/conf.toml',
    ]

    for config in configs:
        os.chdir(file_dir)
        run_exp(config)

