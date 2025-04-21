
from ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from vae.utils.utils import load_config
import torch 



from VQ.autoencoder import VQModel



def train(config):

    config = load_config(config_path=config)
    print("COnfig: ", config)
    
    dataset = ImageDataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["data"]["dataloader"]["batch_size"],
        shuffle=config["data"]["dataloader"]["shuffle"],
        num_workers=config["data"]["dataloader"].get('num_workers', 0)
    )

    model = VQModel(
        ddconfig=config['model']['params']['ddconfig'],
        n_embed=config['model']['params']['n_embed'],
        embed_dim=config['model']['params']['embed_dim'],
        lossconfig=config['model']['params']['lossconfig'],
        monitor='val/total_loss',
        use_ema=True
    )

    callbacks = [
        ModelCheckpoint(
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            dirpath="./checkpoints",
            filename='vae-{epoch:02d}-{val_loss:.2f}',
            save_last=True
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]


    # trainer configuration 
    trainer = Trainer(
        max_epochs=1,
        callbacks=callbacks,
        devices=1 if torch.cuda.is_available() else None,  # More flexible device handling
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        precision=32,  # Can use '16-mixed' for FP16
        accumulate_grad_batches=1
    )

    # Training 
    trainer.fit(model, dataloader)
    print("Training completed!")




if __name__ == "__main__":
    train(config="E:\\YouTube\\stable-diffusion\\VQ\\config.yaml")