from ldm.models.diffusion.ddpm import LatentDiffusion



if __name__ == "__main__":
    
    import torch 
    import torch.nn as nn 
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    
    from ImageDataset import ImageDataset
    from vae.utils.utils import load_config
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning import Trainer
    

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    


    # Prepare Dataset 
    config = "E:\\YouTube\\Git\ldm\\config.yaml"
    # config = "/content/sd/ldm/config.yaml"
    config = load_config(config_path=config)
    # root_dir = "/content/sd/dataset/cat_dog_images"
    root_dir = "E:\\YouTube\\Git\\dataset\\cat_dog_images"
    # print("COnfig: ", config)
    

    train_dataset = ImageDataset(
                                root_dir=root_dir,
                                split="train",
                                image_size=256,
                                )
    

    val_dataset = ImageDataset(
                                root_dir=root_dir,
                                split="val",
                                image_size=256
                                )
    

    
    
    train_datloader = DataLoader(
                                dataset=train_dataset,
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=2,
                                persistent_workers=True,
                                prefetch_factor=2
                                )
    

    
    val_datloader = DataLoader(
                                dataset=val_dataset,
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=2,
                                persistent_workers=True,
                                prefetch_factor=2
                                )





    config = config['model']['params']
    # Initialize model 
    model = LatentDiffusion(**config)
    model = model.to("cuda")
    print(model)
    



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
        precision="16-mixed",  # Can use '16-mixed' for FP16
        accumulate_grad_batches=1,
        # pin_memory=True,
        # num_workers=4,
        # persistent_workers=True
        # gradient_clip_val=0.5,  # Reduced from 1.0
        # accumulate_grad_batches=4,  # Increased accumulation
        # enable_progress_bar=True,
        enable_model_summary=False,
        limit_train_batches=0.1,  # Train on 10% of data initially
        limit_val_batches=0.1,    # Validate on 10% of data initially
        deterministic=True,       # For reproducibility
        # amp_backend='native',
        strategy='ddp_find_unused_parameters_false'  # More efficient distributed training
    )




    # # Add memory management at the start of training
    # torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    # torch.backends.cuda.enable_flash_sdp(True)  # Enable FlashAttention if available

    # # Training 
    # trainer.fit(model, train_datloader, val_datloader)
    # print("Training completed!")




    # Add memory monitoring
    def print_memory():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    print("Memory before training:")
    print_memory()
    


    try:
        trainer.fit(model, train_datloader, val_datloader)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("Out of memory error caught! Trying recovery...")
            torch.cuda.empty_cache()
            gc.collect()
            # Reduce memory usage and try again
            model.half()
            trainer.fit(model, train_datloader, val_datloader)
        else:
            raise


    
    
    print("Training completed!")
    print("Final memory stats:")
    print_memory()
    