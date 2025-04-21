from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import trainer

@rank_zero_only
def log_something(trainer, message):
    print(f"[{trainer.global_rank}] Logging: {message}")

if __name__ == "__main__":
    # In your training loop:
    log_something(trainer, "Training started") # Only rank 0 will print this message
