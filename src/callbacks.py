import pytorch_lightning as pl

class ModelCheckpoint(pl.Callback):
    def __init__(self, monitor='val_loss', save_top_k=1, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        # Logic to save the model checkpoint based on validation metrics
        pass

class EarlyStopping(pl.Callback):
    def __init__(self, monitor='val_loss', patience=3, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.wait = 0
        self.best_score = None

    def on_validation_end(self, trainer, pl_module):
        # Logic to stop training early based on validation metrics
        pass

class LoggingCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Logic to log metrics at the end of each epoch
        pass