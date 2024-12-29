import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup
import logging
from src.utils.logger import ExperimentLogger

class ContinualTrainer:
    def __init__(self, config, memory_buffer):
        self.config = config
        self.memory_buffer = memory_buffer
        self.device = torch.device(config["experiment"]["device"])
        self.current_task = None
        self.metrics_history = {}
        self.logger = ExperimentLogger(config)
    
    def train_task(self, model, task_name, task_dataset, rehearsal_rate):
        """Train model on a single task with rehearsal."""
        self.current_task = task_name
        self.logger.log_training(task_name, -1, 0.0, {"status": "starting"})
        
        # Get rehearsal data
        rehearsal_data = self.memory_buffer.get_rehearsal_data(task_name)
        combined_dataset = self._combine_datasets(task_dataset, rehearsal_data)
        
        # Create data loader
        train_loader = DataLoader(
            combined_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"]
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * self.config["training"]["epochs_per_task"]
        )
        
        # Training loop
        for epoch in range(self.config["training"]["epochs_per_task"]):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                loss = self._training_step(model, batch, optimizer, scheduler)
                total_loss += loss
            
            avg_loss = total_loss / len(train_loader)
            self.logger.log_training(task_name, epoch, avg_loss)
        
        # Store examples in memory buffer
        num_examples = int(len(task_dataset) * rehearsal_rate)
        self.memory_buffer.add_examples(task_name, task_dataset, rehearsal_rate)
        self.logger.log_memory_buffer(task_name, num_examples)
    
    def _training_step(self, model, batch, optimizer, scheduler):
        """Perform one training step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        
        # Gradient accumulation
        loss = loss / self.config["training"]["gradient_accumulation_steps"]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.config["training"]["max_grad_norm"]
        )
        
        optimizer.step()
        scheduler.step()
        
        return loss.item()
    
    def _combine_datasets(self, task_dataset, rehearsal_data):
        """Combine current task dataset with rehearsal data."""
        if not rehearsal_data:
            return task_dataset
        return ConcatDataset([task_dataset, rehearsal_data]) 