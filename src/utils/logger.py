import logging
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, config):
        self.exp_name = config["experiment"]["name"]
        self.log_dir = os.path.join("experiments", "logs")
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{self.exp_name}_{timestamp}.log")

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(self.exp_name)
        self.logger.info(f"Started logging to {log_file}")

    def log_training(self, task_name, epoch, loss, metrics=None):
        """Log training information"""
        log_msg = f"Task: {task_name}, Epoch: {epoch}, Loss: {loss:.4f}"
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_msg += f", {metrics_str}"
        self.logger.info(log_msg)

    def log_evaluation(self, task_name, metrics):
        """Log evaluation results"""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Evaluation - Task: {task_name}, {metrics_str}")

    def log_memory_buffer(self, task_name, num_examples):
        """Log memory buffer updates"""
        self.logger.info(f"Memory Buffer - Added {num_examples} examples from task: {task_name}") 