from collections import defaultdict
import random
import torch
import logging

class MemoryBuffer:
    def __init__(self, buffer_size, selection_strategy="random"):
        self.buffer_size = buffer_size
        self.selection_strategy = selection_strategy
        self.buffer = defaultdict(list)
        self.current_size = 0
        self.logger = logging.getLogger("memory_buffer")
    
    def add_examples(self, task_name, examples, rehearsal_rate):
        """Add examples to the memory buffer for a specific task."""
        num_examples = int(len(examples) * rehearsal_rate)
        if num_examples == 0:
            self.logger.info(f"No examples added for task {task_name} (rate: {rehearsal_rate})")
            return
        
        if self.selection_strategy == "random":
            selected_examples = random.sample(examples, num_examples)
        else:
            # Implement other selection strategies (e.g., uncertainty-based)
            raise NotImplementedError
        
        # Manage buffer size
        available_space = self.buffer_size - self.current_size
        if available_space < num_examples:
            self.logger.info(f"Freeing up space: {num_examples - available_space} examples")
            self._free_space(num_examples - available_space)
        
        self.buffer[task_name].extend(selected_examples)
        self.current_size += len(selected_examples)
        self.logger.info(f"Added {len(selected_examples)} examples from task {task_name}. Buffer size: {self.current_size}") 