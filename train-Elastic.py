# train_elastic.py
import torch
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.Elastic import ElasticBlock  # Assumes you saved it here

# 1. Register the custom ElasticBlock module with the tasks engine namespace
setattr(tasks, 'ElasticBlock', ElasticBlock)

# 2. Define the ElasticResourceTrainer right here
class ElasticResourceTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        # 1. Safely intercept 'overrides' whether it is a positional argument or inside kwargs
        overrides = kwargs.get("overrides", None)
        if overrides is None and len(args) > 0:
            overrides = args[0]
            
        # 2. Ensure overrides is a dictionary so parent class methods like .pop() don't crash
        if overrides is None:
            kwargs["overrides"] = {}
        elif not isinstance(overrides, dict):
            # If it's passed as a string configuration path, let it pass through
            pass

        # 3. Pass everything directly up using clean variable unpacking (*args, **kwargs)
        # This allows Ultralytics to map arguments exactly how its internal version expects
        super().__init__(*args, **kwargs)
        
        # 4. Initialize your custom reinforcement learning metadata
        self.beta = 0.15          # Elastic penalty multiplier
        self.target_latency = 0.5 # Maximum target ratio of blocks allowed to run

    def compute_loss(self, preds, batch):
        # Compute default Ultralytics box/class losses
        loss, loss_items = super().compute_loss(preds, batch)
        
        # Traverse model components to collect hidden routing penalties
        resource_penalties = []
        for module in self.model.modules():
            if isinstance(module, ElasticBlock) and getattr(module, 'current_action_probs', None) is not None:
                route_probabilities = module.current_action_probs[:, 1]
                mean_layer_usage = torch.mean(route_probabilities)
                
                # Apply squared hinge penalty if usage exceeds target budget bounds
                penalty = torch.clamp(mean_layer_usage - self.target_latency, min=0.0) ** 2
                resource_penalties.append(penalty)
                
        if resource_penalties:
            total_penalty = sum(resource_penalties) / len(resource_penalties)
            loss += self.beta * total_penalty
            
        return loss, loss_items

# 3. Main execution scope
if __name__ == "__main__":
    # Point directly to your custom elastic layout yaml
    model = YOLO("elactic_yolo.yaml") 
    
    # Pass the trainer class into the model trainer parameter
    model.train(
        data="ultralytics/cfg/datasets/coco.yaml", 
        epochs=50, 
        imgsz=640, 
        trainer=ElasticResourceTrainer  # <--- Injected here
    )