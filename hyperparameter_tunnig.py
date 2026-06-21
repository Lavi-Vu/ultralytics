from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="edgerf-n.yaml", help="Model config or weights path")
parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset path")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
opt = parser.parse_args()

model = YOLO(opt.model)  # or "edgerf-n.pt"
results = model.tune(
    data=opt.data,
    epochs=opt.epochs,
    iterations=opt.iterations,         # or num_samples for Ray
    optimizer="auto",       # or MuSGD
    use_ray=False,          # set True for advanced
    plots=True,
    save=True
)