from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import rootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

root = rootutils.setup_root(__file__, pythonpath=True)
from src.models.dogbreed_classifier import DogBreedGenericClassifier
from src.utils.utils import get_rich_progress, setup_logger, task_wrapper


@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img, transform(img).unsqueeze(0)


@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_labels = [
        "Beagle",
        "Boxer",
        "Bulldog",
        "Dachshund",
        "German_Shepherd",
        "Golder_Retriever",
        "Labrador_Retriever",
        "Poodle",
        "Rottweiler",
        "Yorkshire_Terrier",
    ]

    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence


@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@task_wrapper
@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    model = DogBreedGenericClassifier.load_from_checkpoint(cfg.ckpt_path)
    model.eval()

    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    log_dir = Path(cfg.paths.log_dir)
    setup_logger(log_dir / "infer_log.log")

    image_files = list(input_folder.glob("*"))
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))

        for image_file in image_files:
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device))

                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)

                progress.console.print(
                    f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})"
                )
                progress.advance(task)


if __name__ == "__main__":
    main()
