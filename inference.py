import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from ultralytics import YOLO

if __name__ != "__main__":
    CATEGORY_NAMES = [
        "__background__",  # index 0 is always background
        "pod_rot",         # index 1 in the model outputs
        "healthy",         # index 2
        "pod borer"        # index 4
    ]
    # =====================
    # 1. Settings
    # =====================
    image_chosen = random.randint(40,400)
    print(image_chosen)
    MODEL_PATH = r"C:\Programming\checkpoints\Stuff\model_epoch_15.pth"      # path to your trained weights
    IMAGE_PATH = fr"C:\Users\Erron Philip Manatad\Downloads\cacao_diseases\cacao_photos\healthy\healthy_{image_chosen}.jpg"  # image to run inference on
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4  # adjust to match your training (includes background)

    def get_model(num_classes):
        # Create a ResNet-101 backbone with FPN
        backbone = resnet_fpn_backbone('resnet101', weights='DEFAULT')
        model = MaskRCNN(backbone, num_classes=num_classes)
        return model

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()


    # =====================
    # 3. Load image
    # =====================
    image = Image.open(IMAGE_PATH).convert("RGB")
    image_tensor = F.to_tensor(image).to(DEVICE)

    # =====================
    # 4. Run inference
    # =====================
    with torch.no_grad():
        output = model([image_tensor])[0]

    # =====================
    # 5. Filter results
    # =====================
    score_threshold = 0.6
    keep = output["scores"] >= score_threshold

    boxes = output["boxes"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    scores = output["scores"][keep].cpu().numpy()
    masks = output["masks"][keep].cpu().numpy().squeeze(1)  # [N, H, W]

    # =====================
    # 6. Visualization
    # =====================
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, mask, label, score in zip(boxes, masks, labels, scores):
        category_name = CATEGORY_NAMES[label]
        x1, y1, x2, y2 = box

        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                fill=False, edgecolor='red', linewidth=2))
        ax.text(x1, y1, f"{category_name}: {score:.2f}", fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5))

        # Overlay mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0.5] = [1, 0, 0, 0.5]
        ax.imshow(colored_mask)

    plt.axis("off")
    plt.show()
else:
    model = YOLO(r"C:\Programming\runs\segment\train7\weights\best.pt")
    source = r"C:\Users\Erron Philip Manatad\Downloads\cacao_diseases\cacao_photos\pod_borer\pod_borer_66.jpg"
    model.predict(source,
                  save=True, 
                  visualize=True,
                  conf=0.3)
