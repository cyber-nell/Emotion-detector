import torch
import json
import timm
from torchvision import transforms
from PIL import Image
import gradio as gr


# Open class_mapping.json and load the dictionary
with open("emotion_class_mapping.json", "r") as f:
    id_to_label = json.load(f)

# Convert to an ordered list matching indices 0..37
CLASS_NAMES = [id_to_label[str(i)] for i in range(len(id_to_label))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Instantiate the ResNet34 structure
model = timm.create_model('resnet34', pretrained=False, num_classes=len(CLASS_NAMES))

#Load the trained weights
model.load_state_dict(torch.load("best_emotion_model.pth", map_location=device))
model.to(device)
model.eval()    #Freeze weights for inference

#Ensure images is standadises the same way in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return {}
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert('RGB')
    
    # Preprocess and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Return a dictionary mapping {Label: Probability}
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return confidences

# Define input (Image with upload + webcam support) and output (Label showing top predictions)
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["upload", "webcam"], type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Emotion Detector",
    description="Upload a photo or use your webcam to detect facial emotions!"
)

if __name__ == "__main__":
    interface.launch()