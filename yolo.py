# Step 1: Set up Google Colab
from google.colab import drive
drive.mount(r'C:\Users\Hp\Downloads')



# Step 2: Prepare the Dataset
# Make sure you upload your dataset to Google Drive and organize it properly

# Create your dataset YAML file
dataset_yaml = """
train: /content/drive/MyDrive/your_dataset/train/images
val: /content/drive/MyDrive/your_dataset/val/images
nc: 1  # number of classes
names: ['your_class_name']
"""

with open('/content/drive/MyDrive/your_dataset.yaml', 'w') as f:
    f.write(dataset_yaml)

# Step 3: Train the Model
python train.py --img 640 --batch 16 --epochs 30 --data /content/drive/MyDrive/your_dataset.yaml --weights yolov5s.pt

# Step 4: Test the Model
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source /content/drive/MyDrive/your_dataset/test/images

# Step 5: Plot Results
from utils.plots import plot_results
plot_results(save_dir='runs/train/exp')

from IPython.display import Image, display
import glob

# Display inference results
for img_path in glob.glob('runs/detect/exp/*'):
    display(Image(filename=img_path))
