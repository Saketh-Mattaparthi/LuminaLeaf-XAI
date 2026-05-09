import os
import json
from datasets import load_dataset
from PIL import Image

def main():
    print("🌿 Downloading your Hugging Face Dataset (mohanwithdata/Medical_Plants_image)...")
    
    # Load the Hugging Face dataset
    try:
        ds = load_dataset("mohanwithdata/Medical_Plants_image")
        print("✅ Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Extract class names to create labels.json
    class_names = ds['train'].features['label'].names
    labels_dict = {str(i): name for i, name in enumerate(class_names)}
    
    with open('labels.json', 'w') as f:
        json.dump(labels_dict, f, indent=4)
    print("✅ Created labels.json mapped perfectly to your 30 plants!")

    output_dir = "data"
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)

    def save_split(split_name, folder_name):
        print(f"Saving {split_name} images...")
        dataset_split = ds[split_name]
        for idx, item in enumerate(dataset_split):
            image = item['image']
            label_idx = item['label']
            label_name = class_names[label_idx]
            
            # Create directory for the class
            class_dir = os.path.join(output_dir, folder_name, label_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save the PIL image
            file_path = os.path.join(class_dir, f"{idx}.jpg")
            # If the image has an alpha channel, convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(file_path)

    # Save training and validation data into directories
    # This prepares the data so `train_model.py` can use tf.keras.utils.image_dataset_from_directory perfectly
    save_split('train', 'train')
    save_split('validation', 'validation')

    print("\n🎉 Dataset setup complete!")
    print("Images are now properly organized in 'data/train' and 'data/validation'.")
    print("You can now begin training by running: python train_model.py")

if __name__ == "__main__":
    main()
