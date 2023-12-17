import os

# Define the directories containing your train and test images
train_dir = '/path/to/train'
test_dir = '/path/to/test'

# Define the paths to the output train.txt and test.txt files
train_txt_path = '/path/to/train.txt'
test_txt_path = '/path/to/test.txt'

# Function to list image file paths in a directory and its subdirectories
def list_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Create train.txt and write the train image paths
train_image_paths = list_image_paths(train_dir)
with open(train_txt_path, 'w') as train_txt_file:
    for image_path in train_image_paths:
        train_txt_file.write(image_path + '\n')

# Create test.txt and write the test image paths
test_image_paths = list_image_paths(test_dir)
with open(test_txt_path, 'w') as test_txt_file:
    for image_path in test_image_paths:
        test_txt_file.write(image_path + '\n')

print(f"Train.txt file created with {len(train_image_paths)} image paths.")
print(f"Test.txt file created with {len(test_image_paths)} image paths.")
