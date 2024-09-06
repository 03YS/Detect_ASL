# The below gives a detailed explanation on how to run the project 
# Create a virtual environment; Activate the virtual environment

	.venv\Scripts\activate 

# Install the following packages

	pip install torch torchvision torchaudio

	pip install albumentations>=0.4.3

	pip install scikit-learn>=0.22.1

	pip install pandas

	pip install imutils

	pip install tqdm

	pip install matplotlib

# Execute the following commands after installing above

	pip uninstall opencv-python opencv-python-headless

	pip install opencv-python

# Change to src folder

	cd src
 
# Run the following commands one by one in the SAME order

	python preprocess_image.py --num-images 1200

	python create_csv.py

	python train.py --epochs 10

	python test.py --img A_test.jpg

	python cam_test.py
