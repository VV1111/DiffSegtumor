
# Preprocessing: 
## Step 1: Split the dataset. preprocess.py

Use function ```preprocessor.split_dataset()``` to split the dataset into training, validation, and test sets.

Example command:

```python
python -m data.preprocess
```

⚡ Note: The split results have already been generated and are available in data_prep/split_txt/.

## Step 2: Crop and select slices.

1️⃣ crop and save: preprocess.py
```python
preprocessor.process_split("train")
```
crops the data based on the split results and saves each case as separate .nii.gz files.

2️⃣ Merge Slices: ```merge.py```
merges the cropped .nii.gz files from individual cases into a single .npy or .nii.gz file.

⚡ Note: the merged training set can only be saved as .npy (saving as .nii.gz is not feasible).

## Step 4: Data Loading Example: dataloader.py
A basic example of how to read and load the processed data is provided in dataloader.py.


## Configuration: util/config.py
All relevant parameter settings are located in util/config.py.

You can adjust paths such as:
```
original_dir = "/path/to/your/data"
```

#  Training & Testing (Diffusion with U-shape model)

## training & evaluating
```
bash train.sh -c 0 -e diffusion -t brats_2d -i '' -l 1e-2 -w 10 -n 300 -d true 
```
-c: use which gpu to train

-e: use which training script, can be diffusion for train_diffusion_2d.

-t: switch to different tasks

-i: name of current experiment, can be whatever you like

-l: learning rate

-w: weight of unsupervised loss

-n: max epochs

-d: whether to train,

## testing
```
python code/inference.py
```
