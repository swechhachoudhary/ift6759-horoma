# Humanware project - Block 1
The first part of the humanware project consisted of predicting the amount of digits seen in a
54 x 54 image. The amount of digits found in the pictures was between 1 and 5, for a total of
5 possible quantities. The original dataset is first pre-processed through image transformations,
performed as to induce data augmentation during training. The network architecture used for 
this task is DenseNet, a convolutional neural network (CNN). 

### Required packages
This projects includes some packages not found in the standard bundle:
- Numpy
- Torch
- Torchvision
- Tdqm
- PIL
- Matplotlib

Other packages imported are generally included in the standard python interpreter 

### Folder Structure
```
COURS2019/etudiants/submissions/
└── b1phut3
    ├── code
    │   └── .git/
    │   └── evaluation
    │   │   └── eval.py
    │   └── experiments
    │   │   └── base_model
    │   │       └── params.json
    │   └── notebooks
    │   │   └── eda.ipynb
    │   └── processing
    │   │   └── loader.py
    │   │   └── transforms.py
    │   └── utils
    │   │   └── classes.py
    │   │   └── functions.py
    │   └── README.md
    │   └── run.pbs
    │   └── train.py
    ├── model
    │   └── best_model.pth
    └── report.pdf

```

### File Implementation
The repository consists of several files, organized in folders depending on their function. 
The **experiment** folder contains folders for different models. The file `params.json` contains the hyperparameters
related to the model architecture and the training optimizer. This .json file has a default configuration; network-specific
hyperparameters need to be added.

In the **processing** folder can be found *loader.py*, *transforms.py*:

- *loader.py* : This file contains the _SVHNDataset_ class and the _fetch_dataloader_ function.
The dataset (hosted on the Helios server) is fetched by the _SVHNDataset_ class, and incorporated in 
a custom PyTorch _Dataset_ object. The preprocessing transformations are incorporated to the datasets.
 _fetch_dataloader_ function creates such datasets with _SVHNDataset_ and instantiates PyTorch _DataLoaders_
 from those datasets.
 
- *transforms.py* : Transformations part of the preprocessing of the original dataset.

The **model** folder contains the different network architectures tested and utilities.

- *architecture.py* : Contains declaration of the models used in this project

- *metrics.py* : Various quantifying metrics for the classification accuracy

The **notebooks** folder contains the data analysis notebook `eda.ipynb`, used for statistics and data visualization.

The **evaluation** folder contains the evaluation file, `eval.py`, used for testing the code at the end of the block.

The **utils** folder contains various class and function declaration used to improve the code structure and
functionalities.

- *classes.py* : Contains project-related class declaration.

- *functions.py* : Contains various utilities function such as the logging function and the save/load checkpoint functions.

The remaining files located directly in **humanware** folder are directly related do the training and
evaluation process

- *run.pbs* : PBS file used in the Helios server to run *train.py*.

- *train.py* : Contains training-related functions and the essential *main()* function to run.


## Functions description:

-  *eval.py*:
    
    - def eval_model(dataset_dir, metadata_filename, model_filename): 
    
        This function use the model information and the dataset to return the y_predicted.
      
-  *architecture.py*:
    
    - class HumanwareDenseNet(DenseNet):
      def __init__(self, params): 
      
        This method is used to let the model DenseNet available to be used. 
    
    - class HumanwareResNet(ResNet):
      def __init__(self, params):
      
        This method is used to let the model ResNet available to be used. 

-  *metrics.py*:

    - class Metric(object):
      def reset(self):
        
        Initialize all the values to 0.
    
      def dump(self):
    
        It's used to save the attributes information on a dictionary. 
    
    - class LossAverage(Metric):
      def __init__(self):

        Initialize an object Metric with the parameters steps and total

      def __call__(self, output, labels, params):
      
        Abstract method defined by AccuracyPerClass

      def __str__(self):
        
        It calculates and returns the average loss that is used while the model is training.

      def update(self, val):
        
        Increase the steps and the total variables. Update and return the average of losses.
    
      def get_average(self):
      
        Calcule and return the average.

    - class AccuracyPerClass(Metric):
      def __init__(self):
        Initialize the attributes correct and total

      def __call__(self, outputs, labels, params):
        
        Update correct and total samples using the outputs of the model and the labels.

      def __str__(self):
    
        Calculate and return the accuracy 

      def reset(self):

        Initialize correct and total values. 

      def dump(self):
        
        It's used to dump the attributes information on a dictionary. The values saved are accuracy, correct and 
        total values.

-  *loader.py*:

   - class SVHNDataset(Dataset):
    Street View House Numbers dataset class.
    
    def __init__(self, pickle_path, image_dir, transform=None, normalize=True):
        
        Used to instantiate an object with the dataset information (filename and metadata, location of the images, 
        transform and normalize information)
    
    def __len__(self):
    
        Returns the lenght of the dataset.

    def __getitem__(self, idx):
        
    def label_modifier(metadata):
    
        Static method that use the dictionary and returns the number of digits of the image. 5 or more digits are 
        grouped in one class.

    def fetch_dataloader(type, params, pickle_path, image_dir):

        Return the dictionary containing the dataloader. 


-  *transforms.py*:
   - class ComposeMultipleInputs(object):
     def __init__(self, transforms):
        
        Initialization with the information of the transformations to be applied.
        
     def __call__(self, image, metadata):
        
        Using the image to be transformed and the dictionary, it returns the image with all the transformations applied.
    
   - class BoundingBoxCrop(object):      

     def __init__(self, margin=0.3):
        
        Initialization with the value of the margin to add to the smallest box that encloses all bounding boxes.

     def __call__(self, image, metadata):
        
        Get the image and dictionary and returns the image cropped with the additional margin and the metadata 
        dictionary updated.

     def scale_bounding_box(self, bounding_box, image):
    
        Given a bounding box (x,y,width and height), it scales the bounding box in x and y direction by the specified 
        margin. It returns x,y,width and height scaled.

     def get_bounding_box(metadata):
        
        Get the smallest box that encloses all the annotated bounding boxes. It returns x,y,width and height.

     def crop(image, x, y, width, height):
        
        Use the image and its dimension information, this function returns the cropped image.  

     def update_annotations(metadata, x, y)
        
        Update the annotation dictionary with the values x and y with the image cropped.

   - class Rescale(object):
    
    def __init__(self, output_size):
    
        Initialize the information with the output_size (width and height) 

    def __call__(self, image, metadata):
        
        Return the image with all the transformation applied and update the dictionary. It rescale the image in to a 
        given size and update annotations.

    def update_annotations(metadata, width_ratio, height_ratio):
        
        Update the annotation dictionary with the ratios of change between widths and heights. 
        
  
  - class RandomCrop(object):

    def __init__(self, size=(54, 54)):
        
        Initialize the object with values of width and height.

    def __call__(self, image, metadata):
    
        Return the image with all the transformations applied. It crops a random part smaller than the image.

    def update_annotations(self, metadata, x_crop, y_crop):
        
        Given x and y position of crop. Update the annotation dictionary according to the random crop done.

    def get_params(image, output_size):
        
        Given the expected output_size of the crop, it gets the parameters for cropping for random crop process. 
        It returns x,y, and crop_width and crop_height.

    def crop(image, x, y, width, height):
        
        Crop an image using the top position (x, y), width and height. It returns the cropped image using 
        specified dimensions.

  - class ToTensor(object):

    def __call__(self, image, metadata):
        
        Return the given image (PIL image or ndarray) as a tensor. 
  
-  *classes.py*:

    def __init__(self, json_path):
    
        Initialize the dictionary with the parameters contained in json file.

    def save(self, json_path):
        
        Saves parameters to json file
        
    def update(self, json_path):
    
        Loads parameters from json file

    def dict(self):
    
        Gives dict-like access to Params instance by params.dict['learning_rate']
    

-  *functions.py*:

    def set_logger(log_path):
    
        Set the logger to log info in terminal and file log_path.

    def save_dict_to_json(d, json_path):
    
        Saves dictionary content into the json file. 

    def save_checkpoint(model, state, is_best, checkpoint):
    
        Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, 
        also saves checkpoint + 'best.pth.tar'

    def load_checkpoint(checkpoint, model, params, optimizer=None, scheduler=None):
    
        Given the model parameters, it returns the checkpoint (dictionary with saved states). Loads model parameters 
        (state_dict) from file_path. If optimizer is provided, loads state_dict of optimizer assuming it is present 
        in checkpoint.

- *train.py*:

    def train(model, optimizer, loss_fn, dataloader, metrics, params):

        Train the model on all batches in one epoch. It uses the information of the model, dataloader, 
        parameters (hyperparameters), metrics (dictionary), loss function to be optimized and optimizer of the 
        model parameters.

    def validate(model, loss_fn, dataloader, metrics, params):
        Train the model on all validation data. It returns the dictionary with the results for validation split.

    def train_and_validate(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, metrics, params,
                       model_dir, restore_file=None):
    
        Train the model and validate every epoch. It saves latest val metrics in a json file in the model directory. 


## Data flow

Dataset is fetched with `fetch_dataloader` function. It returns a DataLoader incorporating the 
preprocessing transformations. The function returns both the train dataloader and the validation dataloader. 
Meanwhile, a `DenseNet` model is instantiated,
the choice of hyperparameters depending on the user in the file `params.json`.
With the dataloader and the model, the `train_and_validate`
function can be called. This function implements both the learning and validation process.

## Deployment
If you're using PyCharm Professional, it is possible to deploy this project directly to the Helios server. 
In Setting - Build, Execution, Deployment - Deployment, choose the SFTP server option. Enter the host name of the cluster
with your username (userXX@helios.calculquebec.ca), then add username and password. In the *Mappings* section of Deployment, 
write the deployment target in Deployment Path (path to your folder on Helios server).

When running `run.pbs` on the Helios server, the `train.py` parser arguments can be added on the command line. Example:
`msub -v model='x', model_path='experiments/resnet50/' run.pbs`. Here, this allows the user to quickly change the model
trained on and the set of parameters used from one command line to the next.

## Built With
JetBrains PyCharm 2018.3.3, Professional Edition

## Authors
Ramón Emiliani - ramon.emiliani@umontreal.ca
Alejandra Jimenez Nieto - alejjimn@gmail.com
Jean-Philippe Letendre - jp.letendre1@gmail.com