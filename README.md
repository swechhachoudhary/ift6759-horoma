# IFT6759 Winter 2019
## Horoma Project Block 3

Authors:
Swechha Swechha
Timothy Nest
Benjamin Rosa

Project Organization
------------

    ├── logs	          <- log files saved when executing a runner
    ├── configs           <- all the configs files
    ├── models            <- all the models architecture
    ├── notebooks         <- notebooks used mainly for quick testing
    ├── pbs		          <- all the pbs files used for running different runner on helios
    ├── runner		      <- all the python files having a main in it (and who are executed with a pbs file)
    ├── template          <- template for generating pbs files
    ├── trainer           <- each trainer for each models
    ├── utils             <- utilities used across the application
    └── README.md         <- The top-level README for developers using this project.

--------

Third block of the Horoma project

```
usage: ./runner/train.py [-h] [-c CONFIG | -r RESUME | --test-run] [-d DEVICE]
                [--helios-run HELIOS_RUN]

AutoEncoder Training

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  --test-run            execute a test run on MNIST
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --helios-run HELIOS_RUN
                        if the train is run on helios with the run_experiment
                        script,the value should be the time at which the task
                        was submitted
```

```
usage: hyperparameter_search.py [-h] [-c CONFIG] [-r RESUME] [--test-run]
                                [-d DEVICE] [--helios-run HELIOS_RUN]

AutoEncoder Training

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file path (default: None)
  -r RESUME, --resume RESUME
                        path to latest checkpoint (default: None)
  --test-run            execute a test run on MNIST
  -d DEVICE, --device DEVICE
                        indices of GPUs to enable (default: all)
  --helios-run HELIOS_RUN
                        if the train is run on helios with the run_experiment
                        script,the value should be the time at which the task
                        was submitted
```

## Running an experiment

In order to run an experiment, run the script `run_experiment.sh` while specifying the configuration file to use. This will create the PBS configuration script, initialise the result folder and launch the experiment.

`sh run_experiment.sh configs/model_config.in`

## Github conventions
* Each feature must have his own branch for development
  * git checkout -b nameOfTheNewBranch
* When changes are made, push is only made to the feature's branch
  * git add .
  * git commit -m "Relevant message regarding the changes"
  * git checkout master
  * git pull --rebase
  * git checkout nameOfTheNewBranch
  * git merge master
  * Fix conflicts if there is any
  * git push origin nameOfTheNewBranch
* Once the changes in a personal branch are ready, do a pull request for the master branch
  * go to the github page of the project https://github.com/swechhachoudhary/ift6759-horoma
  * select your branch using the drop down button
  * click on create pull request
  * put master branch as the head
  * confirm the creation of the pull request
  
## Configuration file

### Configuration file guide
The default configuration file is ./configs/template.in
This default configuration file should not be modify by the users.
It is only here to provide a template of what the program is expected to have.
Everyone should his own config file in the folder configs.
When launching the program, add the --config arguments in the command line to specify the path to your config file.

If someone change the config file, it is important to update template.in accordingly.
That way everyone would be able to do the changes in their own config files.

### Configuration file parameters
The configuration file is an input file containing several parameters :
* general.use_gpu             : boolean - whether or not we use the GPU for training/validation/testing


## Data Files
### Input Data Format
Location : `/rap/jvb-000-aa/COURS2019/etudiants/data/horoma`
  * train_x.dat: 152,000 x 32 x 32 x 3.
  * train_labeled_x.dat: 228 x 32 x 32 x 3 (used to label clusters).
  * valid_x.dat: 252 x 32 x 32 x 3 (used to evaluate your models).

We also have access to files containing overlapped patches to increase the size of your
datasets (pixel patches with ~50% overlap):
  * train_overlapped_x.dat: 548,720 x 32 x 32 x 3.
  * train_labeled_overlapped_x.dat: 635 x 32 x 32 x 3.
  * valid_overlapped_x.dat: 696 x 32 x 32 x 3.

###  Output(Label) Data Format
Outputs are provided as text files (can be easily read from a terminal).
  * train_labeled_y.txt: contains 228 tree species (2 characters).
  * valid_y.txt: contains 252 tree species (2 characters).
  * train_labeled_overlapped_y.txt: contains 635 tree species (2 characters).
  * valid_overlapped_y.txt: contains 696 tree species (2 characters).
  * The i-th value in filename_y.txt is associated to the i-th pixel patch in
filename_x.dat.

### Data region ids
To split labeled datasets efficiently, we have access to files
containing ids representing the pixel subregion where each pixel patch has
been extracted from images:
  * train_regions_id.txt.
  * train_overlapped_regions_id.txt.
  * train_labeled_regions_id.txt.
  * train_labeled_overlapped_regions_id.txt.
  * valid_regions_id.txt.
  * valid_overlapped_regions_id.txt. 

