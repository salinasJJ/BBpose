# Glossary

## Main

* **version**: The version number assigned to the current model being trained on. \
Note: CHECK VERSION NUMBER! Users should get into the habit of checking which version number is being set in order to avoid overwriting old models (unless that is the intended results).

* **img_dir**: Absolute path to where image files are stored to. \
Note: do not set path to within the project's 'data' or 'results' directories.

* **dataset**: Image set and accompanying data files which the user wishes to download. \
Supported datasets currently only include 'mpii'.

* **use_records**: Will switch the model over to use TFRecords. \
Note: the TFRecords dataset will be much larger than its Tensorflow Dataset counterpart but will lead to faster training times. If operational cost is more important than memory cost, set this param to True.
Note: unless the user has a reason to keep the videos after creating the TFRecords, it is recommended to set delete_videos to True in order to avoid massive use of memory space.

* **use_cloud**: Set to True if user wished to retrieve data from a remote location.
Note: the code has been set up to work with Google Cloud Storage and most likely will not work with any other service.

* **batch_per_replica**: Total number of samples passed to each replica. \
Do not set this number to a global value since this value will be passed to each device available.

* **img_size**: Spatial length that each image should be resized/cropped to. \
Supported values include: 128, 192, 256, 320, 384, 448, and 512. \
Note: the value of img_size must be 4 times larger than hm_size.

* **hm_size**: Spatial length of each heatmap. \
Supported value include: 32, 48, 64, 80, 96, 112, and 128. \
Note: the value of hm_size must be 4 times smaller than img_size.

* **num_stacks**: Total number of hourglass stacks to use. \
Supported value include: 2, 4, and 8.

## Ingestion

* **download_images**: Set to False in order to bypass downloading of the images. This may be desired if the user already contains a copy of the image dataset. \
Note: do not forget to set the img_dir param to the location of your copy.

* **toy_set**: Set to True if user wishes to work a smaller subset to experiment with. \
Note: this is for experimentation only, as any results gathered from this toy set will not be accurate.

* **toy_samples**: Total number of images to subset for the toy dataset.

* **examples_per_record**: Total number of examples to be placed in each training TFRecord. \

* **interleave_cycle**: Total number of records to simultaneously interleave. \
As a visual example: for a simplest case, allow your hands to represent two records and your fingers to represent examples from each each record. If you now interlock your hands together, this will represent the interleaving performed by this operation. \
This option is necessary for TFRecords since you cannot shuffle binary data. \
For further information, check the [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave) documentation on this operation.

* **interleave_block**: Total number of consecutive elements that are interwoven from each record at one time. \
This option just allows for blocks of examples from the same record to be placed next to one another when interwoven with examples (also in a block) from another record.
For further information, check the [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave) documentation on this operation.

* **gcs_records**: Remote path where TFRecords are stored.
Note: the code has been set up to work with Google Cloud Storage and most likely will not work with any other service.
Note: The directory defined by this path should contain the 'train' and 'val' directories. But they should not be included in the param's path name.

## Preprocess

* **mean**: Mean values calculated (from train dataset) for each RGB channel.

* **sigma**: STD used for Gaussian kernel in creation of the heatmaps.

## Models

* **arch**: Model architecture to be used. \
Supported architectures include: 'softgate' and 'hourglass'

* **num_filters**: The base number of filters/channels used at each layer. All layers will either hold this number of filters or be a factor of this value.
Supported values include: any multiple of 8.

* **initializer**: Technique to initialize the model weights. \
Recommended initializers include: 'glorot_normal', 'glorot_uniform', 'he_normal', and 'he_uniform'. \
If the user wishes to use a different initializer not in this list, please check with the [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/initializers) documentation first to determine if your requested initializer is supported. If not, you may receive an error at runtime.

* **momentum**: Defines the momentum for the moving average used in each batch norm layer.

* **epsilon**: Value added to each batch norm's variance to avoid division by zero.

* **dropout_rate**: The percentage of units to drop at a specific layer. \
Recommended value is 0.2 (if used). Otherwise, default value of 0 will effectively lead to no dropout used.

## Train / Test

* **is_eager**: Set to True if user wishes to run tf.functions in eager mode. This can be set to True when debugging the code, but should be set to False when actually training the model.

* **strategy**: Tensorflow Strategy to use during training. \
Supported strategies include: 'default', 'mirrored', and 'tpu'. \
Use default when using only a single GPU, use mirrored when using multiple GPUs, and use tpu when using one or multiple TPUs.

* **tpu_address**:  Set this param either to the TPU's name or the TPU's gRPC address.

* **gcs_results**: Remote path where SaveModels are stored. \
Note: the code has been set up to work with Google Cloud Storage and most likely will not work with any other service. \
Note: do not include the version number at the end of the path name.

* **mixed_precision**: Set to True if user wishes to use a mix of 16 and 32 bit floating-point types. This should greatly reduce the memory footprint and minimal loss to accuracy. \
Note: this param does not currently work when used with TPUs. Only available for GPUs.

* **num_epochs**: Total number of epochs to train the model for.

* **steps_per_execution**: Total number of batches to simultaneously push through each tf.function at one time. \
This a brand new feature so there is not much documentation available yet (although faster training times are claimed). So the optimal value chosen will have to be up to the user. If set to 1, this model will run as normal (one step at a time). \
Note: the only limitation on this param is that 'track_every' must be a multiple of 'steps_per_execution'.

* **track_every**: Determines how often to save a set of values ('best', 'step', 'epoch') needed in case the model is interrupted and needs to be restored. \
Note: 'track_every' must be a multiple of 'steps_per_execution'.

* **threshold**: Acceptable range in difference between the true joint position and the predicted joint position (when calculating the PCK metric).

* **decay_epochs**: Epochs at which to apply a decay factor to the learning rate.

* **decay_factor**: Factor to decay the learning rate by.

* **learning_rate**: Initial learning rate used in calculating the training step size. \
Note: this value will be passed to all available devices (GPU or TPU). But set this param as if only one device is available. The job of scaling the learning rate will be left upto the 'scale' param.

* **scale**: Learning rate scaler used in distributed training. A scale of 0.35 seems to be a good starting point, but the optimal value may require experimentation. Either way, it is not recommended to scale the learning rate with a value greater than 1. \
Note: The learning rate for each replica is determined by multiplying the learning_rate, num_replicas, and the scale.

* **schedule_per_step**: Set to True if user wishes to use a per step schedule (exponential decay) on the learning rate.

* **decay_rate**: Factor to decay the learning rate by when used in an exponential schedule.

* **decay_steps**: Frequency of decay on the learning rate when used in an exponential schedule.

## Hidden

* **switch**: Globally switches the whole codebase from using the base config file to a specific frozen config file (i.e. saved copy of a versioned model). \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model.

* **num_replicas**: Defines the total number of devices (CPU, GPU, or TPU) available. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model.

* **train_size**: Defines the total number of training examples found in all TFRecords. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved using DataGenerator's _get_dataset_size method.

* **val_size**: Defines the total number of validation/testing examples found in all TFRecords. \
Note: **DO NOT CHANGE THIS PARAMETER.** This param is defined and used by the model. \
Note: if this value is accidentally erased and truly gone (i.e. not in a frozen config), the value may be retrieved using DataGenerator's _get_dataset_size method.


