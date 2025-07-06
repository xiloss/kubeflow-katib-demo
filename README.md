# ENHANCE AI WORKFLOWS: AUTOMATING HYPERPARAMETER TUNING WITH KUBERNETES AND KUBEFLOW KATIB

## Initial Setup

### Install kind

To install kind you can use the next code snippet:

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

or use the [helper](kind/get-kind.sh) in the kind folder

```bash
kind/get-kind.sh
```

and select a proper version, for this tutorial the selected one is the 0.20.0.

### Create a kind cluster

Use the following command to create a cluster with name kubeflow-katib-demo

```bash
kind create cluster --name kubeflow-katib-demo
```

and check it

```bash
kubectl cluster-info --context kind-kubeflow-katib-demo
```

check there is only one node, there is no need to have more:

```bash
kubectl get nodes
```

## Katib Setup

The reference to the Katib repository is available at `https://github.com/kubeflow/katib/tree/master`

The official documentation for hyperparameters and tuning is available at `https://www.kubeflow.org/docs/components/katib/overview/#hyperparameters-and-hyperparameter-tuning`

In the [examples](examples/) folder there is a selection of the official experiments available from the official katib repo.

### Install Katib Control Plane

Now install katib control plane, with the latest stable version, for our demo the selected one is the following

```bash
kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=v0.17.0"
```

wait for all the kubeflow pods to be up and running, it should be a matter of seconds

```bash
kubectl get pods -n kubeflow -w
```

Wait until all pods related to Katib (e.g., katib-controller, katib-ui) are in the Running state.

### Test the Katib UI

To reach the katib ui and control it's all up as expected, run

```bash
kubectl port-forward svc/katib-ui 8080:80 -n kubeflow &
```

Point now the browser to the local exposed service, opening

`http://localhost:8080/katib/`

The UI show be available, and the namespace selector should at least show the current available namespaces.
Generally the experiments are visible in the kubeflow namespace.

## Build the first experiment

To simplify the wait time on complex experiment, a specific fast demo has been written and its available in the [docker](docker/) subfolder. The file [train.py](docker/train.py) defines this experiment, with an exahustively commented code.

### Build the Docker image

Go to the [docker](docker/) subfolder.

There is a [Dockerfile](docker/Dockerfile) that can be used to build a local container.

```bash
docker build -t fsgurapsl/mnist-trainer:latest .
```

An already built artifact is available on docker hub.

```bash
docker pull fsgurapsl/mnist-trainer:latest
```

### Build the Docker image for the modified mnist.py

There is also an improvement to use the experiment file [katib-bayesian-optimization-cnn.yaml](docker/katib-bayesian-optimization-cnn.yaml).

The manifest is using a special argument, `optimizer` that can be better understood looking at the changes applied to the file [mnist-optimizer.py](docker/mnist-optimizer.py)]

The [Dockerfile.optimizer](docker/Dockerfile.optimizer) can be used to build that additional local container.

```bash
docker build -t fsgurapsl/mnist-trainer:optimizer .
```

An already built artifact is available on docker hub.

```bash
docker pull fsgurapsl/mnist-trainer:optimizer
```

### Load the docker image in the kind cluster

To speed up the process related to use the built container image, it can be load in the current kind cluster.

```bash
kind load docker-image fsgurapsl/mnist-trainer:latest --name kubeflow-katib-demo
```

### Create the experiment

Now that the container is present inside the kind cluster, the experiment can be created and it will start immediately.

```bash
kubectl create -n kubeflow docker/katib-random-search-mnist.yaml
```

### Check the current elapsing experiment

Opening the UI, there should be now an available `katib-random-search-mnist` link in the list. Opening it should show the elapsing evolution of the experiment and the report.

## Hyperparameters Generics

|Hyperparameter|Purpose|
|--------------|-------|
|Learning Rate|Controls how big the updates to the model weights are during training. A high value may overshoot minima, while a low value may slow down convergence.|
|Batch Size|Number of samples processed before the model updates its weights. Larger batches can be faster but may generalize worse.|
|Number of Epochs|Number of times the model sees the entire training dataset. More epochs allow more learning, but can lead to overfitting.|
|Optimizer|Algorithm used to update model weights (e.g., SGD, Adam, RMSprop). Affects convergence speed and stability.|
|Dropout Rate|Proportion of neurons randomly "dropped" during training to reduce overfitting.|
|Number of Layers|Defines the depth of the neural network. More layers increase capacity but also complexity and overfitting risk.|
|Units per Layer|Number of neurons in each layer. Affects the model's representational power.|
|Weight Initialization|Method to set initial weight values (e.g., Xavier, He). Helps avoid issues like vanishing/exploding gradients.|
|Activation Function|Determines neuron output behavior (e.g., ReLU, Sigmoid, Tanh). Adds non-linearity to the model.
|Momentum|Helps accelerate gradient descent by smoothing updates, especially in SGD.|
|Early Stopping Patience|Number of epochs without validation improvement before stopping training early. Helps prevent overfitting.|

## Additional Experiments

In the [docker](docker/) directory, there are two additional files, that are experiments based on the container image available from the katib official experiments repository, `ghcr.io/kubeflow/katib/pytorch-mnist-cpu:latest`

They are explictly named with suffix `simple` and `cnn`, and they are more complex tuning options for the Bayesian Optimization.

A summary comparison table of the three proposed experiments would be:

|FILENAME|ALGORITHM|MODEL TYPE|GOAL|
|-|-|-|-|
|katib-random-search-mnist.yaml|Random Search|MLP|Maximize Accuracy|
|katib-bayesian-optimization-simple.yaml|Bayesian|MLP|Minimize Loss|
|katib-bayesian-optimization-cnn.yaml|Bayesian|CNN|Maximize Accuracy|

### MLP Multy-layer Perception briefly

An MLP is the most basic type of neural network. It’s also known as a fully connected *feedforward neural network*.

#### How It Works

- Input data (like an image) is flattened into a 1D vector.
- Each neuron in one layer is connected to every neuron in the next layer.
- The network learns patterns by adjusting weights during training.

#### Use Cases
- Simple classification tasks
- Tabular data
- MNIST digit recognition

#### Analogy
Think of an MLP like a spreadsheet with many layers , where each cell calculates something based on all the cells above.

### CNN Convolutional Neural Network

A CNN is a more advanced neural network designed specifically for processing grid-like data , such as images.

#### How It Works

Uses convolutional layers that scan the image with small filters to detect patterns like edges, corners, or textures.
Followed by pooling layers that reduce complexity while preserving important features.
Ends with fully connected layers (like an MLP) for final classification.

#### Use Cases
- Image classification (e.g., identifying cats vs dogs)
- Object detection
- Medical imaging
- Handwriting recognition (like MNIST digits)

#### Analogy

A CNN is like a detective scanning a photo for clues , looking at small sections at a time and gradually building up an understanding of the whole picture.

### MLP vs CNN Comparison

In this table, I've compared the main aspects of the two MLP and CNN to help understand where to use them and what they are better for.

||MLP|CNN|
|-|-|-|
|Best For|Tabular/flat data|Grid-based data (images)|
|Input Format|Flattened vector|2D/3D (e.g., height × width × channels)|
|Parameter Sharing|No|Yes (filters reused across image)|
|Captures Spatial Patterns|Poorly|Very well|
|Number of Parameters|Often high (can overfit)|Lower due to parameter sharing|
|Used In|Basic models|Modern image recognition|

## Summary of the proposed experiments

### The MLP simple bayesian optimization

This refers to [katib-bayesian-optimization-simple.yaml](docker/katib-bayesian-optimization-simple.yaml)

This Kubeflow experiment uses Bayesian Optimization — a powerful sequential design strategy for global optimization of black-box functions — to find the best hyperparameters for training a machine learning model. Specifically, it optimizes the learning rate (lr) and momentum parameters used in a simple PyTorch-based MNIST classification model.

The goal is to minimize the loss (validation loss) down to a target value of 0.001.

#### Key Components of the Experiment

1. **Objective Function**

    - Metric: The experiment minimizes the loss metric
    - Goal: Reach a loss of 0.001 or lower
    - Type: Minimization problem

2. **Algorithm: Bayesian Optimization**

    - Reason: Why Bayesian? It builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters to evaluate next.
    - Efficient Search: Unlike grid or random search, Bayesian optimization learns from previous trials to make smarter suggestions.
    - Random State: A fixed random seed (10) ensures reproducibility across runs.

3. **Search Space**

    The following two hyperparameters are being tuned:

    |PARAMETER|TYPE|RANGE|DESCRIPTION|
    |-|-|-|-|
    |`lr`|double|0.01-0.05|Learning rate|
    |`momentum`|double|0.5-0.9|Momentum term for optimizer|

    These define the feasible space over which the algorithm will explore.

4. **Trial Execution Settings**

    - Max Trials: Up to 12 trials will be run during this experiment
    - Parallelism: 3 trials can run simultaneously
    - Fault Tolerance: Allow up to 3 failed trials before marking the experiment as failed

5. **Trial Template**

    Each trial runs as a Kubernetes Job, executing the training script inside a container:

    Container Image: `ghcr.io/kubeflow/katib/pytorch-mnist-cpu:latest`
    Command : Runs `mnist.py` with configurable hyperparameters:
    ```bash
    python3 /opt/pytorch-mnist/mnist.py \
        --epochs=1 \
        --batch-size=16 \
        --lr=${trialParameters.learningRate} \
        --momentum=${trialParameters.momentum}
    ```
    **Note:**
    `${trialParameters.learningRate}` and `${trialParameters.momentum}` are dynamically substituted by Katib with values suggested by the Bayesian optimization algorithm for each trial. 

#### How the Experiment Works

##### Initialization

The experiment starts with the Bayesian optimization algorithm initializing its internal model.
Optionally, the first few trials might be randomly sampled to bootstrap the model.

##### Iterative Process

For each trial:
1. Katib suggests a set of hyperparameters (lr, momentum) based on the current model.
2. A Kubernetes Job is created and runs the training container with those hyperparameters.
3. The job outputs metrics (e.g., loss), which are collected via Katib’s metrics collector sidecar.

##### Model Update

After each trial completes, the Bayesian model is updated with the new result (hyperparameters + observed loss).
The next trial is scheduled with more informed parameter choices.

##### Termination Conditions

The experiment stops when one of the following is effective:
- The minimum loss goal (0.001) is achieved
- All 12 trials have completed
- Too many trials fail (more than 3)

##### Result

At the end, you’ll get the best-performing trial with the lowest loss, including its optimal hyperparameters.

#### Use Case & Relevance

This experiment demonstrates how Kubeflow Katib can automate and scale hyperparameter tuning using advanced algorithms like Bayesian optimization.

It’s ideal for:

1. optimizing ML models efficiently without manual tweaking.
2. running scalable experiments on Kubernetes clusters.
3. integrating into MLOps pipelines for robust, repeatable model development.




### The CNN bayesian optimization

This refers to [katib-bayesian-optimization-cnn.yaml](docker/katib-bayesian-optimization-cnn.yaml)

This is an evolution of the previous MLP based bayesian experiment. There are some main difference in goals, being this experiment optimized for accuracy and not to reduce rate loss.

#### Key Components of the Experiment

1. **Objective Function**
    
    Type: Maximize
    
    Goal: Reach an accuracy of 0.99 or higher.
    
    Objective Metric Name : accuracy

2. **Algorithm**

    Algorithm: Bayesian Optimization

    Settings: No specific settings mentioned beyond the default behavior.

3. **Search Space**

    Parameters

    - Learning Rate
        
        `--learning-rate`

        Type: Double

        Range: 0.001 to 0.1

    - Batch Size

        `--batch-size`
        
        Type: Integer
        
        Range: 32 to 128

    - Optimizer

        `--optimizer`

        Type: Categorical

        Options: [sgd, adam, rmsprop]


4. **Trial Execution Settings**

    Parallel Trial Count : 3

    Max Trial Count : 12

    Max Failed Trial Count : 6

5. **Trial Template**

    Primary Container Name : training-container

    Trial Parameters :

    - learningRate (mapped to --learning-rate)
    - batchSize (mapped to --batch-size)
    - optimizer (mapped to --optimizer)

    Trial Spec : Kubernetes Job

    Image: `ghcr.io/kubeflow/katib/pytorch-mnist-cpu:latest`

    Command:

    ```bash
    python3 /opt/pytorch-mnist/mnist.py \
        --batch-size=${trialParameters.batchSize} \
        --lr=${trialParameters.learningRate} \
        --optimizer=${trialParameters.optimizer}
    ```

#### How the Experiment Works

##### Initialization

The Bayesian optimization algorithm initializes its internal probabilistic model.
It may start with a few random trials to bootstrap the model.

##### Iterative Process

1. **Suggest Hyperparameters**

   The algorithm suggests values for learning-rate, batch-size, and optimizer.

2. **Run Trial**

   A Kubernetes Job is created, running the training script with the suggested hyperparameters.

3. **Collect Metrics**

   The trial outputs the validation accuracy (accuracy), which is collected by Katib.

4. **Model Update**

   After each trial completes, the Bayesian model updates its understanding of the objective function based on the observed accuracy.
   This informs the next set of hyperparameter suggestions.

5. **Termination Conditions**

    The experiment stops when one of the next conditions is met:

    - The maximum accuracy goal (0.99) is achieved.
    - All 12 trials have completed.
    - More than 6 trials fail.

##### Result

At the end, the best-performing trial (highest accuracy) is identified, along with its optimal hyperparameters.

##### Use Case & Relevance

**Purpose**

Automate hyperparameter tuning for a CNN-based MNIST classifier using Bayesian Optimization.

**Relevance**

Demonstrates how Bayesian Optimization can be applied to more complex models (like CNNs) with multiple hyperparameters, including categorical ones like optimizers.


### The random search optimization

This refers to [katib-random-search-mnist.yaml](docker/katib-random-search-mnist.yaml)

This experiment is completely customized to ensure the whole scenario is understood. It is very similar to the previously describe Bayesian MLP one, but it is consistently simplified to uncover the steps.

This is another Kubeflow Katib experiment designed to find the best hyperparameters for training a PyTorch MNIST classifier, but this time using a different algorithm and objective .

The goal is to maximize accuracy up to 99% , by trying out different combinations of:

- Learning rate (`--lr`)
- Batch size (`--batch_size`)

Each trial runs as a Kubernetes Job, using a custom Docker image that trains the model with the given parameters.

##### Key Components explained

- **Objective**: Maximize accuracy to 0.99
    
    *Goal is high accuracy, not low loss*

- **Algorithm**: Random search

    *Tries random parameter combinations*

- **Max Trials**: 9

    *Total number of trials allowed*

- **Parallel Trials**: 3

    *Up to 3 trials run at the same time*

- **Max Failed Trials**: 3

    *Stop if more than 3 trials fail*

- **Hyperparameters**:
    - `--lr`
    Learning rate (continuous)
    - `--batch_size`
    batch size (categorical)

- **Image**: `fsgurapsl/mnist-trainer:latest`

    *Custom image for training*

    - **Command**: Runs `train.py` with params

        *Trains model using provided hyperparameters*

#### How the Experiment Works

##### Initialization

The experiment starts by defining the search space for `lr` and `batch_size`.

##### Iterative Process

1. **Suggest Hyperparameters**

    Randomly selects values for lr and batch_size.

2. **Run Trial**

    A Kubernetes Job is created, running the training script with the selected hyperparameters.

3. **Collect Metrics**

    The trial outputs the validation accuracy (accuracy), which is collected by Katib.

##### Model Update

  Since this is Random Search, there is no learning between trials, each trial is independent.

##### Termination Conditions

  The experiment stops when one of the following conditions is fulfilled:
  - The maximum accuracy goal (0.99) is achieved.
  - All 9 trials have completed.
  - More than 3 trials fail.

##### Result

  At the end, the best-performing trial (highest accuracy) is identified, along with its optimal hyperparameters.

##### Use Case & Relevance

  **Purpose**: Automate hyperparameter tuning for a PyTorch MNIST classifier using Random Search.

  **Relevance**: Demonstrates a simpler, baseline approach to hyperparameter tuning, useful for comparing against more advanced methods like Bayesian Optimization.


### Comparison Table of the Three Experiments

|FEATURE|EXP 1 - BAYESIAN SIMPLE|EXP 2 - RANDOM SEARCH|EXP 3 - BAYESIAN CNN|
|-|-|-|-|
|Goal|Minimize loss to 0.001|Maximize accuracy to 0.99|Maximize accuracy to 0.99|
|Algorithm|Bayesian Optimization|Random Search|Bayesian Optimization|
|Search Strategy|Sequential, learns from past trials|Random sampling, no learning|Sequential, learns from past trials|
|Hyperparameters|`lr`,`momentum` (both double)|`lr`(double),`batch_size`(categorical)|`learning-rate`(double),`batch-size`(int),`optimizer`(categorical)|
|Training Image|`ghcr.io/kubeflow/katib/pytorch-mnist-cpu:latest`|`fsgurapsl/mnist-trainer:latest`|`fsgurapsl/mnist-trainer:optimizer`|
|Trial Count|Max: 12, Parallel: 3|Max: 9, Parallel: 3|Max: 12, Parallel: 3|
|Objective Metric|Loss|Accuracy|Accuracy|
|Batch Size Type|Fixed (16)|Categorical ([32, 64, 128])|Continuous range (32–128)|
|Momentum|Tunable (double,0.5–0.9)|Not used|Not used|