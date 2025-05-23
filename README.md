# ENHANCE AI WORKFLOWS: AUTOMATING HYPERPARAMETER TUNING WITH KUBERNETES AND KUBEFLOW KATIB

## Initial Setup

### Install kind

The install kind you can use the next code snippet:

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

There is a Dockerfile that can be used to build a local container.

```bash
docker build -t fsgurapsl/mnist-trainer:latest .
```

An already built artifact is available on docker hub.

```bash
docker pull fsgurapsl/mnist-trainer:latest
```

### Load the docker image in the kind cluster

To speed up the process related to use the built container image, it can be load in the current kind cluster.

```bash
kind load docker-image fsgurapsl/mnist-trainer:latest --name kubeflow-katib-demo
```

### Create the experiment

Now that the container is present inside the kind cluster, the experiment can be created and it will start immediately.

```bash
kubectl create -n kubeflow docker/mnist-hyperparameter-tuning.yaml
```

### Check the current elapsing experiment

Opening the UI, there should be now an available `mnist-hyperparameter-tuning` link in the list. Opening it should show the elapsing evolution of the experiment and the report.

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
|Momentum	Helps accelerate gradient descent by smoothing updates, especially in SGD.|
|Early Stopping Patience|Number of epochs without validation improvement before stopping training early. Helps prevent overfitting.|
