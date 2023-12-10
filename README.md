# MLOps (Work in Progress)

This repository is to demonstrate MLOps infrastructure setup for production environment. For this demo we are gonna do a single server deployment of services.

## MLflow: Tracking and Managing Your Machine Learning Experiments

In this section, we'll explore how MLflow can be a valuable tool for tracking and managing your machine learning experiments. MLflow is an open-source platform that provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models.

**Experiment Tracking**: With MLflow, you can easily track and log key parameters and metrics from your machine learning experiments. This allows you to maintain a clear record of your experiments, making it easy to compare different runs and understand how hyperparameters affect model performance.

**Code Packaging**: MLflow allows you to package your machine learning code into reproducible runs, ensuring that others can reproduce your experiments exactly as you did. It captures the code, dependencies, and data versions, making it easy to share and collaborate with team members.

**Model Management**: MLflow provides tools for managing machine learning models throughout their lifecycle. You can register models, version them, and deploy them to various platforms, making it easier to transition from experimentation to production.

**MLflow UI Dashboard**: MLflow includes a user-friendly dashboard that provides a visual representation of your experiments. You can access the dashboard through a web browser, allowing you to interactively explore and compare different runs, metrics, and parameters. The dashboard also provides a timeline view to track the progression of experiments over time.

**Using the MLflow UI Dashboard**:

To launch the MLflow UI dashboard, you can use the following command in your terminal:

```bash
mlflow ui
```

This command starts the MLflow server, and you can access the dashboard by opening a web browser and navigating to the provided URL (usually http://localhost:5000).

**Example Code**: Below is an example code snippet demonstrating how MLflow can be integrated into your machine learning workflow:

In the provided code, MLflow is used to start and manage a run, log hyperparameters and metrics, and maintain a record of the training process.

**Getting Started**: To get started with MLflow, you can install it using `pip` and refer to the official MLflow documentation for more detailed usage instructions: [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

By incorporating MLflow into your machine learning projects, you can streamline experiment tracking, ensure reproducibility, simplify model management, and take advantage of the MLflow UI dashboard to gain insights from your experiments. This enhances your overall workflow and collaboration with team members.

## DVC: Efficient Data and Model Versioning Without Bloating Your Git Repo

In this tutorial, we'll explore how Data Version Control (DVC) can be a game-changer for efficiently managing data and model checkpoints in your machine learning projects without causing your Git repository to balloon in size. DVC doesn't store binary files directly in Git, making it an ideal companion for large datasets and model checkpoints.

**No Cloud Storage Required**: Unlike some other solutions that rely on cloud storage, DVC allows you to store your data and models locally, making it perfect for cases where you don't have cloud credentials or want to keep everything on your own machine.

**Git for Code, DVC for Data**: Think of DVC as working in harmony with Git, but focused on handling non-code assets. While Git efficiently manages your source code, DVC takes care of data and model artifacts.

**Efficient Storage**: DVC doesn't store the actual binary files but rather stores pointers (metadata) to where the artifacts are located. This means your Git repository remains lightweight and responsive, no matter how large your datasets or model checkpoints become.

**Local Storage for This Example**: In this demonstration, we'll store both the data and model locally. You can extend these concepts to cloud storage solutions when needed, with DVC providing the same versioning benefits.

**Analogous to GitHub**: Just as GitHub hosts and manages your source code repositories, cloud storage solutions can be analogous to where you store your machine learning models and checkpoints. DVC ensures that you version, track, and share your data and models as effectively as you do with your code.

With DVC, you'll be able to version and track your machine learning project assets with confidence, knowing that your Git repository remains agile and focused on code while your data and models are handled efficiently in the background. Let's get started with DVC and supercharge your machine learning workflow!