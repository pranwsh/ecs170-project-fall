# ECS 170 Stage 2 - Balanced Work Distribution

## Project Goal

Implement a PyTorch Multi-Layer Perceptron (MLP) for handwritten digit classification using the provided Stage 1 code template and Stage 2 dataset.

Each member is responsible for coding, testing, analysis, and contributing to the final report.

---

### Daniel Park - Data Processing and Preprocessing Experiments

**Primary File:** `Dataset_Loader.py`

**Responsibilities:**
- Adapt the Stage 1 dataset loader for the Stage 2 CSV files.
- Load training and testing data separately.
- Extract labels and normalize pixel values.
- Convert data into appropriate NumPy arrays or PyTorch tensors.
- Visualize sample images and class distributions.
- Compare different preprocessing approaches, such as normalized vs. unnormalized inputs.
- Analyze how preprocessing affects model performance.
- Write the dataset, preprocessing, and related experiment sections of the report.

**Deliverables:** Dataset loader, visualizations, preprocessing experiment results, report section.

### Person 2 - MLP Model and Architecture Experiments

**Primary File:** `Method_MLP.py`

**Responsibilities:**
- Adapt the Stage 1 MLP model for 784 inputs and 10 output classes.
- Implement the forward pass and mini-batch training.
- Support configurable hidden layers, loss functions, optimizers, and learning rates.
- Track training loss and predictions.
- Experiment with different numbers of hidden layers and neurons.
- Compare architecture performance and identify effective configurations.
- Write the model architecture and training methodology sections of the report.

**Deliverables:** MLP implementation, architecture comparison experiments, report section.

### Person 3 - Evaluation and Hyperparameter Experiments

**Primary File:** `Evaluate_Metrics.py`

**Responsibilities:**
- Implement accuracy, precision, recall, and F1 score for multiclass classification.
- Generate training convergence plots showing loss and accuracy.
- Compare optimizers such as Adam and SGD.
- Experiment with different learning rates and appropriate loss functions.
- Create comparison tables and figures for experimental results.
- Analyze training behavior and model performance.
- Write the evaluation, hyperparameter experiments, and results sections of the report.

**Deliverables:** Evaluation functions, convergence plots, hyperparameter experiment results, report section.

### Person 4 - Pipeline Integration, Testing, and Error Analysis

**Primary Files:** `Setting_Stage2.py`, `script_mlp.py`

**Responsibilities:**
- Adapt the Stage 1 settings and main script for Stage 2.
- Integrate the dataset loader, MLP model, evaluation functions, and result saver.
- Configure training and testing using the provided datasets rather than K-fold cross-validation.
- Implement reproducible experiment configurations and organized result saving.
- Test the full pipeline and resolve integration issues.
- Generate and analyze a confusion matrix to identify frequently misclassified digits.
- Compile all report sections, write the introduction and conclusion, and verify the five-page limit.

**Deliverables:** Integrated execution pipeline, testing results, error analysis, final report.

---

## Shared Responsibilities

- Each person owns their assigned files to minimize merge conflicts.
- Agree on function interfaces before writing code.
- All members help debug issues involving their components.
- Everyone performs their assigned experiments using the same baseline configuration.
- Use a validation subset of the training data for experiment comparisons, keeping the provided test data for final evaluation.
- Everyone writes their assigned report section.
- All members review the final code and report before submission.

## Workload Summary

| Member | Main Coding | Experiments / Analysis | Report |
|---|---|---|---|
| Daniel Park | Data loader | Preprocessing comparisons and visualizations | Data and preprocessing |
| Person 2 | MLP model | Architecture comparisons | Model and training |
| Person 3 | Evaluation code | Optimizer, loss, and learning rate comparisons | Results and analysis |
| Person 4 | Integration scripts | Pipeline validation and error analysis | Introduction, conclusion, final compilation |

## Development Order

1. Agree on the data, model, and evaluation interfaces.
2. Persons 1, 2, and 3 develop their components while Person 4 prepares the integration scripts.
3. Integrate the components into one working baseline.
4. Run the assigned experiments using the baseline pipeline.
5. Compare results, write report sections, and assemble the final submission.
