# ProtoLens

This repository contains the implementation of **ProtoLens**, a flexible and interpretable framework designed for prototype-based learning and sub-sentence-level interpretability in text classification tasks.

## Features

- **Dirichlet Process Gaussian Mixture Models (DPGMM)**: Enables flexible text span extraction for fine-grained prototype learning.
- **Prototype-based Interpretability**: Employs prototype representations for fine-grained interpretability at the sub-sentence level.
- **Prototype Alignment**: Ensures prototypes remain semantically meaningful through adaptive updates.

---

## Repository Structure

| File                | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| `args.py`           | Contains argument definitions and configurations for training and evaluation.                   |
| `DPGMM.py`          | Implements the Dirichlet Process Gaussian Mixture Model for extracting text spans.              |
| `experiment.py`     | Main script to train, validate, and evaluate the framework.                                     |
| `PLens.py`          | Core implementation of the ProtoLens model and related components.                             |
| `utils.py`          | Utility functions for data processing, evaluation, and visualization.                           |

---

## Installation

1. Clone the repository:
   ```bash
   git clone <link>
   cd ProtoLens

## Usage

### Training the Model
Run the `experiment.py` script with appropriate configurations:
```bash
python experiment.py
