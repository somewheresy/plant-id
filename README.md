# plant-id

This project demonstrates how to fine-tune a CLIP model for plant identification using the PlantNet300K dataset.

## Prerequisites

- Python 3.11 or higher
- Poetry (Python package manager)

## Installation

1. Install Poetry if you haven't already:

   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone this repository:

   ```
   git clone https://github.com/yourusername/plant-id.git
   cd plant-id
   ```

3. Install dependencies using Poetry:

   ```
   poetry install
   ```

## Usage

To fine-tune the CLIP model on the PlantNet300K dataset, run:

    ```
    poetry shell
    python finetune.py
    ```
