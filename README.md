# Deep learning in the setting of Iron Deficiency Anemia (211212_labCollab)

211212_labCollab is a PyTorch-based deep learning framework for collaborative learning in a medical research setting.

# Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/lab_collaborator.git
   cd lab_collaborator
   
2. Install required dependencies:
   ```pip install -r requirements.txt


# Get Started
1.  Define your PyTorch Lightning module 
2.  Implement your dataset class 
3.  Configure the data module 
4.  Modify the configuration file **130_config.yaml** to suit your experiment settings (model type, learning rate, batch size, etc.)
5.  Run the training script:

python lab_collaborator.py --config_path 130_config.yaml

# Features
1.  Supports various deep learning models including **BiGRU**, **BiLSTM**, **Transformer**, and **simpleANN**. These classes define the architecture and forward pass logic for various 
    models used in the project.
2.  Provides metrics like accuracy, AUROC, precision, recall, sensitivity, and specificity for evaluation.The main training loop is implemented within the labCollabLM class,
    where the **training_step** and **validation_step** methods handle forward passes, loss calculations, and metric computations.
4.  Offers PyTorch Lightning integration for efficient training and logging.

# Contributing
Contributions to Lab Collaborator are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (git checkout -b feature/your_feature).
3.  Make your changes and commit them (git commit -am 'Add new feature').
4.  Push your branch (git push origin feature/your_feature).
5.  Create a pull request.


