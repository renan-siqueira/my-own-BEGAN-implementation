# My Own BEGAN Implementation

---

## About

This project utilizes the __Boundary Equilibrium Generative Adversarial Network (BEGAN)__ architecture. 

`BEGAN` is a type of Generative Adversarial Network (GAN) that focuses on balancing the generator and discriminator during training, leading to a stable training process and high-quality generated images.

---

## Getting Started

---

### 1. Clone the Repository

To clone the repository to your local machine, run:

```bash
git clone https://github.com/renan-siqueira/my-own-BEGAN-implementation.git
```

---

### 2. Setting up a Virtual Environment

Navigate to the project directory:

```bash
cd my-own-BEGAN-implementation
```

Then, set up a virtual environment:

```bash
python -m venv .venv
```

### Activate the virtual environment:

- For Linux/Mac:

```bash
source .venv/bin/activate
```

- For Windows:

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

After activating the virtual environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

*Ensure you have a requirements.txt file listing all your project dependencies.*

---

__If you have a GPU, follow the steps in the "How to Use GPU" section (below). Otherwise, if you're not using a GPU, install PyTorch with the following command:__

```bash
pip install torch torchvision torchaudio
```

---

### How to Use GPU:

### 1. Installing specific dependencies:
After creating and activating your virtual environment:

Windows/Linux/Mac:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

*Note: Make sure your hardware and operating system are compatible with CUDA 12+.*

---

### 4. Preparing the dataset:

- 1. Create a folder named `dataset` at the root of the project.
- 2. Inside the `dataset` folder, create another folder with a name of your choice for the labels.
- 3. Copy all the images you wish to use for training into this folder.

---

### 5. Configuring training parameters:

The `src/json/training_params.json` file is set up with optimized parameters for this type of architecture. However, feel free to modify it according to your needs.

---

### 6. How to use the main script:

The `run.py` script is now your central point for executing various operations. It has been set up to accept arguments to dictate its behavior. Here's how to use it:

#### Training the model:

To train the model, execute the following command:

```bash
python run.py --training
```

The `--training` flag indicates that the training will be executed.

---

### 7. Monitoring the Training:

- You can follow the progress directly in the terminal or console.
- A log file will be generated in the directory specified version training.
- At the end of each epoch, samples of generated images will be saved in the folder of version training, inside the samples folder.
