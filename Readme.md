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

---

## 8. How to generate images after completing the training (Beta version):

To generate images after completing the training, execute:

```bash
python run.py --image
```

*You can adjust the parameters for image generation in the configuration file at `settings.PATH_IMAGE_PARAMS`.*

---

## 9. How to generate a video through interpolation of the generated images (Beta version):

To generate a video through interpolation of the generated images, execute:

```bash
python run.py --video
```

*Adjust the parameters for video generation in the configuration file located at `settings.PATH_VIDEO_PARAMS`.*

---

## 10. Upscaling:

If you want to upscale the generated __images__ or __video__, use the `--upscale` argument followed by the width value:

```bash
python run.py --image --upscale 1024
```

*Replace `--image` with `--video` if you're generating a video. The above command will upscale the images to a width of 1024 pixels. Adjust as needed.*

---

## License

This project is open-sourced and available to everyone under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you find any bugs or have suggestions for improvements.
