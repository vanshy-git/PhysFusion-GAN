import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import from our 'src' package
from src import config
from src.dataloader import SuperResDataset
from src.model import Generator, Discriminator
from src.transforms import data_transform

# --- Hyperparameters ---
LEARNING_RATE = 2e-4   # From the Pix2Pix paper
BATCH_SIZE = 4         # Adjust this based on your Colab GPU (T4=4, P100=8)
NUM_EPOCHS = 100       # Start with 5-10 just to test
LAMBDA_L1 = 100        # Weight for L1 (pixel) loss, from Pix2Pix
LAMBDA_PHYSICS = 0     # We will implement this in the next step!

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- Create output directories ---
os.makedirs("checkpoints", exist_ok=True, mode=0o777)
os.makedirs("train_samples", exist_ok=True, mode=0o777)

# --- Helper function to save sample images ---
def save_sample_images(generator, loader, epoch, device):
    """Saves a plot of (Input, Generated, Ground Truth)"""
    generator.eval() # Set model to evaluation mode
    
    with torch.no_grad():
        # Get one batch of data
        batch = next(iter(loader))
        lr_th = batch['LR_thermal'].to(device)
        hr_opt = batch['HR_optical'].to(device)
        hr_th = batch['HR_thermal'].to(device)
        
        # Concatenate inputs for the generator
        gen_input = torch.cat((lr_th, hr_opt), 1)
        fake_th = generator(gen_input)
        
        # Denormalize from [-1, 1] to [0, 1] for plotting
        lr_th = (lr_th * 0.5) + 0.5
        hr_th = (hr_th * 0.5) + 0.5
        fake_th = (fake_th * 0.5) + 0.5

        # Save the first image in the batch
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(lr_th[0, 0].cpu(), cmap='hot')
        axs[0].set_title("Input (Blurry Thermal)")
        axs[0].axis('off')
        
        axs[1].imshow(fake_th[0, 0].cpu(), cmap='hot')
        axs[1].set_title("Generated (Fake Thermal)")
        axs[1].axis('off')
        
        axs[2].imshow(hr_th[0, 0].cpu(), cmap='hot')
        axs[2].set_title("Ground Truth (Real Thermal)")
        axs[2].axis('off')
        
        plt.savefig(f"train_samples/epoch_{epoch:03d}.png")
        plt.close(fig)
        
    generator.train() # Set model back to training mode

# --- Main Training Function ---
def main():
    # --- Load Data ---
    print("Loading dataset...")
    dataset = SuperResDataset(
        data_dir=config.PROCESSED_DATA_DIR,
        transform=data_transform
    )
    # Use num_workers > 0 for faster data loading
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- Initialize Models, Optimizers, Losses ---
    gen = Generator(in_channels=4, out_channels=1).to(device)
    disc = Discriminator(in_channels=5).to(device)

    # Use Adam optimizer as in the Pix2Pix paper
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Define loss functions
    bce_loss = nn.BCEWithLogitsLoss() # GAN loss (more stable than simple BCE)
    l1_loss = nn.L1Loss() # Pixel-wise loss

    # --- Training Loop ---
    print("--- Starting training loop ---")
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        for batch in loop:
            lr_th = batch['LR_thermal'].to(device)
            hr_opt = batch['HR_optical'].to(device)
            hr_th = batch['HR_thermal'].to(device)
            
            gen_input = torch.cat((lr_th, hr_opt), 1)
            
            # --- (1) Train Discriminator ---
            opt_disc.zero_grad()
            
            # Create the fake thermal image
            fake_th = gen(gen_input).detach() # Detach to stop gradients to Generator
            
            # Real Case
            real_output = disc(lr_th, hr_opt, hr_th)
            real_labels = torch.ones_like(real_output).to(device)
            loss_disc_real = bce_loss(real_output, real_labels)
            
            # Fake Case
            fake_output = disc(lr_th, hr_opt, fake_th)
            fake_labels = torch.zeros_like(fake_output).to(device)
            loss_disc_fake = bce_loss(fake_output, fake_labels)
            
            # Total Discriminator loss
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()
            
            # --- (2) Train Generator ---
            opt_gen.zero_grad()
            
            # Re-compute generated image (without detaching)
            fake_th_gen = gen(gen_input) 
            fake_output_gen = disc(lr_th, hr_opt, fake_th_gen)
            
            # GAN loss (Generator wants Discriminator to think fake is real)
            loss_gen_gan = bce_loss(fake_output_gen, real_labels)
            
            # L1 (Pixel) loss
            loss_gen_l1 = l1_loss(fake_th_gen, hr_th) * LAMBDA_L1
            
            # Physics loss (placeholder for next step)
            loss_gen_physics = torch.tensor(0.0).to(device) * LAMBDA_PHYSICS
            
            # Total Generator loss
            loss_gen = loss_gen_gan + loss_gen_l1 + loss_gen_physics
            loss_gen.backward()
            opt_gen.step()

            # Update the progress bar
            loop.set_postfix(
                G_Loss=f"{loss_gen.item():.4f}",
                D_Loss=f"{loss_disc.item():.4f}"
            )
        
        # --- End of Epoch ---
        # Save some sample images
        save_sample_images(gen, dataloader, epoch + 1, device)
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'gen_state_dict': gen.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'epoch': epoch,
            }, f"checkpoints/epoch_{epoch+1}.pth")
            print(f"Saved checkpoint for epoch {epoch+1}")

    print("--- Training finished ---")

if __name__ == "__main__":
    main()