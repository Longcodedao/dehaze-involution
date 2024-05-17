import torch
from tqdm.notebook import tqdm
import datetime
import os 
from .metrics import psnr 
from skimage.metrics import structural_similarity as ssim

def train_per_epoch(model, train_loader, criterion, 
                    optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for xs, ys in train_bar:
        xs = xs.to(device)
        ys = ys.to(device)
        
        preds = model(xs)
        loss = criterion(preds, ys)  # Compute loss using model predictions
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Update tqdm bar with the current loss
        train_bar.set_postfix(loss=loss.item())
        
    avg_train_loss = total_loss / len(train_loader)
    # print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def valid_per_epoch(model, val_loader, criterion, device, epoch, 
                    num_epochs, val_logger, display = True, batch_size = 32):
    
    model.eval()
    total_psnr = 0.0
    total_val_loss = 0.0
    total_ssim = 0.0  # Initialize SSIM accumulator
    running_loss = 0.0

    val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=False)

    val_logger.info(f"Epoch {epoch+1}/{num_epochs}")
    batch_num = 0
    with torch.no_grad():
        for xs, ys in val_bar:
            xs = xs.to(device)
            ys = ys.to(device)
            preds = model(xs)
            loss = criterion(preds, ys)
            total_val_loss += loss.item()
            
            psnr_score = psnr(ys, preds)
            total_psnr += psnr_score
            
            ssim_batch = 0
            # Convert tensors to numpy arrays and compute SSIM
            for i in range(xs.shape[0]):  # Iterate over batch dimension
                ys_np = ys[i].cpu().numpy().squeeze().transpose((1, 2, 0))  # Convert PyTorch tensor to numpy array
                preds_np = preds[i].cpu().numpy().squeeze().transpose((1, 2, 0))  # Convert PyTorch tensor to numpy array

                # max_val = max(ys_np.max(), preds_np.max())
                # min_val = min(ys_np.min(), preds_np.min())
                # data_range = max_val - min_val

                # print(ys_np)
                # print(preds_np)

                # The range of the ssim of the float image is 0 to 1
                ssim_score = ssim(ys_np, preds_np, data_range = 1,
                                  multichannel = True, channel_axis = -1)  # Compute SSIM
                total_ssim += ssim_score
                ssim_batch += ssim_score

            # Update tqdm bar with the current loss
            val_bar.set_postfix(
                loss = loss.item(),
                psnr = psnr_score.item(),
                ssim = ssim_batch / batch_size
            )

            if display:
                val_logger.info(f"Batch num: {batch_num}")
                val_logger.info(f"Loss: {loss.item()}, PSNR: {psnr_score.item()}, SSIM: {ssim_batch / batch_size}")
            # if loss.item() >= 1:
            #     val_logger.warning(f"High loss observed: {loss.item()}")
            #     val_logger.warning(f"Predictions: {preds.cpu().numpy()}")
            #     val_logger.warning(f"Ground Truth: {ys.cpu().numpy()}")

            batch_num += 1

    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / (len(val_loader) * batch_size)  # Calculate average SSIM
    
    # print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

    return avg_val_loss, avg_psnr, avg_ssim 


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                       num_epochs, device, train_log, val_log, batch_size = 32, fine_tune = None):
    
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': []}
    
    # Create a timestamp for the model filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs("weights", exist_ok = True)
    model_path = f"weights/model_{timestamp}.pth" if not fine_tune else f"weights/model_{fine_tune}_{timestamp}.pth"
    
    for epoch in range(num_epochs):
        train_loss = train_per_epoch(model, train_loader, criterion, 
                                     optimizer, device, epoch, num_epochs)
        val_loss, psnr, ssim = valid_per_epoch(model, val_loader, criterion,
                                                  device, epoch, num_epochs, val_log, batch_size = batch_size)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['psnr'].append(psnr)
        history['ssim'].append(ssim)
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {psnr}, SSIM: {ssim}")

        train_log.info(f"Epoch {epoch+1}/{num_epochs}")
        train_log.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {psnr}, SSIM: {ssim}")
        
        # Save the model if it has the best validation loss so far
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")
    
    return history