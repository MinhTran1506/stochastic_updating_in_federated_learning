import time
import torch
from helper_evaluation import compute_accuracy
from nvflare.client.tracking import SummaryWriter


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer, criterion, device, input_model=None, summary_writer=None, scheduler=None):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    steps = num_epochs * len(train_loader)
    if summary_writer is not None:
        summary_writer = SummaryWriter()
    print("Starting training...")
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            # if not batch_idx % 50:
            #     print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
            #           f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
            #           f'| Loss: {loss:.4f}')
            if batch_idx % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss:.4f}")

                if input_model is not None:
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + batch_idx
                if summary_writer is not None:
                    summary_writer.add_scalar(tag="loss_for_each_batch", scalar=loss, global_step=global_step)
        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():  # save memory during inference
            
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list