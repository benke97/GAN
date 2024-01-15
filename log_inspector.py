import matplotlib.pyplot as plt
import ast

# File path
log_file_path = 'training_log_22.txt'

# Initialize lists to store the loss values
loss_G__AB_values = []
loss_G__BA_values = []
loss_D_A_values = []
loss_D_B_values = []
loss_id_A_values = []
loss_id_B_values = []
loss_GAN_AB_values = []
loss_GAN_BA_values = []
loss_cycle_A_values = []
loss_cycle_B_values = []

# Read and parse the log file
with open(log_file_path, 'r') as file:
    for line in file:
        # Convert string to dictionary
        loss_dict = ast.literal_eval(line.strip())

        # Append the loss values to the respective lists
        loss_G__AB_values.append(loss_dict['loss_G_AB'])
        loss_G__BA_values.append(loss_dict['loss_G_BA'])
        loss_D_A_values.append(loss_dict['loss_D_A'])
        loss_D_B_values.append(loss_dict['loss_D_B'])
        loss_id_A_values.append(loss_dict['loss_id_A'])
        loss_id_B_values.append(loss_dict['loss_id_B'])
        loss_cycle_A_values.append(loss_dict['loss_cycle_A'])
        loss_cycle_B_values.append(loss_dict['loss_cycle_B'])



# Plot the losses
plt.figure(figsize=(12, 8))
plt.plot(loss_G__AB_values, label='Generator AB Loss')
plt.plot(loss_G__BA_values, label='Generator BA Loss')
plt.plot(loss_D_A_values, label='Discriminator A Loss')
plt.plot(loss_D_B_values, label='Discriminator B Loss')
plt.plot(loss_id_A_values, label='Identity A Loss')
plt.plot(loss_id_B_values, label='Identity B Loss')
plt.plot(loss_GAN_AB_values, label='GAN AB Loss')
plt.plot(loss_GAN_BA_values, label='GAN BA Loss')
plt.plot(loss_cycle_A_values, label='Cycle A Loss')
plt.plot(loss_cycle_B_values, label='Cycle B Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Losses Over Batches')
plt.legend()
plt.show()