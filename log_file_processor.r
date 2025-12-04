## Log File Processor
## This script turns model training log files into graphs for analysis

logged_data <- read.csv('logs/006_add_a_dense_layer_and_add_back_history.csv')

# print(head(logged_data))


plot(logged_data$epoch, 
     logged_data$val_loss, 
     type = 'ol', 
     main = 'Validation Loss',
     xlab = 'Epoch', 
     ylab = 'Validation Loss')


old_data <- read.csv('logs/001_binary_focal_crossentropy_without_core_with_batch_size_32.csv')
print(logged_data$val_loss)
print(old_data$val_loss)