## Log File Processor
## This script turns model training log files into graphs for analysis

logged_data <- read.csv('binary_focal_crossentropy_without_core_with_batch_size_32.csv')

# print(head(logged_data))


plot(logged_data$epoch, 
     logged_data$val_loss, 
     type = 'ol', 
     main = 'Validation Loss',
     xlab = 'Epoch', 
     ylab = 'Validation Loss')


