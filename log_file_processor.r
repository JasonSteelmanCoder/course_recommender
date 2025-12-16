## Log File Processor
## This script turns model training log files into graphs for analysis

## bests:
## 011_take_out_core_and_add_gpa

logged_data <- read.csv('logs/019_try_dropouts_of_0_25_and_0_2.csv')

# print(head(logged_data))


plot(logged_data$epoch, 
     logged_data$val_loss, 
     type = 'l', 
     main = 'Validation Loss',
     xlab = 'Epoch', 
     ylab = 'Validation Loss')


old_data <- read.csv('logs/011_take_out_core_and_add_gpa.csv')
print(old_data$val_loss)
print(logged_data$val_loss)


