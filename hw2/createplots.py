import matplotlib.pyplot as plt

epochs = range(30)
loss = [4.62, 4.61, 4.60, 4.59, 4.58, 4.58, 4.57, 4.57, 4.54, 4.46, 4.35, 4.32, 4.23, 4.11, 4.28, 4.36, 4.26, 4.26, 4.21, 4.14, 4.19, 4.16, 4.09, 4.16, 4.00, 4.15, 4.20, 4.17, 4.20, 4.12]
accuracy = [8.16, 8.41, 8.36, 8.41, 8.52, 8.01, 8.46, 8.06, 8.52, 8.26, 8.41, 8.46, 8.06, 8.77, 8.52, 8.31, 9.48, 12.65, 11.63, 12.75, 12.44, 12.85, 13.31, 13.41, 13.62, 13.62, 13.21, 13.46, 13.41, 13.41]
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, 'b--')
plt.savefig('loss.png', dpi=250)
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.plot(epochs, accuracy, 'r--')
plt.savefig('accuracy.png', dpi=250)

transfer_loss = [2.592, 1.544, 0.613, 0.481, 0.307, 0.279, 0.160, 0.177, 0.121, 0.123, 0.088, 0.059, 0.085, 0.032, 0.045, 0.058, 0.019, 0.037, 0.090, 0.030, 0.049, 0.020, 0.019, 0.024, 0.015, 0.019, 0.016, 0.016, 0.019, 0.032]
transfer_accuracy = [41.53, 66.78, 79.43, 84.54, 87.39, 88.77, 90.05, 89.74, 90.51, 90.76, 90.96, 91.22, 91.07, 91.27, 91.39, 90.86, 91.32, 91.27, 91.22, 91.37, 91.27, 91.37, 91.37, 91.37, 91.37, 91.37, 91.37, 91.37, 91.37, 91.37]
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, 'b--', label="baseline")
plt.plot(epochs, transfer_loss, 'b^', label="transfer l.")
plt.legend(loc="center right")
plt.savefig('transfer_loss.png', dpi=250)
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.plot(epochs, accuracy, 'r--', label="baseline")
plt.plot(epochs, transfer_accuracy, 'r^', label="transfer l.")
plt.legend(loc="center right")
plt.savefig('transfer_accuracy.png', dpi=250)
