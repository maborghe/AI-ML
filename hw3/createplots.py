import matplotlib.pyplot as plt

iters = range(30)
class_loss = [2.341, 0.806, 0.682, 0.279, 0.216, 0.186 , 0.146, 0.162, 0.164, 0.086, 0.083, 0.123, 0.113, 0.091, 0.139, 0.132, 0.086, 0.099, 0.072, 0.075, 0.112, 0.077, 0.149, 0.088, 0.057, 0.092, 0.098, 0.080, 0.076, 0.088]
source_loss = [0.936, 0.850, 0.075, 0.040, 0.057, 0.067, 0.092, 0.070, 0.018, 0.065, 0.071, 0.068, 0.177, 0.088, 0.054, 0.023, 0.081, 0.070, 0.045, 0.067, 0.076, 0.079, 0.057, 0.057, 0.121, 0.049, 0.060, 0.012, 0.048, 0.117]
target_loss = [0.545, 0.139, 0.130, 0.102, 0.120, 0.079, 0.055, 0.115, 0.024, 0.047, 0.033, 0.044, 0.095, 0.023, 0.022, 0.039, 0.062, 0.029, 0.026, 0.123, 0.032, 0.038, 0.071, 0.047, 0.029, 0.063, 0.077, 0.040, 0.044, 0.044]
plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(iters, class_loss, 'b--', label="classifier")
plt.plot(iters, source_loss, 'g--', label="discriminator source")
plt.plot(iters, target_loss, 'r--', label="discriminator target")
plt.legend(loc="center right")
plt.savefig('losses.png', dpi=250)

