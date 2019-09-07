import matplotlib.pyplot as plt
import numpy as np

def show_resoult(history):
    # Рисуем графики  Accuracy и Loss для Тренеровочной и валидационной выборки
    titles = ['Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss']
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(history):
        plt.subplot(2, 2, i + 1)
        plt.title(titles[i])
        plt.grid()
        if i == 0 or i == 1:
            color = 'b'
        else:
            color = 'orange'
        plt.plot(image, c=color, label='mean: %.2f' % np.array(image).mean())
        plt.legend(loc='best')
    plt.show()

    # Рисуем совмещенные графики  Accuracy и Loss для Тренеровочной и валидационной выборки
    titles_ = ['Accuracy', 'Loss']
    plt.figure(figsize=(10, 5))
    k = 0
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.plot(history[0 + k], label='Tain', c='b')
        plt.plot(history[2 + k], label='Test', c='orange')
        plt.title(titles_[i])
        plt.grid()
        plt.legend(loc='best')
        k += 1