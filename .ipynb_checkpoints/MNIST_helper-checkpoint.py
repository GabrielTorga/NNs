from matplotlib import pyplot as plt
import numpy as np
import tensorflow.keras
from IPython.display import clear_output


def plot_number(x_train, y_train, number, show_label=True, figsize=(10, 5)):
    plt.imshow(x_train[number], cmap='gray')
    if show_label:
        plt.text(0,0,str(y_train[number]), color='w', size=20, verticalalignment="top")
    plt.show()
    
def create_row(x_train, numbers):
    concatenated = x_train[numbers[0]]
    numbers=numbers[1:]
    for n in numbers:
        concatenated = np.concatenate((concatenated, x_train[n]), axis=1)
    return concatenated

def plot_numbers(x_train, numbers, columns=10, show_label=True, figsize=(20, 5)):
    plt.figure(figsize=figsize)
    numbers = np.array(numbers).reshape(-1, columns)
    concatenated = create_row(x_train, numbers[0])
    numbers = numbers[1:,:]
    for row in numbers:
        concatenated = np.concatenate((concatenated, create_row(x_train, row)))
    plt.imshow(concatenated, cmap='gray')
    plt.show()
 
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')
            
            
            
class PlotLosses(tensorflow.keras.callbacks.Callback):
    def __init__(self, plot_interval=1, 
                 evaluate_interval=10, 
                 val_samples = 512, 
                 x_val=None, 
                 y_val_categorical=None):
        self.plot_interval = plot_interval
        self.evaluate_interval = evaluate_interval
        self.x_val = x_val
        self.y_val_categorical = y_val_categorical
        self.val_samples = val_samples
        #self.model = model
    
    def on_train_begin(self, logs={}):
        print('Begin training')
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        if self.evaluate_interval is None:
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('accuracy'))
            self.val_acc.append(logs.get('val_accuracy'))
            self.i += 1
        
        if (epoch%self.plot_interval==0):
            clear_output(wait=True)
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20,5))
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="val_accuracy")
            ax2.legend()
            plt.show();
        #score = self.model.evaluate(x_test, y_test_categorical, verbose=0)
        
        #print("accuracy: ", score[1])
    
    def on_batch_end(self, batch, logs={}):
        if self.evaluate_interval is not None:
            if (batch%self.evaluate_interval==0):
                self.i += 1
                self.logs.append(logs)
                self.x.append(self.i)
                self.losses.append(logs.get('loss'))
                self.acc.append(logs.get('accuracy'))
                if self.x_val is not None:
                    indexes = np.random.permutation(range(self.x_val.shape[0]))
                    score = self.model.evaluate(self.x_val[indexes][:self.val_samples], 
                                                self.y_val_categorical[indexes][:self.val_samples], verbose=0)
                    self.val_losses.append(score[0])
                    self.val_acc.append(score[1])