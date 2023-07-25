import matplotlib.pyplot as plt


def drawLoss(loss, output: str, file='train_loss'):
    #对测试Loss进行可视化
    plt.plot(loss, label=file)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(output+file+'.png')
    plt.close()


def drawAccuracy(accuracy, output: str, file='train_accuracy'):
    #对测试准确率进行可视化
    # plt.plot(history['Train Accuracy'],color = 'red',label = 'Train Accuracy')
    plt.plot(accuracy, color = 'red', label=file)
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    # plt.show()
    plt.savefig(output+file+'.png')
    plt.close()


def drawMPR(precision_dict: dict, recall_dict: dict, save_path):
    plt.figure()
    plt.step(recall_dict['macro'], precision_dict['macro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average precision score, micro-averaged over all classes')
    plt.savefig(f'{save_path}Matro PR.png')
    plt.close()
    # plt.show()