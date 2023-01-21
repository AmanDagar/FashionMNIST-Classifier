import os
import sys
import imageio.v3 as io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    path = sys.argv[1]
    height = int(sys.argv[2])
    width = int(sys.argv[3])
    print("--------------------------------------------------------------------------")
    print("Input parameters are: ",path, height, width)
    print("--------------------------------------------------------------------------")
    shape = height*width
    imageList = []
    for image in os.listdir(path):
        imageList.append(io.imread(path+image))

    imageList = np.array(imageList)

    files = os.listdir(path)
    labelCounts = {}
    for i in range(0, len(imageList)):
        n = (int)(files[i][0])
        if(n in labelCounts):
            labelCounts[n] += 1
        else:
            labelCounts[n] = 1

    print("Number of images per label are: ",labelCounts)
    print("--------------------------------------------------------------------------")

    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        axis[x,y].imshow(imageList[currentIndex+10])
        axis[x,y].set_title('Label: ' + str(i))
        currentIndex += labelCounts[i]
    plt.show()

    
    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        data = imageList[currentIndex+10].reshape(shape,1)
        list_x = range(0,shape)
        currentIndex += labelCounts[i]
        axis[x,y].plot(list_x, data[list_x])
        axis[x,y].set_title('Label: ' + str(i))
    plt.show()


    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        data = imageList[currentIndex+10].reshape(shape,1)
        list_x = range(0,shape)
        currentIndex += labelCounts[i]
        axis[x,y].boxplot(data)
        axis[x,y].set_title('Label: ' + str(i))
    plt.show()

    countMaxOver60 = 0
    for i in imageList:
        if(i.max()>60):
            countMaxOver60 += 1
    print("Number of images having a random value more than 60: ", countMaxOver60)

    print("Lengh of list: ", len(imageList))
    maxIndexCount = np.zeros((len(imageList[0])*len(imageList[0][0])), dtype=int)
    for i in imageList:
        maxIndexCount[np.argmax(i)] += 1

    print('Number of times each index contains the highest value in the images: \n', maxIndexCount)
    print('Index that contains the highest number of highest values in the images: ', np.unravel_index(maxIndexCount.argmax(), (height,width)))
    print("--------------------------------------------------------------------------")

    newList = []
    for i in range(0, len(imageList)):
        highestIndex = np.unravel_index(imageList[i].argmax(), imageList[i].shape)
        if(highestIndex == (14,14)):
            imageList[i][highestIndex] = 0
            




    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        axis[x,y].imshow(imageList[currentIndex+10])
        axis[x,y].set_title('Label: ' + str(i))
        currentIndex += labelCounts[i]
    plt.show()


    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    x = np.linspace(0, 10, 1000)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        data = imageList[currentIndex+10].reshape(shape,1)
        list_x = range(0,shape)
        currentIndex += labelCounts[i]
        axis[x,y].boxplot(data)
        axis[x,y].set_title('Label: ' + str(i))
    plt.show()


    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    x = np.linspace(0, 10, 1000)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        data = imageList[currentIndex+10].reshape(shape,1)
        list_x = range(0,shape)
        currentIndex += labelCounts[i]
        axis[x,y].plot(list_x, data[list_x])
        axis[x,y].set_title('Label: ' + str(i))
    plt.show()

    print("Lengh of list: ", len(imageList))
    maxIndexCount = np.zeros((len(imageList[0])*len(imageList[0][0])), dtype=int)
    for i in imageList:
        maxIndexCount[np.argmax(i)] += 1


    print('Number of times each index contains the highest value in the image: \n', maxIndexCount)
    print("--------------------------------------------------------------------------")

    countMaxOver60 = 0
    for i in imageList:
        if(i.max()>60):
            countMaxOver60 += 1
    print("Number of images having a random value more than 60: ", countMaxOver60)
    print("--------------------------------------------------------------------------")

    imageList = []
    for image in os.listdir(path):
        imageList.append(io.imread('FashionMNIST/'+image))
    imageList = np.array(imageList)
    files = os.listdir(path)
    labelCounts2 = {}
    countOver60 = 0
    labelSum = {}
    for i in range(0, len(imageList)):
        highestIndex = np.unravel_index(imageList[i].argmax(), imageList[i].shape)
        if(highestIndex != (14,14)):
            if(imageList[i][highestIndex]>60):
                countOver60 += 1
            n = (int)(files[i][0])
            if(n in labelCounts2):
                labelSum[n].append(imageList[i][(14,14)])
                labelCounts2[n] += 1
            else:
                labelSum[n] = [imageList[i][(14,14)]]
                labelCounts2[n] = 1


    print("Number of images with correct 14,14 index having highest value > 60: ", countOver60)
    print("Number of images per label with correct 14,14 index: ", labelCounts2)
    print("Values at 14,14 in correct images per label", labelSum)
    print("--------------------------------------------------------------------------")

    for i in labelSum.keys():
        labelSum[i] = int(np.mean(labelSum[i]))
    print("Average value for (14,14) in the correct images: ", labelSum)

    imageList = []
    for image in os.listdir(path):
        imageList.append(io.imread('FashionMNIST/'+image))

    labels = []
    imageList = np.array(imageList)
    files = os.listdir(path)
    labelCounts = {}
    for i in range(0, len(imageList)):
        one_hot = np.zeros(10)
        n = (int)(files[i][0])
        highestIndex = np.unravel_index(imageList[i].argmax(), imageList[i].shape)
        if(highestIndex == (14,14)):
            imageList[i][highestIndex] = labelSum[n]
    #     if(highestIndex != (14,14)):
    #         imageList[i][highestIndex] = np.median(imageList[i].reshape(1,shape))
    #         imageList[i][(14,14)] = 0
        if(n in labelCounts):
            labelCounts[n] += 1
        else:
            labelCounts[n] = 1
        one_hot[n] = 1
        labels.append(one_hot)
    labels = np.array(labels)
    print("Updated images with incorrect value at 14,14 by the median of values at 14,14 in the correct images belonging to the same label")
    print("--------------------------------------------------------------------------")
    print("Label Counts: ", labelCounts)
    print("--------------------------------------------------------------------------")

    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        axis[x,y].imshow(imageList[currentIndex+10])
        axis[x,y].set_title('Label: ' + str(i))
        currentIndex += labelCounts[i]
    plt.show()

    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    x = np.linspace(0, 10, 1000)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        data = imageList[currentIndex+10].reshape(shape,1)
        list_x = range(0,shape)
        currentIndex += labelCounts[i]
        axis[x,y].boxplot(data)
        axis[x,y].set_title('Label: ' + str(i))
    plt.show()

    f, axis = plt.subplots(2,5)
    f.set_figwidth(10)
    f.set_figheight(10)
    f.set_dpi(142)
    x = np.linspace(0, 10, 1000)
    currentIndex = 0
    for i in labelCounts.keys():
        x = i%2
        y= i%5
        data = imageList[currentIndex+10].reshape(shape,1)
        list_x = range(0,shape)
        currentIndex += labelCounts[i]
        axis[x,y].plot(list_x, data[list_x])
        axis[x,y].set_title('Label: ' + str(i))
    plt.show()

    imagesRequired = max(labelCounts.values())
    imageIndex = 0
    listIndex = 0
    balancedImageList = []
    balancedLabels = []
    testCount = imagesRequired//3
    trainCount = imagesRequired - (testCount)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    trainIndex = 0
    testIndex = 0
    for i in labelCounts.keys():
        one_hot = np.zeros(10)
        one_hot[i] = 1
        for k in range(listIndex, (listIndex+imagesRequired)):
             balancedLabels.append(one_hot)
        imagesAvailable = labelCounts[i]
        print("Available images for " + str(i) + " are: " + str(imagesAvailable))
        timesDuplicate = imagesRequired//imagesAvailable
        print("Duplication Required is " + str(timesDuplicate) + " times")
        remainder = imagesRequired%imagesAvailable
        print("Remainder Images will be " + str(remainder))
        for j in range(0,timesDuplicate):
            balancedImageList[(j*imagesAvailable)+listIndex:(j*imagesAvailable)+listIndex+imagesAvailable] = imageList[imageIndex:imageIndex+imagesAvailable]
            print("Start Index: " + str((j*imagesAvailable)+listIndex) + " | End Index: " + str((j*imagesAvailable)+listIndex+imagesAvailable))
        balancedImageList[(timesDuplicate*imagesAvailable)+listIndex:(timesDuplicate*imagesAvailable)+listIndex+remainder] = imageList[imageIndex:imageIndex+remainder]
        print("Start Index: " + str((timesDuplicate*imagesAvailable)+listIndex) + " | End Index: " + str((timesDuplicate*imagesAvailable)+listIndex+remainder))
        print("Indexes used are: imageIndex - " + str(imageIndex) + " | lastAvailableImageIndex - " + str(imageIndex+imagesAvailable))
        train_data[trainIndex:trainIndex+trainCount] = balancedImageList[listIndex:listIndex+trainCount]
        train_label[trainIndex:trainIndex+trainCount] = balancedLabels[listIndex:listIndex+trainCount]
        test_data[testIndex:testIndex+testCount] = balancedImageList[listIndex+trainCount:listIndex+imagesRequired]
        test_label[testIndex:testIndex+testCount] = balancedLabels[listIndex+trainCount:listIndex+imagesRequired]
        trainIndex += trainCount
        testIndex += testCount
        imageIndex += imagesAvailable
        listIndex += imagesRequired
        print("Added " + str(imagesAvailable) + " images.... Current length of list is " + str(len(balancedImageList)))
        print("--------------------------------------------------------------------------")

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras import callbacks
    from sklearn.utils import shuffle

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    traind = ((train_data/imageList.max()))
    trainl = train_label
    traind = np.reshape(traind,(len(train_data),height,width,1))

    # rand_index = np.arange(0,len(traind))
    # np.random.shuffle(rand_index)
    # traind = traind[rand_index]
    # trainl = trainl[rand_index]

    traind, trainl = shuffle(traind, trainl, random_state=0)

    # print(traind[0].shape)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=traind[0].shape))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    # model.add(Dense(64))
    # model.add(Dropout(0.5))

    # model.add(Dense(64))
    # model.add(Dropout(0.5))

    # model.add(Dense(1))

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                            mode ="min", patience = 5, 
                                            restore_best_weights = True)

    model.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=['accuracy'])



    model.fit(traind, trainl, epochs=50, batch_size=64, shuffle=True, validation_split=0.1, callbacks =[earlystopping])
    
    print("--------------------------------------------------------------------------")
    
    testd = ((test_data/imageList.max()))
    testl = test_label
    testd = np.reshape(testd,(len(test_data),height,width,1))

    outputs = model.predict(testd)

    score = model.evaluate(testd, testl, batch_size=128)

    confusion_matrix = np.zeros((len(labelCounts), len(labelCounts)), dtype=int)
    for i in range(0,len(outputs)):
        predicted = np.argmax(outputs[i])
        actual = np.argmax(testl[i])
        confusion_matrix[actual][predicted] += 1

    N = len(outputs)
    print("Confusion Matrix: ", confusion_matrix)
    print("--------------------------------------------------------------------------")

    #Classification Error
    sum = 0
    for i in range(0, len(confusion_matrix)):
        sum += confusion_matrix[i][i]
    classification_error = 1 - (sum/N)
    print("Classification Error: ", classification_error)
    print("--------------------------------------------------------------------------")

    #elementary_probabilities
    sum_column = np.sum(confusion_matrix, axis=0)
    for i in range(0, len(confusion_matrix)):
        print("Elementary probability for predicting k = ", i, ": ", (sum_column[i]/N))
        
    sum_column = np.sum(confusion_matrix, axis=1)
    for i in range(0, len(confusion_matrix)):
        print("Elementary probability result is t = ", i, ": ", (sum_column[i]/N))
    print("--------------------------------------------------------------------------")


    #probability that predicted is 2 given that actual is 1
    #P(y=2 | t=1) = P(y=2, t=1)/P(t=1)
    res = (confusion_matrix[6][4]/N)/(np.sum(confusion_matrix, axis=1)[6]/N)
    print("Probability of getting 4 while actual is 6: ", res)
    print("--------------------------------------------------------------------------")
    
    print("Model Accuracy: ", score[1])
    