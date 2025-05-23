# import packages
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

def datasize(datatype,num_of_classes):
    #Train/test split is 5:1 for CIFAR-10 and 4:1 for colors
    if datatype == "COLORS":
        train_size = 2880
        test_size = 720

    if datatype == "COLORS_SHAPE":
        train_size = 7680
        test_size = 1920

    if datatype == "CHANNELS":
        train_size = 1000
        test_size = 200

    if datatype == "CIFAR10":
        train_size = num_of_classes*640
        test_size = num_of_classes*100
    
    return train_size, test_size

def build_model_datasets(datatype,details,num_of_classes):
    
    resize_x = details[1]
    resize_y = details[2]
    global_batch_size = details[4]

    if datatype == "CIFAR10":
        # all classes in CIFAR-10 dataset
        full_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
        # select classes to train on
        if num_of_classes == 2:
            #classes = ['frog','ship']
            classes = ['airplane','automobile']
        if num_of_classes == 3:
            #classes = ['automobile','frog','ship']
            classes = ['airplane','automobile','bird']
        if num_of_classes == 4:
            #classes = ['automobile','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat']
        if num_of_classes == 5:
            #classes = ['airplane','automobile','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat','deer']
        if num_of_classes == 6:
            #classes = ['airplane','automobile','bird','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat','deer','dog']
        if num_of_classes == 7:
            #classes = ['airplane','automobile','bird','cat','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat','deer','dog','frog']
        if num_of_classes == 8:
            #classes = ['airplane','automobile','bird','cat','horse','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse']
        if num_of_classes == 9:
            #classes = ['airplane','automobile','bird','cat','horse','dog','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship']
        if num_of_classes == 10:
            #classes = ['airplane','automobile','bird','cat','horse','dog','deer','frog','ship','truck']
            classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
        print("Preparing classes: ")
        print(classes)
        
        class_indicies = []
        for x in classes:
            class_indicies.append(full_classes.index(x))
        
        # load CIFAR-10 dataset
        (full_x_train, full_y_train), (full_x_test, full_y_test) = tf.keras.datasets.cifar10.load_data()
        
        # normalize CIFAR-10 dataset to 0.0-1.0
        full_x_train, full_x_test = full_x_train/255.0, full_x_test/255.0
        
        # pick classes out of CIFAR-10
        full_x_train = full_x_train[np.isin(full_y_train, class_indicies).flatten()]
        full_y_train = full_y_train[np.isin(full_y_train, class_indicies).flatten()]
        full_x_test = full_x_test[np.isin(full_y_test, class_indicies).flatten()]
        full_y_test = full_y_test[np.isin(full_y_test, class_indicies).flatten()]
        
        
        # ensure the same number of pictures of each class are in the training/test set
        train_images_per_class = datasize(datatype,num_of_classes)[0]//len(classes)
        print("Training Images Per Class: "+str(train_images_per_class))
        test_images_per_class = datasize(datatype,num_of_classes)[1]//len(classes)
        print("Testing Images Per Class: "+str(test_images_per_class))
        
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        
        for i in tqdm(range(len(full_y_train)),desc='Filtering Training Images'):
            if y_train.count(full_y_train[i]) < train_images_per_class:
                x_train.append(full_x_train[i])
                y_train.append(full_y_train[i])
        
        for i in tqdm(range(len(full_y_test)),desc='Filtering Testing Images'):
            if y_test.count(full_y_test[i]) < test_images_per_class:
                x_test.append(full_x_test[i])
                y_test.append(full_y_test[i])
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        for x in range(len(class_indicies)):
            y_train[y_train == class_indicies[x]] = x
            y_test[y_test == class_indicies[x]] = x
        
        print("Train Shape: "+str(np.shape(x_train)))
        print("Test Shape: "+str(np.shape(x_test)))
            
        x_train,y_train = shuffle(x_train,y_train)
        x_test,y_test = shuffle(x_test,y_test)
        
        # resize CIFAR-10 data
        x_train, x_test = tf.image.resize(x_train[:,:,:datasize(datatype,num_of_classes)[0]], (resize_x,resize_y)).numpy(), tf.image.resize(x_test[:,:,:datasize(datatype,num_of_classes)[1]], (resize_x,resize_y)).numpy()
        
        # truncate CIFAR-10 dataset to specified train/test size
        x_train, x_test = x_train[:datasize(datatype,num_of_classes)[0]], x_test[:datasize(datatype,num_of_classes)[1]]
        y_train, y_test = y_train[:datasize(datatype,num_of_classes)[0]], y_test[:datasize(datatype,num_of_classes)[1]]      
        
        return x_train, x_test, y_train, y_test, classes

    if datatype == "COLORS" or datatype == "COLORS_SHAPE":
        # set directory and classes for COLORS dataset
        if datatype == "COLORS":
            data_dir = './mixed_colors/noisy_colors'
            classes = ['blue','cyan','cyan_tert','green','magenta','magenta_tert','red','yellow','yellow_tert']
        
        # set directory and classes for COLORS_SHAPE dataset
        if datatype == "COLORS_SHAPE":
            data_dir = './mixed_colors_shapes/noisy_colors'
            classes = ['blue','blue_corner','blue_plus','blue_x',
                        'cyan','cyan_corner','cyan_plus','cyan_x',
                        'green','green_corner','green_plus','green_x',
                        'magenta','magenta_corner','magenta_plus','magenta_x',
                        'red','red_corner','red_plus','red_x',
                        'yellow','yellow_corner','yellow_plus','yellow_x']
        
        # import the dataset from the directory, create batches, and shuffle data
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            seed=42,
            batch_size = global_batch_size,
            shuffle=True,
            image_size=(resize_x, resize_y))

        # infer class names from directory
        class_names = dataset.class_names
        
        # split the dataset into train and test data
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        
        # split the train/test data into data and labels
        x_train = []
        y_train = []
        for images, labels in train_dataset:
            x_train.append(images.numpy())
            y_train.append(labels.numpy())

        x_train = tf.concat(x_train, axis=0)
        y_train = tf.concat(y_train, axis=0)

        x_test = []
        y_test = []
        for images, labels in test_dataset:
            x_test.append(images.numpy())
            y_test.append(labels.numpy())

        x_test = tf.concat(x_test, axis=0)
        y_test = tf.concat(y_test, axis=0)
        
        
        return x_train, x_test, y_train, y_test, classes
        
    if datatype == "CHANNELS":
        # specify desired number of channels and classes
        channels = 12 
        n_classes = 10
        
        # number of channels to add scalar to for each class
        classes_to_add_to = channels-n_classes+1

        # calculate the number of tensors in each class to be evenly distributed
        train_class_size = int(datasize(datatype,num_of_classes)[0]/n_classes)
        test_class_size = int(datasize(datatype,num_of_classes)[1]/n_classes)
        
        # create empty tensor of size (?,resize_x,resize_y,channels)
        x_train = np.array([], dtype=np.int64).reshape(0,resize_x,resize_y,channels)
        y_train = np.array([]*train_class_size)
        x_test = np.array([], dtype=np.int64).reshape(0,resize_x,resize_y,channels)
        y_test = np.array([]*test_class_size)

        # create synthetic training and testing data and labels
        for i in range(n_classes):
        
            # populate tensor of size (train_class_size,resize_x,resize_y,channels) with random numbers between 0 and 1
            x_training_class = np.random.rand(train_class_size,resize_x,resize_y,channels)
            
            # assign one hot encoded label
            y_training_class = np.array([i]*train_class_size)
            
            # create test sets similarly
            x_test_class = np.random.rand(test_class_size,resize_x,resize_y,channels)
            y_test_class = np.array([i]*test_class_size)
            
            # add 0.5 to the relevant channels to differentitate them
            for j in range(classes_to_add_to):
                x_training_class[...,i+j] += 0.5 
                x_test_class[...,i+j] += 0.5 
            
            # compile all classes to form final datasets
            x_train = np.concatenate((x_train,x_training_class))
            y_train = np.concatenate((y_train,y_training_class))
            x_test = np.concatenate((x_test,x_test_class))
            y_test = np.concatenate((y_test,y_test_class))

        # shuffle the newly generated data
        x_train,y_train = shuffle(x_train,y_train)
        x_test,y_test = shuffle(x_test,y_test)
        classes = ['0-2','1-3','2-4','3-5','4-6','5-7','6-8','7-9','8-10','9-11']
        
        return x_train, x_test, y_train, y_test, classes
