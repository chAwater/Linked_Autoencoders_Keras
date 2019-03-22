# Linked Autoencoders - by Keras

## Summary

![](./imgs/Linked_Autoencoders.jpg)



## What is Autoencoder (AE)?

[Keras blog: building-autoencoders-in-keras](https://blog.keras.io/building-autoencoders-in-keras.html)

## Three Steps

- **Step 1**: Train Autoencoders on two datasets (**Train 1**)
  - MNIST
  - HandSign (from [deeplearning.ai](https://www.floydhub.com/deeplearningai/datasets/signs))
- **Step 2**: Train neural network that link two latent layers (**Train 2**)
- **Step 3**: **Reconstruct** the model

#### Trick - Use pre-trained classification model

Both of our datasets are images, usually we use convolutional neural network (**CNN**) to extract the feature in the images. There are many instances/examples that use CNN to perform classification/recognizing for both datasets and can get pretty good result.

So maybe the CNN model, which was trained for classification, already "learned" the latent representation of the data.

If that is true, the our **Step 1** will be separated into two step:
- Step 1a: Build CNN model for classification
- Step 1b: Use pre-trained model to build AE (Drop top layer, fix weight)

There is a very good code example that [jointly train autoencoder and classifier](https://github.com/keras-team/keras/issues/10037#issuecomment-387213211). In fact, this code train a classifier and autoencoder simultaneously, but I think "train a classifier - fix weight - train a autoencoder (decoder)" is a different case (Am I right?).

## Build Convolutional Neural Networks (CNN)

-- Recognize (Classification) HandSign and HandWrite (MNIST) images

### For MNIST dataset

A piece of cake.

```python
# pseudo-code
model = Sequential()
model.add( Conv2D(
    32, kernel_size=3,
    activation='relu', padding='same',
    input_shape=(28,28,1))
)
model.add( MaxPooling2D() )
model.add( Dropout(0.2) )

model.add( Conv2D(
    64, kernel_size=3,
    activation='relu', padding='same')
)
model.add( MaxPooling2D() )
model.add( Dropout(0.2) )

model.add( Flatten() )
model.add( Dense(32, activation='relu')
model.add( Dense(10, activation='softmax') )

model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
)

model.fit( epochs=10 ...)
model.evaluate(...)
# Test accuracy: 0.9921

```

### For HandSign dataset

The HandSign dataset is really small: 1080 for Training, 120 for Testing. And it is also noisy and have more pixel (64 x 64 x 3).

Although there are some examples that have good accuracy ([by Tensorflow](https://github.com/mesolmaz/HandSigns_CNN)), I tried one of them, It can get good result in classification, but not work well in autoencoder.

So I build one in my way.

#### Trick - Data Augmentation

Small dataset + Noisy, biggest enemy of machine learning.

Data Augmentation is a powerful tool that can _create new data from existing one_ through shifting/shearing/rotating/flipping.

![](./imgs/HandSign_data_with_augmentation.jpg)

This trick can "increase the size of data from nowhere" and make your model more robust.

[Keras blog: building-powerful-image-classification-models-using-very-little-data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

```python
train_datagen = ImageDataGenerator( rescale=1./255,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    shear_range        = 0.7,
    zoom_range         = 0.1,
    rotation_range     = 25,
    horizontal_flip    = True,
    vertical_flip      = False,
    fill_mode          = 'nearest',
)
```

#### Trick - Stack CNN, LeakyReLU

```python
signs_inputs = Input(name='SIGNS', shape=(64,64,3,))

x2 = signs_inputs
x2 = Conv2D( 32, kernel_size=3, padding='same', name='Conv2D-1-1' )(x2)
x2 = LeakyReLU(                                 name='LReLU-1-2'  )(x2)
x2 = Conv2D( 32, kernel_size=3, padding='same', name='Conv2D-1-3' )(x2)
x2 = LeakyReLU(                                 name='LReLU-1-4'  )(x2)
x2 = MaxPooling2D(                              name='MaxPool-1-5')(x2)
x2 = Dropout(0.4,                               name='Dropout-1-6')(x2)

x2 = Conv2D( 64, kernel_size=3, padding='same', name='Conv2D-2-1' )(x2)
x2 = LeakyReLU(                                 name='LReLU-2-2'  )(x2)
x2 = Conv2D( 64, kernel_size=3, padding='same', name='Conv2D-2-3' )(x2)
x2 = LeakyReLU(                                 name='LReLU-2-4'  )(x2)
x2 = MaxPooling2D(                              name='MaxPool-2-5')(x2)
x2 = Dropout(0.4,                               name='Dropout-2-6')(x2)

x2 = Conv2D( 64, kernel_size=3, padding='same', name='Conv2D-3-1' )(x2)
x2 = LeakyReLU(                                 name='LReLU-3-2'  )(x2)
x2 = Conv2D( 64, kernel_size=3, padding='same', name='Conv2D-3-3' )(x2)
x2 = LeakyReLU(                                 name='LReLU-3-4'  )(x2)
x2 = MaxPooling2D(                              name='MaxPool-3-5')(x2)
x2 = Dropout(0.2,                               name='Dropout-3-6')(x2)

x2 = Flatten(                                   name='Flatten'    )(x2)
x2 = Dense(512, activation='relu',              name='FC-Latent'  )(x2)
x2 = Dense(6,   activation='softmax',           name='Softmax'    )(x2)
signs_pred = x2

signs_cnn_model = Model(inputs=signs_inputs, outputs=signs_pred, name='SIGNS_CNN')

signs_cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
)

history = signs_cnn_model.fit_generator(
    train_datagen.flow(X_train_orig, Y_train, batch_size=32, shuffle=True),
    epochs=30,
    ...
)
# Test accuracy: 0.9583333333333334
```

There are two mis-predict example. It is excusable, right? ^_^

![](./imgs/Mispredict_HandSign.jpg)


## Build Autoencoders from CNN

-- Based on pre-trained CNN model for classification


---


---

## Linked Latent Layer

-- Just a simple Neural Network


---

## Additional - VAE

---

#### PS

This idea was raise up when I was processing single cell DNA sequencing data.

If you treat the call segment process, which usually use CBS or HMM algorithm, as a dimensional reduction process from the raw copy number data, then maybe you can use a AE to do that.

So I thought I need two AE, one for "dimensional reduction" the raw data, the other one for "dimensional reduction" the segment data (generated by CBS/HMM as training set). Then, use a neural network to link two latent layers.

To do that, first I will try this idea on machine learning benchmark dataset (MNIST), then I will try on real (DNA sequencing) dataset.

If it works, I will create another repository. Hopefully, we can get a neural network version of segment caller!

---

#### BTW

This repository is a branch from a **Private** repository.

[Retaining History When Moving Files Across Repositories in Git](https://stosb.com/blog/retaining-history-when-moving-files-across-repositories-in-git/)

```bash
# Clone repo-A and rename
git clone <repo-A> <tmp>
# Goto repo-A
cd <tmp>
# Avoid mess up repo-A
git remote rm origin
# Keep files (in repo-A/folder) that you want to transfer
git filter-branch --subdirectory-filter <folder> -- --all

# Goto repo-B
cd <repo-B>
# Add branch
git remote add repo-A-branch <tmp>
# Merge
git pull repo-A-branch master --allow-unrelated-histories
# Done
```
