#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras          import backend as K

from keras.layers   import Input, Lambda, Dense
from keras.models   import Model, load_model, model_from_json
from keras.datasets import mnist
from keras.losses   import binary_crossentropy

import numpy as np

import matplotlib.pyplot as plt


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.


# In[3]:


# mnist_autoencoder = load_model('mnist_autoencoder.h5',compile=False)
mnist_autoencoder = load_model('mnist_dAE.h5')

# mnist_autoencoder = model_from_json(mnist_autoencoder.to_json())

vae_latent_dim = 2

mnist_encoder = mnist_autoencoder.layers[1]
mnist_decoder = mnist_autoencoder.layers[2]

mnist_latent_dim = mnist_encoder.outputs[0].shape[1].value
# mnist_latent_dim = 512


# In[4]:


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_loss(x_raw, x_decoded):
    xen_loss = K.mean(binary_crossentropy(x_raw, x_decoded) * 784)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xen_loss + kl_loss


# In[5]:


# _ = mnist_encoder.layers.pop()

vae_inputs = mnist_encoder(mnist_autoencoder.input)
# vae_inputs = Dense(mnist_latent_dim, name='latent_AE')(vae_inputs)

z_mean     = Dense(vae_latent_dim, name='z_mean'   )(vae_inputs)
z_log_var  = Dense(vae_latent_dim, name='z_log_var')(vae_inputs)
z          = Lambda(sampling, output_shape=(vae_latent_dim,), name='z')([z_mean, z_log_var])

vae_encoder = Model(mnist_autoencoder.input, [z_mean, z_log_var, z], name='vae_encoder')


vae_latent = Input(shape=(vae_latent_dim,), name='z_sampling')
vae_latent_output = Dense(mnist_latent_dim, name='latent_sampling', activation='relu')(vae_latent)

# mnist_decoder._layers[0].batch_input_shape = (None, mnist_latent_dim)
# mnist_decoder = model_from_json(mnist_decoder.to_json())

vae_decoder = Model(vae_latent, mnist_decoder(vae_latent_output), name='vae_decoder')


vae_input  = mnist_autoencoder.input
vae_output = vae_decoder( vae_encoder(vae_input)[2] )

vae = Model(
    inputs  = vae_input, 
    outputs = vae_output, 
    name    = 'MNIST_VAE'
)

vae.compile(optimizer='adam',loss=vae_loss)


# In[6]:


history = vae.fit(
    np.expand_dims(x_train, axis=3),
    np.expand_dims(x_train, axis=3),
    epochs=200,
    batch_size=128,
    validation_data=(np.expand_dims(x_test, axis=3), np.expand_dims(x_test, axis=3)),
    verbose=2
)


# In[7]:


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[8]:


x_test_encoded = vae_encoder.predict(np.expand_dims(x_test, axis=3), batch_size=128)[0]
plt.figure(figsize=(8, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, alpha=0.7, s=5)
plt.colorbar()
plt.show()


# In[9]:


# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * 1
        x_decoded = vae_decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.show()


# In[ ]:





# In[ ]:




