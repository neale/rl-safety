import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import matplotlib.pyplot as plt 
# Dataset
from tensorflow.examples.tutorials.mnist import input_data
"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
num_sample = mnist.train.num_examples
input_dim = mnist.train.images[0].shape[0]
w = h = int(np.sqrt(input_dim))
"""

def trainer_safelift(learning_rate=1e-4, data, batch_size=64, num_epoch=100, n_z=8, log_step=5):
    print ('trainer')
    # Create a model    
    input_dim = data[0].shape[0]
    w = h = int(np.sqrt(input_dim))
    model = VariationalAutoencoder(input_dim, learning_rate, batch_size, n_z=n_z)
    # Training loop    
    num_sample = len(data)
    for epoch in range(num_epoch):
        start_time = time.time()
        # Run an epoch
        for iter in range(num_sample // batch_size):
            # Get a batch
            if iter*(batch_size=1) > num_sample:
                continue
        
            batch = data[iter*batch_size:iter*(batch_size=1)]
            # Execute the forward and backward pass 
            # Report computed loss
            losses = model.run_single_step(batch[0])
        end_time = time.time()
        
        # Log the loss
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)
            
    print('Done!')
    return model


def trainer_mnist(model_class, learning_rate=1e-4, 
            batch_size=64, num_epoch=100, n_z=16, log_step=5):
    print ('trainer')
    # Create a model    
    model = model_class(
        learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)

    # Training loop    
    for epoch in range(num_epoch):
        start_time = time.time()
        
        # Run an epoch
        for iter in range(num_sample // batch_size):
            # Get a batch
            batch = mnist.train.next_batch(batch_size)
            # Execute the forward and backward pass 
            # Report computed loss
            losses = model.run_single_step(batch[0])
        end_time = time.time()
        
        # Log the loss
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)
            
    print('Done!')
    return model

class VariantionalAutoencoder(object):

    def __init__(self, input_dim, learning_rate=1e-4, batch_size=64, n_z=16):
        # Set hyperparameters
        print('vae')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        # Build the graph
        self.build()
        self.input_dim = input_dim
        # Initialize paramters
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        tf.reset_default_graph()
        self.x = tf.placeholder(
            name='x', dtype=tf.float32, shape=[None, self.input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 128, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu', 
                       activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', 
                                 activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 64, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(g3, input_dim, scope='dec_fc4', 
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + 
            (1-self.x) * tf.log(epsilon+1-self.x_hat), 
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between 
        # the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - 
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = self.recon_loss + self.latent_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        
        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }        
        return
    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.x: x}
        )
        return losses
    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat
    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


def test_transformation(model_2d, mnist, batch_size=3000):
    # Test the trained model: transformation
    print ('test transformation')
    assert model_2d.n_z == 2
    batch = mnist.test.next_batch(batch_size)
    z = model_2d.transformer(batch[0])
    plt.figure(figsize=(10, 8)) 
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(batch[1], 1), s=20)
    plt.colorbar()
    plt.grid()


def test_generation(model, z=None, h=28, w=28, batch_size=100):
    # Test the trained model: generation
    # Sample noise vectors from N(0, 1)
    if z is None:
        z = np.random.normal(size=[batch_size, model.n_z])
    x_generated = model.generator(z)    

    n = np.sqrt(batch_size).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = x_generated[i*n+j, :].reshape(h, w)
            
    plt.figure(figsize=(8, 8))
    plt.imshow(I_generated, cmap='gray')
    plt.show()

def main():
    model_vae = trainer(VariantionalAutoencoder)
    test_generation(model_vae)
    model_vae_2d = trainer(VariantionalAutoencoder, n_z=2)
    test_transformation(model_vae_2d, mnist)
    # Test the trained model: uniformly samlpe in the latent space
    n = 20
    x = np.linspace(-2, 2, n)
    y = np.flip(np.linspace(-2, 2, n))
    z = []
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            z.append(np.array([xi, yi]))
    z = np.stack(z)

    # generate images
    test_generation(model_vae_2d, z, batch_size=n**2)

if __name__ == '__main__':
    main()
