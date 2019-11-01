import time
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt 
# Dataset
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.examples.tutorials.mnist import input_data
from safelife.render_graphics import render_game
from safelife.game_physics import CellTypes
from skimage.util import view_as_blocks

"""
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
num_sample = mnist.train.num_examples
input_dim = mnist.train.images[0].shape[0]
w = h = int(np.sqrt(input_dim))
"""

class VariationalAutoencoder(object):

    def __init__(self, input_dim, n_z, learning_rate=1e-4, batch_size=100):
        # Set hyperparameters
        print('vae init with LR={}, BS={}, Z={}'.format(learning_rate, batch_size, n_z))
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.input_dim = input_dim
        # Build the graph
        self.build()
        # Initialize paramters
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        # tf.reset_default_graph()
        self.x = tf.placeholder(
            name='x', dtype=tf.float32, shape=[None, self.input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 512, scope='enc_fc1', activation_fn=tf.nn.elu)
        f2 = fc(f1, 256, scope='enc_fc2', activation_fn=tf.nn.elu)
        f3 = fc(f2, 128, scope='enc_fc3', activation_fn=tf.nn.elu)
        f4 = fc(f3, 64, scope='enc_fc4', activation_fn=tf.nn.elu)
        self.z_mu = fc(f4, self.n_z, scope='enc_fc5_mu', 
                       activation_fn=None)
        self.z_log_sigma_sq = fc(f4, self.n_z, scope='enc_fc5_sigma', 
                                 weights_initializer=tf.initializers.zeros,
                                 activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 64, scope='dec_fc1', activation_fn=tf.nn.elu)
        g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.elu)
        g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.elu)
        g4 = fc(g3, 512, scope='dec_fc4', activation_fn=tf.nn.elu)
        self.x_hat = fc(g4, self.input_dim, scope='dec_fc5', 
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-8
        log_input = tf.maximum(self.x_hat + epsilon, epsilon)
        log_input2 = tf.maximum(epsilon + 1 - self.x_hat, epsilon)
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(log_input) + (1-self.x) * tf.log(log_input2), axis=1)

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


def get_board_state(env):
        board = env.state.board
        goals = env.state.goals
        agent_loc = env.state.agent_loc

        board = board.copy()
        goals = goals & CellTypes.rainbow_color

        if env.remove_white_goals:
            goals *= (goals != CellTypes.rainbow_color)

        board += (goals << 3)

        # And center the array on the agent.
        # remove cause I dont want the Q function to be a function of the agent's fov
        # board = recenter_view(
        #    board, self.view_shape, agent_loc[::-1], self.state.exit_locs)
        if env.output_channels:
            shift = np.array(list(env.output_channels), dtype=np.int16)
            board = (board[...,None] & (1 << shift)) >> shift
        return board


def preprocess_env_state(env):
    s = render_game(env.state)  # get full game frame ( oh I do call render game, Carroll was right )
    s_g = np.dot(s[...,:3], [0.299, 0.587, 0.114])  # convert to intensity
    block_shape = (5, 5)
    view = view_as_blocks(s_g, block_shape)
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)
    mean_view = np.mean(flatten_view, axis=2)  # mean pool
    mean_view *= 1.0 / mean_view.max()  # normalize states
    mean_view = np.reshape(mean_view, [1, -1])  # flatten
    return mean_view


def train_state_vae(envs, replay_size, z_dim):
    # collect experience_buffer
    states = []
    for env in envs:
        _ = env.reset()
    replay_size /= len(envs)
    for step in range(int(replay_size)):
        for env in envs:
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            s = get_board_state(env)  # [25, 25, 15]
            s_flat = s.astype(np.float32).reshape([25*25*15])
            # s = preprocess_env_state(env)
            states.append(s)
            if done:
                _ = env.reset()
            if len(states) % 1000 == 0:
                print ('collected {} states'.format(len(states)))

    states = np.stack(states)
    np.random.shuffle(states)
    states = np.reshape(states, [states.shape[0], -1])
    vae_model = trainer_safelife(states, n_z=z_dim)
    # test_generation(vae_model)  # Not meaningful with bitfield representation
    print ('Trained VAE on random trajectories')
    return vae_model

def trainer_safelife(data, n_z, num_epoch=200, log_step=5):
    batch_size = 100
    input_dim = data[0].shape[0]
    w = h = int(np.sqrt(input_dim))
    model = VariationalAutoencoder(input_dim, n_z)
    # Training loop    
    num_sample = len(data)
    for epoch in range(num_epoch):
        start_time = time.time()
        # Run an epoch
        for iter in range(num_sample // batch_size):
            # Get a batch
            if (iter+1)*(batch_size) > num_sample:
                continue
        
            batch = data[iter*batch_size:(iter+1)*(batch_size)]
            # Execute the forward and backward pass 
            # Report computed loss
            losses = model.run_single_step(batch)
        end_time = time.time()
        
        # Log the loss
        if epoch % log_step == 0:
            log_str = '[Epoch {}] '.format(epoch)
            for k, v in losses.items():
                log_str += '{}: {:.3f}  '.format(k, v)
            log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
            print(log_str)
        
    print('***********************************************8')
    print('Done Training VAE!')
    return model

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


def test_generation(model, z=None, h=25, w=25, batch_size=100):
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
    plt.savefig('./generated2')

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
