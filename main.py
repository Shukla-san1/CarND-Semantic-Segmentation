import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from sklearn.utils import shuffle



# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    print("Inside Load VGG fun")
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    print("Size of input layer in encoder",input_image.get_shape())
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #vgg_layer3_out = tf.placeholder(tf.float32, [None, None, None, 256])
    # = tf.placeholder(tf.float32, [None, None, None, 512])
    #vgg_layer7_out = tf.placeholder(tf.float32, [None, None, None, 4096])
    
    print("Inside layers fun")
    # creating 1x1 conv layer
    conv_1_by_1 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    
    # Up sample using 1x1 conv layer by 2
    output = tf.layers.conv2d_transpose(conv_1_by_1, num_classes, 4, strides=(2, 2),padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    
    # Up sample using 1x1 conv layer by 2
    conv4_1_by_1 = tf.layers.conv2d(vgg_layer4_out,num_classes,1,padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    input = tf.add(conv4_1_by_1, output)
    input = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2),padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    
    #
    # Up sample using 1x1 conv layer by 8
    conv3_1_by_1 = tf.layers.conv2d(vgg_layer3_out,num_classes,1,padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    input = tf.add(conv3_1_by_1, input)
    Input = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8),padding='same',kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    print("size of output layers",Input.get_shape())
    
    
    return Input

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)

def loss_graph(iteration, loss):
    fig = plt.figure(figsize=(3,4))
    plt.xlabel('EPOCHS',fontweight='bold')
    plt.ylabel('LOSS',fontweight='bold')
    plt.plot(iteration, loss)
    plt.title('Training Loss Graph ', fontweight='bold')
    fig.savefig('Training_Loss_Graph.png', bbox_inches='tight')
    print('Figure is saved!!!')

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    print("Training...")
    loss = []
    iteration = []
    for i in range(epochs):
        for image,label in get_batches_fn(batch_size):
            _, training_loss = sess.run([train_op,cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.65, learning_rate:  0.0001})
        loss.append(training_loss)
        iteration.append(i+1)
        
        print('Epoch ', i, ' Loss ', training_loss)
    
    print("Training... DONE")
    loss_graph(iteration, loss)
    print("Graph Saved !!!")
    pass
	
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    epochs = 50
    batch_size = 32
    learning_rate = .0001
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)


    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)																																																																																																																																																																																																																																																																														

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network


        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image,keep_prob,layer3_out,layer4_out,layer7_out = load_vgg(sess,vgg_path)
        
        op = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        logits, train_op, cross_entropy_loss = optimize(op, correct_label, learning_rate, num_classes)       

        # TODO: Train NN using the train_nn function			
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
            correct_label, keep_prob, learning_rate)
        # saving the trained model
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./tmp/model.ckpt")

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

 

if __name__ == '__main__':
    run()
