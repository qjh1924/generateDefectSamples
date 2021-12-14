import tensorflow as tf
import os
import glob
import cv2
import numpy as np
import pandas as pd
import random

EPS = 1e-12

train_picture_path = './newdataset/0.normal/'
label_picture_path = './newdataset/0.abnormal/'
train_save_path = './train_save/'
model_save_path = './save_model/'

train_picture_list = glob.glob(os.path.join(train_picture_path, "*"))
label_picture_list = glob.glob(os.path.join(label_picture_path, "*"))

def batch_norm(inp,name="batch_norm"):
    batch_norm_fi = tf.keras.layers.BatchNormalization()(inp, training=True)
    return batch_norm_fi

def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)

gen_input = tf.keras.Input(shape=(256,256,3), name='train_img')
c1 = tf.keras.layers.Conv2D(filters=64,kernel_size=4,strides=2,padding='same',input_shape=[256,256,3])(gen_input)
b1 = batch_norm(c1)
#[1,128,128,64]
c2 = tf.keras.layers.Conv2D(filters=128,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b1))
b2 = batch_norm(c2)
#[1,64,64,256]
c3 = tf.keras.layers.Conv2D(filters=256,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b2))
b3 = batch_norm(c3)
#[1,32,32,256]
c4 = tf.keras.layers.Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b3))
b4 = batch_norm(c4)
#[1,16,16,512]
c5 = tf.keras.layers.Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b4))
b5 = batch_norm(c5)
#[1,8,8,512]
c6 = tf.keras.layers.Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b5))
b6 = batch_norm(c6)
#[1,4,4,512]
c7 = tf.keras.layers.Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b6))
b7 = batch_norm(c7)
#[1,2,2,512]
c8 = tf.keras.layers.Conv2D(filters=512,kernel_size=4,strides=2,padding='same',use_bias=False)(lrelu(b7))
b8 = batch_norm(c8)
#[1,1,1,512]

d1 = tf.keras.layers.Conv2DTranspose(512,kernel_size=4,strides=2,padding='same',use_bias=False)(b8)
d1 = tf.nn.dropout(d1, 0.5)
d1 = tf.concat([batch_norm(d1, name='g_bn_d1'), b7],3)
#[1,2,2,512]
d2 = tf.keras.layers.Conv2DTranspose(512,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d1))
d2 = tf.nn.dropout(d2, 0.5)
d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), b6],3)
#[1,4,4,512]
d3 = tf.keras.layers.Conv2DTranspose(512,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d2))
d3 = tf.nn.dropout(d3, 0.5)
d3 = tf.concat([batch_norm(d3, name='g_bn_d3'), b5],3)
#[1,8.8.512]
d4 = tf.keras.layers.Conv2DTranspose(512,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d3))
d4 = tf.concat([batch_norm(d4, name='g_bn_d4'), b4],3)
#[1,16,16,512]
d5 = tf.keras.layers.Conv2DTranspose(256,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d4))
d5 = tf.concat([batch_norm(d5, name='g_bn_d5'), b3],3)
#[1,32,32,256]
d6 = tf.keras.layers.Conv2DTranspose(128,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d5))
d6 = tf.concat([batch_norm(d6, name='g_bn_d6'), b2],3)
#[1,64,64,128]
d7 = tf.keras.layers.Conv2DTranspose(64,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d6))
d7 = tf.concat([batch_norm(d7, name='g_bn_d7'), b1],3)
#[1,128,128,64]
d8 = tf.keras.layers.Conv2DTranspose(3,kernel_size=4,strides=2,padding='same',use_bias=False)(tf.nn.relu(d7))
gen_out = tf.nn.tanh(d8)
#[1.256,256,3]
gen_model = tf.keras.Model(inputs=gen_input, outputs=gen_out, name='gen_model')

dis_input = tf.keras.Input(shape=(256,256,6), name='train_img')
h1 = tf.keras.layers.Conv2D(64,(4,4),strides=(2,2),padding='same',input_shape=[256,256,3])(dis_input)
h1 = lrelu(h1)
#1*128*128*64
h2 = tf.keras.layers.Conv2D(128,(4,4),strides=(2,2),padding='same',use_bias=False)(h1)
h2 = batch_norm(h2)
h2 = lrelu(h2)
#1*64*64*128
h3 = tf.keras.layers.Conv2D(256,(4,4),strides=(2,2),padding='same',use_bias=False)(h2)
h3 = batch_norm(h3)
h3 = lrelu(h3)
#1*32*32*256
h4 = tf.keras.layers.Conv2D(512,(4,4),strides=(1,1),padding='same',use_bias=False)(h3)
h4 = batch_norm(h4)
h4 = lrelu(h4)
#1*32*32*512
output = tf.keras.layers.Conv2D(1,(4,4),strides=(1,1),padding='same',use_bias=False)(h4)
#1*32*32*1
dis_out = tf.sigmoid(output)
dis_model = tf.keras.Model(inputs=dis_input, outputs=dis_out, name='dis_model')

discriminator_optimizer = tf.keras.optimizers.Adam(2*1e-4)#自适应学习率优化算法
generator_optimizer = tf.keras.optimizers.Adam(2*1e-4)

def l1_loss(src, dst):
    return tf.reduce_mean(tf.abs(src - dst))

def train_step(batch_picture,batch_label,count):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_label = gen_model(batch_picture)
        dis_real = dis_model(tf.concat([batch_picture, batch_label], 3))
        dis_fake = dis_model(tf.concat([batch_picture, gen_label], 3))
        gen_loss_L1 = tf.reduce_mean(l1_loss(gen_label, batch_label))
        gen_loss = tf.reduce_mean(-tf.math.log(dis_fake + EPS)) + 100 * gen_loss_L1
        #+ 1*tf.reduce_mean(l1_loss(gen_label, batch_label))
        dis_loss = tf.reduce_mean(-(tf.math.log(dis_real + EPS) + tf.math.log(1 - dis_fake + EPS)))
        
    gradients_of_generator = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(dis_loss, dis_model.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, dis_model.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_model.trainable_variables))
    
    return gen_loss,dis_loss,gen_label

def evaluate(runstep):
    test_picture_path = './test/0.normal/'
    test_save_path = './test_save/'
    #gen_model.load_weights(model_save_path)
    test_picture_list = glob.glob(os.path.join(test_picture_path, "*"))
    for step in range(len(test_picture_list)):
        testimg_path = test_picture_list[step]
        pic_name, _ = os.path.splitext(os.path.basename(testimg_path))
        picture =  cv2.imread(testimg_path)
        height = picture.shape[0] #得到图片的高
        width = picture.shape[1] #得到图片的宽
        picture_resize_t = cv2.resize(picture, (256, 256))
        picture_resize = picture_resize_t / 127.5 - 1. #归一化，满足gan的输入要求
        batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0)
        gen_label = gen_model(batch_picture)
        out_img = (gen_label[0] + 1.) * 127.5
        out_img = out_img.numpy()
        out_img = cv2.resize(out_img, (width, height))
        write_image_name = test_save_path + str(runstep) + '_' + str(step) + ".png"
        #write_image_name = test_save_path + pic_name + ".png"
        cv2.imwrite(write_image_name, out_img)
        print(step)

def train():
    counter = 0
    #gen_model.load_weights(model_save_path)
    for epoch in range(200):
        random.shuffle(train_picture_list)
        for step in range(len(train_picture_list)):
            counter += 1
            img_path = train_picture_list[step]
            pic_name, _ = os.path.splitext(os.path.basename(img_path))
            label_path = label_picture_path + pic_name + '.png'
            picture =  cv2.imread(img_path)
            label =  cv2.imread(label_path)
            height = picture.shape[0] #得到图片的高
            width = picture.shape[1] #得到图片的宽
            picture_resize_t = cv2.resize(picture, (256, 256))
            picture_resize = picture_resize_t / 127.5 - 1.
            label_resize_t = cv2.resize(label, (256, 256))
            label_resize = label_resize_t / 127.5 - 1.
            batch_picture = np.expand_dims(np.array(picture_resize).astype(np.float32), axis = 0)
            batch_label = np.expand_dims(np.array(label_resize).astype(np.float32), axis = 0)
            gen_loss,dis_loss,gen_label = train_step(batch_picture,batch_label,counter)
            if counter % 4500 == 0:
            	evaluate(counter)
            	gen_model.save_weights(model_save_path)
            if counter % 300 == 0:
                out_img = (gen_label[0] + 1.) * 127.5
                out_img = out_img.numpy()
                out_img = cv2.resize(out_img, (width, height))
                #save_img = np.concatenate((picture_resize_t,out_img,label_resize_t), axis=1)
                write_image_name = train_save_path + str(counter) + ".png"
                cv2.imwrite(write_image_name, out_img)
            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss, dis_loss))

#evaluate(0)
train()
