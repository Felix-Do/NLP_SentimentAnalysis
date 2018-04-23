import os, time, datetime, math, random
import numpy as np
import tensorflow as tf
import nltk
import gensim
import utils


data = utils.Data()
timestamp = utils.Timestamp()

dir_path = os.path.dirname(os.path.realpath(__file__))
# ts = str(round(time.time() * 1000))  # UTC count time in ms
ts = timestamp.get()
print("timestamp: " + ts)
writer = tf.summary.FileWriter(dir_path + "/tensorboard/" + ts)
# writer = tf.summary.FileWriter("/tensorboard/" + ts)

# conv filter (word)size and number of filters
filter_size = 6
filter_count = 8

# learning rate
learning_rate = 0.25
# regularization rate
reg_rate = 0.025
# stops after this many iterations
iteration_count = 10000

summary_iter_interval = 100
model_save_interval = 1000

def tf_run():
    print("\nInitializing Neural Network\n")

    # inputs
    x = tf.placeholder(tf.float32, shape=[None, data.w2v_dim, None], name='x')
    x_conv_shape = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1]
    x_conv_reshaped = tf.reshape(x, x_conv_shape)
    # outputs
    y = tf.placeholder(tf.float32, shape=[None, 2], name='y')


    # ---- [0] convolutional layer
    filter0 = [data.w2v_dim, filter_size, 1, filter_count]
    # filter0 = [filter_size, data.w2v_dim, 1, filter_count]
    stride0 = [1, 1, 1, 1]
    w0 = tf.Variable(tf.truncated_normal(filter0, stddev=0.05), name="w0")
    conv0_pre = tf.nn.conv2d(x_conv_reshaped, w0, stride0, "VALID", name="conv0_pre")
    b0 = tf.Variable(tf.truncated_normal([filter_count], stddev=0.05), name="b0")
    conv0_bias = tf.nn.bias_add(conv0_pre, b0, name="conv0_bias")
    conv0 = tf.nn.relu(conv0_bias, name="conv0")
    pooled_shape = [tf.shape(x)[0], filter_count]
    conv0_pooled = tf.reshape(tf.reduce_max(conv0, 2), pooled_shape)
    layer0 = conv0_pooled

    # ---- [1] fully connected layer
    w1 = tf.Variable(tf.truncated_normal([filter_count, 2],stddev=0.05), name="w1")
    b1 = tf.Variable(tf.truncated_normal([2], stddev=0.05), name="b1")
    fc1_pre = tf.matmul(layer0, w1)
    fc1_bias = tf.nn.bias_add(fc1_pre, b1)
    soft_max = tf.nn.softmax(fc1_bias, name="soft_max")
    layer1 = soft_max
    pred = layer1
    # test_op = tf.nn.relu(soft_max, name="test_op")
    
    # ---- loss & regularization
    l2_norm_0 = tf.nn.l2_loss(w0, name="l2_norm_0")
    l2_norm_1 = tf.nn.l2_loss(w1, name="l2_norm_1")
    loss_base = tf.reduce_mean(tf.squared_difference(y, pred))
    # loss = loss_base
    loss = loss_base + l2_norm_0 * reg_rate
    # loss = loss_base + l2_norm_1 * reg_rate
    # loss = loss_base + l2_norm_0 * reg_rate + l2_norm_1 * reg_rate
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    pred_value = tf.argmax(pred, axis=1, name="predicted_int_value")
    y_value = tf.argmax(y, axis=1, name="label_int_value")
    pred_compare = tf.equal(pred_value, y_value)
    accuracy = tf.reduce_mean(tf.cast(pred_compare, tf.float32))
    
    # tf.summary.histogram("w_0", w0)
    # tf.summary.histogram("b_0", b0)
    # tf.summary.histogram("w_1", w1)
    # tf.summary.histogram("b_1", b1)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    loss_val = loss
    accuracy_val = accuracy
    tf.summary.scalar('loss_val', loss_val)
    tf.summary.scalar('accuracy_val', accuracy_val)
    # tf.summary.scalar('accuracy_test', accuracy_test)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    writer.add_graph(sess.graph)
    saver = tf.train.Saver()

    time_start = time.time()
    time_speed_current = time_start
    time_speed_last = time_start
    time_speed_last_iter = -1
    time_speed_elapsed = 0

    print("\nTraining Starting\n")

    for i in range(iteration_count):
        
        # train the model
        batch_train_in, batch_train_out = data.next_batch(0, 2000)
        # batch_train = {x: batch_train_in, y: batch_train_out}
        sess.run(train, {x: batch_train_in, y: batch_train_out})
        
        last_iter = False
        time_current = time.time()
        if i == iteration_count - 1:
            last_iter = True
        if i % summary_iter_interval == 0 or last_iter:
            report_time_start = time.time()
            # summary output to TensorBoard
            # print("reporting:")
            
            # check on train_data
            batch_train_in, batch_train_out = data.next_batch(0, -1, 0)
            # batch_train_in, batch_train_out = data.next_batch(0, 100, 0)
            loss_train, acc_train = sess.run([loss, accuracy], {x: batch_train_in, y: batch_train_out})
            # s = sess.run(merged_summary, {x: batch_train_in, y: batch_train_out})

            # check on val data
            batch_val_in, batch_val_out = data.next_batch(1, -1, 0)
            # batch_val_in, batch_val_out = data.next_batch(1, 100, 0)
            loss_val, acc_val = sess.run([loss, accuracy], {x: batch_val_in, y: batch_val_out})
            s = sess.run(merged_summary, {x: batch_val_in, y: batch_val_out})

            # // TEST BATCH - ONLY FOR FINAL REPORTING
            # batch_test_in, batch_test_out = data.next_batch(2, -1, 0)
            # loss_test, acc_test = sess.run([loss, accuracy], {x: batch_test_in, y: batch_test_out})
            # s = sess.run(merged_summary, {x: batch_test_in, y: batch_test_out})
            
            writer.add_summary(s, i)
            
            # print info to console
            time_speed_elapsed = time_current - time_speed_last
            if time_speed_elapsed <= 0: time_speed_elapsed = 0.000001
            speed_current = (i - time_speed_last_iter) / time_speed_elapsed
            time_speed_last = time_current
            time_speed_last_iter = i
            report_time_finish = time.time()
            report_time_elapsed = report_time_finish - report_time_start
            # print(i, loss_train, acc_train, speed_current)
            txt = format("[" + str(i) + "]", "8")
            # txt = txt + "speed ( " + format(speed_current, "5.2f") + " i/s " + format(speed_current*60, "4.0f") + " i/m )"
            txt = txt + "speed (" + format(speed_current*60, "4.0f") + " i/m)"
            txt = txt + "  train (" + format(loss_train, "8.6f") + "" + format(acc_train*100, "6.1f") + "%)"
            txt = txt + "  val ("   + format(loss_val  , "8.6f") + "" + format(acc_val  *100, "6.1f") + "%)"
            # speed_scale_no_report = time_speed_elapsed / (time_speed_elapsed - report_time_elapsed) * 100 - 100
            # txt = txt + "  report (" + format(report_time_elapsed, "6.3f") + "s | +" + format(speed_scale_no_report, "4.2f") + "%)"
            print(txt)
            # print(i, "\tspeed:", np.round(speed_current*1000)/1000,"\ttrain:", loss_train, acc_train, "\tval:", loss_val, acc_val)
        
        if i > 0 and i % model_save_interval == 0 or last_iter:
            save_path = saver.save(sess, dir_path + "/saved_models/" + ts + ".ckpt", global_step=i)
            print("\nmodel saved at " + save_path + "\n")

    print("\nTraining Finished\n")
    return 1

if __name__ == "__main__":
    # test_in, test_out = data.next_batch(1, 300)
    # for i in range(10):
    #     print(i, ":[", test_out[i],"]", len(test_in[i]))
    # print(len(test_in))
    # print(np.shape(test_in))
    tf_run()
    # return 1
