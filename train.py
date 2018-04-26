import os, time, datetime, math, random
import numpy as np
import tensorflow as tf
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
filter_size = 4
filter_count = 6

# learning rate
learning_rate = 0.23
# regularization rate
reg_rate = 0.026
# stops after this many iterations
iteration_count = 2000

summary_iter_interval = 40
model_save_interval = 400

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
    fc1_soft_max = tf.nn.softmax(fc1_bias, name="soft_max")
    layer1 = fc1_soft_max
    pred = layer1

    # ---- prediction & accuracy
    pred_value = tf.argmax(pred, axis=1, name="predicted_int_value")
    y_value = tf.argmax(y, axis=1, name="label_int_value")
    pred_compare = tf.equal(pred_value, y_value)
    accuracy = tf.reduce_mean(tf.cast(pred_compare, tf.float32))
    
    # ---- loss & regularization
    l2_norm_0 = tf.nn.l2_loss(w0, name="l2_norm_0")
    l2_norm_1 = tf.nn.l2_loss(w1, name="l2_norm_1")
    loss_base = tf.reduce_mean(tf.squared_difference(y, pred))
    # loss = loss_base
    # loss = loss_base + l2_norm_0 * reg_rate
    loss = loss_base + l2_norm_1 * reg_rate
    # loss = loss_base + l2_norm_0 * reg_rate + l2_norm_1 * reg_rate / 2
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    # optimizer = tf.train.AdagradDAOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
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
    batch_train_in, batch_train_out = data.next_batch(0, 8000, 0)
    batch_train = {x: batch_train_in, y: batch_train_out}
    batch_val_in, batch_val_out = data.next_batch(1, -1, 0)
    batch_val = {x: batch_val_in, y: batch_val_out}
    # batch_test_in, batch_test_out = data.next_batch(2, -1, 0)
    # batch_test = {x: batch_test_in, y: batch_test_out}

    for i in range(iteration_count):
        
        # train the model
        sess.run(train, batch_train)
        
        last_iter = False
        time_current = time.time()
        if i == iteration_count - 1:
            last_iter = True
        if i % summary_iter_interval == 0 or last_iter or (i >= 1000 and i % 20 == 0) or (i >= 1600 and i % 5 == 0):
            report_time_start = time.time()
            # // summary output to TensorBoard and console
            
            # // check on train_data
            # batch_train_in, batch_train_out = data.next_batch(0, -1, 0)
            # loss_train, acc_train = sess.run([loss, accuracy], {x: batch_train_in, y: batch_train_out})
            # s, loss_train, acc_train = sess.run([merged_summary, loss, accuracy], batch_train)
            loss_train, acc_train = sess.run([loss, accuracy], batch_train)

            # // check on val data
            # batch_val_in, batch_val_out = data.next_batch(1, -1, 0)
            # batch_val_in, batch_val_out = data.next_batch(1, 100, 0)
            s, loss_val, acc_val = sess.run([merged_summary, loss, accuracy], batch_val)
            # loss_val, acc_val = sess.run([loss, accuracy], batch_val)

            # // TEST BATCH - ONLY FOR FINAL REPORTING
            # batch_test_in, batch_test_out = data.next_batch(2, -1, 0)
            # s, loss_test, acc_test = sess.run([merged_summary, loss, accuracy], batch_test)
            # loss_test, acc_test = sess.run([loss, accuracy], batch_test)
            
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
        
        if i > 0 and i % model_save_interval == 0 or last_iter:
            save_path = saver.save(sess, dir_path + "/saved_models/" + ts + ".ckpt", global_step=i)
            print("\nmodel saved at " + save_path + "\n")

    print("\nTraining Finished\n")
    print("Running and reporting loss and accuracy on test data in 5s, cancel now to stop...")

    time.sleep(5)
    batch_test_in, batch_test_out = data.next_batch(2, -1, 0)
    batch_test = {x: batch_test_in, y: batch_test_out}
    loss_test, acc_test = sess.run([loss, accuracy], batch_test)
    txt = "[" + str(iteration_count) + "]  test (" + format(loss_test, "8.6f") + "" + format(acc_test*100, "6.1f") + "%)"
    print(txt)
    return 1

if __name__ == "__main__":
    tf_run()
