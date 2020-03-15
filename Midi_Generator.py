import tensorflow as tf;
import numpy as np;
import PIL;
import matplotlib.pyplot as plt;
import re;
import os;
import mido;
import random;
import glob;


def output(data):
    mid = mido.MidiFile();
    track = mido.MidiTrack();
    mid.tracks.append(track);

    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(35)))

    for i in data:
        print(i);
        track.append(mido.Message("note_on", note=i  if i <= 127 else 127, velocity=127, time=64));
        #track.append(mido.Message("note_on", note=i[0]  if i[0] <= 127 else 127, velocity=127, time=i[1] if i[1] <= 127 else 127));

    mid.save("new_song.mid")



def split_input_target(chunk):
    input = chunk[:-1];
    target = chunk[1:];
    return input, target;








def generate_text(model, start):

  # number of chars
  num_generate = 1000;

  # Converting our start string to numbers (vectorizing)
  input_eval = start;
  input_eval = tf.expand_dims(input_eval, 0);
  input_eval = tf.expand_dims(input_eval, 0);

  # generated_text
  generated = [];

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.9;

   # Here batch size == 1
  model.reset_states();
  for i in range(num_generate):
      predictions = model(input_eval);
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0);

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature;
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy();

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0);
      input_eval = tf.expand_dims(input_eval, 0);

      generated.append(predicted_id);

  return generated;












BATCH_SIZE = 32;
SEQUENCE_LENGTH = 256;
BUFFER_SIZE = 10000;



files = glob.glob(r"E:\MachineLearningDatabase\Midi\1000 Hardstyle MIDIs and 400 Trance MIDIs\hardstyle\*");







midi_data = [];

for file in files:
    print(file);
    mid = mido.MidiFile(file);
    for track in mid.tracks:
        #print("");
        for msg in track:
            #print(msg);
            if(msg.type == "note_on"):
                midi_data.append([msg.note if msg.note <= 127 else 127]);
                #midi_data.append([msg.note if msg.note <= 127 else 127  , msg.time  if msg.time <= 31 else 31]);


print("");






raw_dataset = tf.data.Dataset.from_tensor_slices(np.array(midi_data));
sequences = raw_dataset.batch(SEQUENCE_LENGTH+1, drop_remainder=True);
dataset = sequences.map(split_input_target);
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True);


print(midi_data);



model = tf.keras.Sequential([
    tf.keras.layers.Flatten(batch_input_shape = (BATCH_SIZE, None, None)),
    tf.keras.layers.Embedding(4096, 256, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.LSTM(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(4096)
]);

model.summary();




def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True);

model.compile(optimizer='adam', loss=loss);




checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir));
model.fit(dataset, epochs=100, callbacks=[checkpoint_callback]);





tf.train.latest_checkpoint(checkpoint_dir)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(batch_input_shape = (1, None, None)),
    tf.keras.layers.Embedding(4096, 256, batch_input_shape=[BATCH_SIZE, None]),
    tf.keras.layers.LSTM(1024, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(4096)
]);
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

output(np.array(generate_text(model, [60])));
#output(np.reshape(np.array(generate_text(model, [60,64])),(-1,2)));



