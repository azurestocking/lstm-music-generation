import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Concatenate, Add
from keras.models import Model
import keras.backend as K
from keras import losses

from util import *
from constants import *



# Loss function
def primary_loss(y_true, y_pred):
    """
    calculate the model's loss based on three separate criteria based on if a note is played or not,
    returning the sum of the three loss components, which provides a comprehensive measure of model performance across different aspects of music generation
    """

    played = y_true[:, :, :, 0]
    bce_note = losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    bce_replay = losses.binary_crossentropy(y_true[:, :, :, 1], tf.multiply(played, y_pred[:, :, :, 1]) + tf.multiply(1 - played, y_true[:, :, :, 1]))
    mse = losses.mean_squared_error(y_true[:, :, :, 2], tf.multiply(played, y_pred[:, :, :, 2]) + tf.multiply(1 - played, y_true[:, :, :, 2]))
    return bce_note + bce_replay + mse

# Feature function - pitch position
def pitch_pos_in_f(time_steps):
    """
    return a constant containing pitch position of each note:
    generate a feature representing the normalized position of each note within the possible range of notes
    """

    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f

# Feature function - pitch class
def pitch_class_in_f(time_steps):
    """
    return a constant containing pitch class of each note:
    generate a one-hot encoded representation of the pitch class (note within an octave), which helps the model learn relationships between different pitch classes
    """

    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f

# Feature function - pitch bins
def pitch_bins_f(time_steps):
    """
    return a constant of pitch bins (categories into which pitches are organized based on certain characteristics),
    computed as a sum of the activations or presences of pitches at intervals determined by the octave across the input tensor
    """
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f

def time_axis(dropout):
    """
    define a nested function that captures the temporal patterns of the music, i.e., how different elements of the music evolve over time, including notes, rhythms (beats), and stylistic elements,
    focusing on how these features interact across the sequence of a musical piece
    """

    def f(notes, beat, style):
        # extract the number of time steps from the `notes` input
        time_steps = int(notes.get_shape()[1])

        # apply convolutional layers to the notes inputs to extract octave information
        # TODO: experiment with when to apply conv
        note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
        note_octave = Activation('tanh')(note_octave)
        note_octave = Dropout(dropout)(note_octave)

        # construct a comprehensive feature set for each note by concatenating, including pitch position, pitch class, pitch bins, processed note octave, and repeated beat information
        note_features = Concatenate()([
            Lambda(pitch_pos_in_f(time_steps))(notes), 
            Lambda(pitch_class_in_f(time_steps))(notes), 
            Lambda(pitch_bins_f(time_steps))(notes), 
            note_octave,
            TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        ])

        # reshape the note features to match the LSTM input shape, making the time dimension come first
        x = note_features
        x = Permute((2, 1, 3))(x) # [batch, notes, time, features]

        # LSTM processing
        for l in range(TIME_AXIS_LAYERS):
            # integrate style
            style_proj = Dense(int(x.get_shape()[3]))(style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Activation('tanh')(style_proj)
            style_proj = Dropout(dropout)(style_proj)
            style_proj = Permute((2, 1, 3))(style_proj)
            x = Add()([x, style_proj])

            # apply LSTM
            x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
            x = Dropout(dropout)(x)

        # rearrange the output to match the expected format
        return Permute((2, 1, 3))(x) # [batch, time, notes, features]
    return f

def note_axis(dropout):
    """
    define a nested function that zooms into individual note characteristics and their immediate relationships, such as whether a note is played, its volume, or other attributes,
    focusing on the characteristics that define each note's presence and qualities in the context of its surrounding notes
    """

    # cache for dense and LSTM layers to avoid reinitializing them for each call
    dense_layer_cache = {}
    lstm_layer_cache = {}
    note_dense = Dense(2, activation='sigmoid', name='note_dense')
    volume_dense = Dense(1, name='volume_dense')

    def f(x, chosen, style):
        time_steps = int(x.get_shape()[1])

        # shift target one note to the left, to predict the next note based on the previous one, a common technique in sequence prediction tasks
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)

        # adjust `shift_chosen` to ensure that its dimensions are compatible for concatenation with x
        shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen) # [batch, time, notes, 1]

        # add a new feature to each note in the sequence that represents the previous note's information
        x = Concatenate(axis=3)([x, shift_chosen]) # [batch, time, notes, features + 1]

        # LSTM processing
        for l in range(NOTE_AXIS_LAYERS):
            # integrate style
            if l not in dense_layer_cache:
                dense_layer_cache[l] = Dense(int(x.get_shape()[3]))

            style_proj = dense_layer_cache[l](style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Activation('tanh')(style_proj)
            style_proj = Dropout(dropout)(style_proj)
            x = Add()([x, style_proj])

            # apply LSTM
            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = LSTM(NOTE_AXIS_UNITS, return_sequences=True)

            x = TimeDistributed(lstm_layer_cache[l])(x)
            x = Dropout(dropout)(x)

        # concatenate the predictions from the `note_dense` and `volume_dense` layers, 
        # providing a combined prediction of note presence/absence and volume for each note in the sequence
        return Concatenate()([note_dense(x), volume_dense(x)])
    return f

# LSTM architecture
def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    """
    set up a versatile architecture that not only handles the generation of music based on learned patterns and styles,
    but also provides three models each serve distinct purposes and operate at different levels of granularity and scope within the music generation process
    """
    
    # inputs:
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS)) # Target input for conditioning

    # process:
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)

    style_l = Dense(STYLE_UNITS, name='style')
    style = style_l(style_in)

    # outputs:
    time_out = time_axis(dropout)(notes, beat, style)

    naxis = note_axis(dropout)
    notes_out = naxis(time_out, chosen, style)

    # model:
    """
    comprehensive model
    - purpose: designed for the full task of music generation, handling all aspects from temporal progression to individual note characteristics
    - inputs: notes_in, beat_in, style_in, chosen_in
    - outputs: notes_out
    """
    model = Model([notes_in, chosen_in, beat_in, style_in], [notes_out])
    model.compile(optimizer='nadam', loss=[primary_loss])

    """
    time generation model
    - purpose: process the temporal dynamics of the musical data, using the inputs related to notes, beats, and style
    - inputs: exclude the `chosen_in` input
    - outputs: time_out
    """
    time_model = Model([notes_in, beat_in, style_in], [time_out])

    """
    note generation model
    - purpose: used for generating individual notes or small sequences of notes within a given musical context
    - inputs: use detailed per-note features (`note_features`) rather than full sequences, indicating a more granular or focused application
    - outputs: note_gen_out
    """
    # inputs:
    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')

    # process:
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    style_gen = style_l(style_gen_in)

    # output:
    note_gen_out = naxis(note_features, chosen_gen, style_gen)

    # model:
    note_model = Model([note_features, chosen_gen_in, style_gen_in], note_gen_out)

    return model, time_model, note_model
