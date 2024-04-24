"""
preprocesses MIDI files
"""
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *
from midi_util import load_midis
from util import *



def compute_beat(beat, notes_in_bar):
    """
    encode the position of a beat within a bar using one-hot encoding
    """
    return one_hot(beat % notes_in_bar, notes_in_bar)

def compute_completion(beat, len_melody):
    """
    compute how much of a melody has been completed at a given beat
    """
    return np.array([beat / len_melody])

def compute_genre(genre_id):
    """
    compute a vector that represents a particular genre using one-hot encoding
    """

    # create an array filled with zeros, where the size is the number of styles
    genre_hot = np.zeros((NUM_STYLES,))
    # enumerate through the sublists of styles in each genre previous to the current genre, 
    # summing the total count of individual style slots that should be skipped in the one-hot vector to reach the starting point for the current genre
    start_index = sum(len(s) for i, s in enumerate(styles) if i < genre_id)
    # calculate the number of styles that define the current genre
    styles_in_genre = len(styles[genre_id])
    # select the current genre (a section of the one-hot vector that corresponds to the current genre) in the vector representing genres,
    # and set the value of each style in the current genre to 1 divided by the total number of styles in the current genre, so each style within the genre contributes equally to the genre's vector representation
    genre_hot[start_index:start_index + styles_in_genre] = 1 / styles_in_genre
    return genre_hot

def stagger(data, time_steps):
    """
    preprocess sequence data into formats suitable for training sequential models
    """

    # create pairs of input (dataX) and target output (dataY) sequences that are offset by one timestep, enabling the model to predict the next step in a sequence based on its current state
    dataX, dataY = [], []

    # buffering: creatw a new data point with zero vectors that has the same shape and type as the first data point in the list,
    # replicate the empty data point for `time_steps` times, creating a fake history of silent or blank moments equal to the number of past moments the model needs,
    # then concatenate the fake history to the beginning of the actual data
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)

    # chop the buffered sequence into overlapping input (dataX) and target (dataY) sequences, which are then used to train the model:
    # iterate over the data array, starting at index 0 and ending at `len(data) - time_steps` to have a complete set of subsequent data points for prediction and avoid out-of-bound errors,
    # the step size is `NOTES_PER_BAR`, reflecting a musical structure where bars are the basic unit
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        # create the input sequence by slicing a segment of the data array starting from index i and extending `time_steps` forward, providing the necessary context for making the next prediction
        dataX.append(data[i:i + time_steps])
        # create the target sequence by slicing a segment of the data array starting from i + 1 (one step ahead of the input sequence's start) and also extends `time_steps` forward,
        # which is what the model will try to predict, given the input sequence; i.e., the "next step" in the sequence
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def load_all(styles, batch_size, time_steps):
    """
    load and prepare all necessary data from MIDI files as piano rolls for neural network training in Keras,
    dealing with musical information such as notes, beats, and styles, and structuring these into sequences that can be used to predict future musical elements based on past context
    """

    # initialize data storage lists
    note_data = []
    beat_data = []
    style_data = []
    note_target = []

    # flatten a list of styles sublists into a single list
    styles = [y for x in styles for y in x]

    # load and process each style's MIDI files in parallel
    for style_id, style in enumerate(styles):
        # iterate over the styles list, creating a one-hot vector representation for each style
        style_hot = one_hot(style_id, NUM_STYLES)
        # convert MIDI files of each style into a sequence with format suitable for model training in parallel to speed up the process
        seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi)(f) for f in get_all_files([style]))
        
        # check each sequence `seq` from the loaded MIDI files while ensuring it has enough length
        for seq in seqs:
            if len(seq) >= time_steps:
                # clamp MIDI to note range
                seq = clamp_midi(seq)

                # create input (training data) and target (labels) sequences using the `stagger` function
                train_data, label_data = stagger(seq, time_steps)
                note_data += train_data
                note_target += label_data
                
                # encode the beat position within a bar and stagger the beat information into sequences
                beats = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
                beat_data += stagger(beats, time_steps)[0]

                # repeat the style vector for the length of the sequence and stagger the style information into sequences,
                # ensuring that each input sequence (a segment of MIDI data) has the corresponding style information
                style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

    # convert the collected lists into numpy arrays suitable for training models in Keras
    note_data = np.array(note_data)
    beat_data = np.array(beat_data)
    style_data = np.array(style_data)
    note_target = np.array(note_target)

    # return a tuple containing arrays of inputs (note data, beat data, style data, note target) and outputs (note target), formatted for model training
    return [note_data, note_target, beat_data, style_data], [note_target]

def clamp_midi(sequence):
    """
    clamp the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]

def unclamp_midi(sequence):
    """
    restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')
