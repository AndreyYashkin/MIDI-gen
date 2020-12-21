# MIDI-gen

This project is an attempt to build a simple MIDI generator using RNN.
The project is written on Pytorch with FastAI. For evaluation purposes only Pytorch is needed if *IntToFloatTensor* will be implemented.
I used IntToFloatTensor from FastAI to have an ability to easily show rolls in notebooks.

I use several different metrics to evaluate a quality of my models. You can see their values in my notebooks.

To gerate new MIDI files you need to pass as arguments *gan.py* a beggining part of your MIDI from *data*,
a path where you want to save output and a time length which you want to append to the beggining. Like

'python3 gan.py data/name.pmg midi/name.midi 100'

