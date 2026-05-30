## MLPico - ANSI C (Multi-Layer Perceptron)

A general-purpose header-only statically-allocatable [MLP *(Multi-Layer Perceptron)*](https://en.wikipedia.org/wiki/Multilayer_perceptron) implementation\\library written in **ANSI C** that utilizes a 2-slot circular-buffer with fixed stride indexing, for the process of feedforward\\prediction<b>;</b> a technique that provides optimal speed and memory efficiency without compromises across most systems. 


## Features

1. Supports PROGMEM for AVR MCUs.
2. Supports int-quantized weights.
3. Blazingly fast and portable.
4. Supports Arduino, too.
5. Memory-efficient.
6. Simple in use.


## Usage

See examples, [for arduino](https://github.com/GiorgosXou/MLPico/tree/main/examples) or [native-os\\bare-metal](https://github.com/GiorgosXou/MLPico/tree/main/native_examples).


## Training
Here's a minimal example showing step-by-step how to train an MLP neural network to emulate a basic digital logic circuit implementing a double XOR gate, defined as ((A⊕B)⊕C):

``` python
# A generic "boilerplate"/"template" for training and generating a complete MLPico compatible MLP Neural Network.
# pip install tensorflow

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import numpy as np
import random


# Reproducibility
random.seed(23)
np.random.seed(41)
tf.random.set_seed(21)

# =========================================================
#   Define <-- with what you want, aka. your preferences. 
# =========================================================

USE_BIASES   = True  # <-- Use biases or not
USE_DOUBLE   = False # <-- Use 64 or 32 bit-precision aka. `float` or `double`
USE_PROGMEM  = True  # <-- Allow PROGMEM compatibility when used with AVR MCUs
QUANTIZATION = ""    # <-- Use "int16_t" or "int8_t" if you want quantization

INT8Q  = 0
INT16Q = 0
if QUANTIZATION: # 💥 Yes I like to use emojis! it's not ai-slop, deal with it. 
    if not QUANTIZATION in ('int16_t', 'int8_t'): # Why suddenly emojis are bad!? wtf.
        raise TypeError("💥 QUANTIZATION !== `int16_t` or `int8_t`") # ^^^^^^^^^^^^^^^
    INT8Q      = 0 if QUANTIZATION == 'int16_t' else 1
    INT16Q     = int(not INT8Q)
    INT_LIMIT  = 32768 * INT16Q + 128 * INT8Q
    INT_RANGE  = 65535 * INT16Q + 255 * INT8Q #     int8_t -128 to 127 mapped to the float range bellow.
    FP__RANGE  = 200.0 * INT16Q + 51  * INT8Q # <-- 51 = -25.5 to 25.5 float range, feel free to change.
    INT_LENGTH = len(str(INT_RANGE))

# =========================================================

# Setting up a few stuff about precision and int-quantization.
DFLOAT = 'float'
PRECISION = 7
if USE_DOUBLE:
    if QUANTIZATION: 
        raise Exception("💥 You can't use int-quantization with double-precision") 
    DFLOAT = 'double'
    PRECISION = 14
else:
    tf.keras.backend.set_floatx('float32')
IDFLOAT = QUANTIZATION if QUANTIZATION else DFLOAT


# Functions to convert float-to-int & int-to-float
def quantize_float32_to_int(w):
    S = (FP__RANGE) / (INT_RANGE)
    return round(w / S) # + Z

def int_to_float32(q):
    return np.float32(np.float32(FP__RANGE) / (INT_RANGE)) * np.float32(q)


# <-- (Training data) - Define the XOR gate inputs and outputs
inputs  = np.array([
    [ 0, 0, 0 ], 
    [ 0, 0, 1 ], 
    [ 0, 1, 0 ], 
    [ 0, 1, 1 ], 
    [ 1, 0, 0 ], 
    [ 1, 0, 1 ], 
    [ 1, 1, 0 ], 
    [ 1, 1, 1 ]
], dtype = np.float32)
outputs = np.array([[0], [1], [1], [0], [1], [0], [0], [1]], dtype = np.float32)

# <-- Create a simple MLP neural network
model = tf.keras.Sequential([
    Input(shape=(3,)), # 3 Inputs/features (no biases) 
    Dense(4, activation='sigmoid', use_bias=USE_BIASES), # Dense  4 units\neurons
    Dense(1, activation='sigmoid', use_bias=USE_BIASES)  # Output 1 units\neurons
])

# Disaplay a summary of the MLP structure
model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.091) # <--
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) # <--

# Train the model
model.fit(inputs, outputs, epochs=300, verbose=1) # <--

# Evaluate the model on the training data
loss, accuracy = model.evaluate(inputs, outputs)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Predict XOR gate outputs
predictions = model.predict(inputs)
print("Predictions:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Predicted Output: {predictions[i][0]:.7f}")


# Get input and output layers with the largest amount of neurons
layers = [model.input_shape[1]] + [layer.units for layer in model.layers]
max_in  = 0 # largest even-indexed layer
max_out = 0 # largest  odd-indexed layer
inout = True
for layer in layers:
    if inout:
        if layer > max_in: max_in = layer
    else:
        if layer > max_out: max_out = layer
    inout = not inout


# Just printing stuff
print()
weights_biases = model.get_weights()
print("#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0])))")

print(
    f"#define MLPICOPTIMIZE 0B{int(USE_PROGMEM)}{int(USE_DOUBLE)}{INT16Q}{INT8Q}{int(not USE_BIASES)}{int(USE_BIASES)}00" +
    " /* ENABLES " +
    ("(USE_PROGMEM) "                  if     USE_PROGMEM else "") +
    ("(USE_64_BIT_DOUBLE) "            if     USE_DOUBLE  else "") +
    ("(USE_INT_QUANTIZATION int16_t) " if     INT16Q      else "") +
    ("(USE_INT_QUANTIZATION int8_t) "  if     INT8Q       else "") +
    ("(MULTIPLE_BIASES_PER_LAYER) "    if     USE_BIASES  else "") +
    ("(NO_BIAS) "                      if not USE_BIASES  else "") +
    "*/"
)

# functions to handle each case
if not QUANTIZATION:
    def handle_weight(w, i, j): print(f"{w[i][j]:{PRECISION+3}.{PRECISION}f}f", end=", ")
    def handle_bias  (b,    j): print(   f"{b[j]:{PRECISION+3}.{PRECISION}f}f", end=", ")
else:
    def handle_weight(w, i, j): 
        x = quantize_float32_to_int(w[i][j])
        w[i][j] = int_to_float32(x)
        print(f"{x:{INT_LENGTH}d}", end=", ")
        if x >= INT_LIMIT or x < -INT_LIMIT:
            raise RuntimeError(f"💥 {QUANTIZATION} weight[{i}][{j}]={x} exceeds INT_RANGE={INT_RANGE}. Please adjust FP__RANGE={FP__RANGE} or clip weights to a limit.") 

    def handle_bias(b, j): 
        x = quantize_float32_to_int(b[j])
        b[j] = int_to_float32(x)
        print(f"{x:{INT_LENGTH}d}", end=", ")
        if x >= INT_LIMIT or x < -INT_LIMIT:
            raise RuntimeError(f"💥 {QUANTIZATION} bias[{j}]={x} exceeds INT_RANGE={INT_RANGE}. Please adjust FP__RANGE={FP__RANGE} or clip biases to a limit.") 

    hlfpr = FP__RANGE/2
    print(f"#define Q_FLOAT_RANGE {FP__RANGE} /* is a float-range of {hlfpr} to -{hlfpr} mapped to int{8 + 8*INT16Q}_t */")


print("#define MLPICO_IMPLEMENTATION /* Defines the implementation inside the header file */")


# check if model has more than one activation function
acts = {}
list_acts = []
for layer in model.layers[0:]:
    l = layer.activation.__name__.replace('_', '')
    acts[l.upper()] = 1
    list_acts.append(l)

# define them.
if len(acts) > 1:
    print("/* Actiavtion functions per layer-to-layer */")
    print("#define ACTIVATION__PER_LAYER /* allows the use of multiple activation functions to be defined */")
    for act in acts:
        print(f"    #define {act} /* Defines the core activation function used internally by MLPico */")
elif not acts.get("SIGMOID"): # sigmoid is mlpico default 
    print(f"#define {next(iter(acts))}")


print("#include <mlpico.h>\n\n")

print(f"unsigned int layers[] = " + "{" + ",".join(map(str, layers)) + "};")
print(f"{DFLOAT} buffer[{max_in + max_out}];")
print(f"{DFLOAT} *inputs = buffer;")
print(f"{DFLOAT} *output;\n")

if len(acts) > 1:
    print("const uint8_t activation_functions[] = {")
    for act in list_acts:
        print(f"  idx_{act}")
    print("}\n")

if USE_BIASES:
    print("/* Pretrained biases */")
    print(f"const {'PROGMEM ' if USE_PROGMEM else ''}{IDFLOAT} biases[] = {{")
    for l, (w, b) in enumerate(zip(weights_biases[::2], weights_biases[1::2])):
        print('  ', end='')
        for j in range(0, w.shape[1]):
            handle_bias(b, j)
        print()
    print('};\n')

shift = int(not USE_BIASES) 
print("/* Pretrained weights */")
print(f"const {'PROGMEM ' if USE_PROGMEM else ''}{IDFLOAT} weights[] = {{", end="")
for l, (w, b) in enumerate(zip(weights_biases[::2-shift], weights_biases[1-shift::2-shift])):
    print()
    for j in range(0, w.shape[1]):
        print('  ', end='')
        for i in range(0, w.shape[0]):
            handle_weight(w, i, j)
        print()
print('};\n')


print(
    "MLPico mlp = {" + "buffer," + (" biases," if USE_BIASES else "") + " weights, layers, " +
    f"{max_in}, NumberOf(layers)}}; /* {max_in} = largest even-indexed layer */ "
)
print("mlpico_print(&mlp);")


if not QUANTIZATION: exit()

# Load quantized weights for NN evaluation
model.set_weights(weights_biases)

# Evaluate the model on the training data again after quantization
print()
loss, accuracy = model.evaluate(inputs, outputs)
print(f"Quantized Model accuracy: {accuracy * 100:.2f}%")

# Predict XOR gate outputs
predictions = model.predict(inputs)
print("Predictions after quantizatioon:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Predicted Output: {predictions[i][0]:.7f}")
```


## Donate
May love shine light to your heart. 

- Paypal: https://www.paypal.com/donate/?hosted_button_id=3FUJ9ZQSM2VQW
- Monero: <sub><sup>`89pdRfcusZr39HNFXEToN38ArKtm7y4o6aHhRfdtW75hMuv3h3QT2TPWZ5F2kBAqRPTxFfCQBtbKSMErcG6PMzeu31XU4aL`</sup></sub>


## Outro
If you need something more feature-complete and [powerful](https://github.com/GiorgosXou/ATTiny85-MNIST-RNN-EEPROM); I then recommend you to take a *(non-judgmental)* look into my [NeuralNetworks library](https://github.com/GiorgosXou/NeuralNetworks) *(that I based MLPico on)*.


<i><sub> (This library is not vibe coded. [Here's the proof](https://github.com/GiorgosXou/NeuralNetworks/commit/4d1f3205afb7f5cbc5378e6043344151c52c9cea) that I'm developing MLP algorithms since 2019; way before the AI-slop epidemic. and [here](https://github.com/GiorgosXou/NeuralNetworks#-research)'s the whole research behind it) </sub></i>

