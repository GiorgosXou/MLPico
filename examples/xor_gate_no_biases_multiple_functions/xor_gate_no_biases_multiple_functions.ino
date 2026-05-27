/*
 * ## Double-XOR gate inference example using MLPico.
 * This example runs a pretrained neural-network that approximates:
 *
 *     (A XOR B) XOR C
 *
 * Three deterministic binary inputs are generated sequentially:
 *
 *     000 001 010 011 100 101 110 111
 *
 * The inputs are streamed one bit at a time through get_next_bit(),
 * simulating a simple digital input source.
 *
 * For each 3-bit combination, the network performs feedforward
 * inference and prints the predicted result.
 *
 * This example features:
 *
 * - 0B00001000 No use biases
 * - ACTIVATION__PER_LAYER 
 */

 
#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0])))
#define MLPICO_OPTIMIZE 0B00001000 /* DISABLES BIASES (NO_BIAS) */
#define ACTIVATION__PER_LAYER /* allows the use of multiple activation functions to be defined */
        #define SIGMOID
        #define BINARYSTEP
#define MLPICO_IMPLEMENTATION
#include "input_source.h"
#include <mlpico.h>
#include <stdio.h>


/* activation_function for each layer-to-layer */
const uint8_t activation_functions[] = {
  idx_sigmoid,     /* Idx 0 = Sigmoid    | Layers : 0 -> 1 */
  idx_binarystep,  /* Idx 1 = BinaryStep | Layers : 1 -> 2 */
};


/* Neuron counts per layer: input=3, hidden=4, output=1 */
const unsigned int layers[] = {3, 4, 1};
/* Buffer size = (largest even-indexed layer) + (largest odd-indexed layer) = 3 + 4 */
float buffer[7];
/* Just a semantic alias for buffer */
float *inputs = buffer;
/* Points somewhere inside buffer after the final layer computation (in our case at index 6) */
float *output; 


/* [Pretrained weights] = 3*4 + 4*1 */
const float weights[] = {
   4.75691605f,  4.43128872f,  4.69899607f,
   4.94573259f, -8.66824341f,  4.78460217f,
   9.23950100f, -5.22824812f, -5.60038996f,
  -4.80031824f, -4.46700144f,  8.05480385f,

   4.14872265f,  8.14781570f, -8.66940784f, -8.78521633f,
};


void setup()
{
  unsigned int i = 8; /* (2^3 = 8 possible input combinations.) */
  MLPico mlp = {buffer, activation_functions, weights, layers, 3, NumberOf(layers)}; /* 3 = (largest even-indexed layer) */

  Serial.begin(9600);
  while (!Serial){ }; 

  while(i--){
    inputs[0] = get_next_bit(); /* [A] gets a 0 or 1 */
    inputs[1] = get_next_bit(); /* [B] gets a 0 or 1 */
    inputs[2] = get_next_bit(); /* [C] gets a 0 or 1 */

    /* print current binary input vector */
    Serial.print(inputs[0], 0); Serial.print(" ^ " ); 
    Serial.print(inputs[1], 0); Serial.print(" ^ " ); 
    Serial.print(inputs[2], 0); Serial.print(" ~= "); 

    /* Predict the output and print it. */
    output = feedforward(&mlp);
    Serial.print(*output); Serial.print(" = "); Serial.println((int)(*output));
  }
}


void loop(){}
