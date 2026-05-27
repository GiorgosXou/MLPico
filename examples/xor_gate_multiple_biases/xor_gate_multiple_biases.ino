/*
 * ## Double-XOR gate inference example using MLPico. (with biases)
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
 */

 
#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0])))
#define MLPICO_OPTIMIZE 0B10000100 /* ENABLES MULTIPLE_BIASES_PER_LAYER */
#define SIGMOID /* Defines the core activation-function used by MLPico internaly */
#define MLPICO_IMPLEMENTATION
#include "input_source.h"
#include <mlpico.h>
#include <stdio.h>


/* Neuron counts per layer: input=3, hidden=4, output=1 */
const unsigned int layers[] = {3, 4, 1};
/* Buffer size = (largest even-indexed layer) + (largest odd-indexed layer) = 3 + 4 */
float buffer[7];
/* Just a semantic alias for buffer */
float *inputs = buffer;
/* Points somewhere inside buffer after the final layer computation (in our case at index 6) */
float *output; 


/* [Pretrained biases] = 4 + 1 */
const float biases[] = {
  -2.32293916f,  6.85155153f,  3.65815043f,  2.60234070f,
   0.94495600f,
};

/* [Pretrained weights] = 3*4 + 4*1 */
const float weights[] = {
   5.62609529f, -6.19054079f,  5.17951250f,
  -4.61203289f, -5.11720181f, -4.63304090f,
   7.65558243f, -7.26023722f, -7.62205410f,
  -6.11576033f, -5.18105030f,  6.10605335f,

   5.92340851f,  6.10771608f, -6.76925325f, -6.76961088f,
};


void setup()
{
  unsigned int i = 8; /* (2^3 = 8 possible input combinations.) */
  MLPico mlp = {buffer, biases, weights, layers, 3, NumberOf(layers)}; /* 3 = (largest even-indexed layer) */

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
    Serial.print(*output, 7); Serial.print(" = "); Serial.println((int)(*output + 0.1));
  }
}


void loop(){}
