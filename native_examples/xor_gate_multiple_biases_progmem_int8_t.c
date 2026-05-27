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
 * - 0B10000000 PROGMEM compatibility for AVR MCUs
 * - 0B00010000 Multipple biases 
 * - 0B00000100 int8_t quantization
 */

 
#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0])))
#define MLPICO_OPTIMIZE 0B10010100 /* ENABLES progmem when AVR + int8_t quantization + MULTIPLE_BIASES_PER_LAYER */
#define Q_FLOAT_RANGE 51.0 /* is a float-range of -25.5 +25.5 mapped between -128 and +127 int8_t */
#define SIGMOID /* Defines the core activation-function used by MLPico internaly */
#define MLPICO_IMPLEMENTATION
#include "input_source.h"
#include "./../mlpico.h"
#include <stdio.h>


/* Neuron counts per layer: input=3, hidden=3, output=1 */
const unsigned int layers[] = {3, 3, 1};
/* Buffer size = (largest even-indexed layer) + (largest odd-indexed layer) = 3 + 3 */
float buffer[6];
/* Just a semantic alias for buffer */
float *inputs = buffer;
/* Points somewhere inside buffer after the final layer computation (in our case at index 5) */
float *output; 


/* [Pretrained int8_t quantized biases] = 3 + 1 */
const PROGMEM int8_t biases[] = {
  -49, -23, 22,
   29,
};

/* [Pretrained int8_t quantized weights] = 3*3 + 3*1 */
const PROGMEM int8_t weights[] = {
   25,  49,  26,
  -33,  32,  48,
  -48, -33,  35,

  -60,  61, -60,
};


int main(int argc, char *argv[])
{
  unsigned int i = 8; /* (2^3 = 8 possible input combinations.) */
  MLPico mlp = {buffer, biases, weights, layers, 3, NumberOf(layers)}; /* 3 = (largest even-indexed layer) */

  while(i--){
    inputs[0] = get_next_bit(); /* [A] gets a 0 or 1 */
    inputs[1] = get_next_bit(); /* [B] gets a 0 or 1 */
    inputs[2] = get_next_bit(); /* [C] gets a 0 or 1 */

    /* print current binary input vector */
    printf("%1.f ^ %1.f ^ %1.f ~= ", inputs[0], inputs[1], inputs[2]);

    /* Predict the output and print it. */
    output = feedforward(&mlp);
    printf("%f = %d \n", *output, (int)(*output + 0.1));
  }

  return 0;
}
