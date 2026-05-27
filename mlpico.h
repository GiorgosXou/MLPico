/* 
 *
 * MLPico - Static-allocation MLP inference in ANSI C using 2-slot circular buffer with fixed stride indexing.
 * Created by George Chousos, May 26 2026 based on: https://github.com/GiorgosXou/NeuralNetworks
 *
 *   
 *   
 *   
 *                                             a@L
 *                                          a@@@
 *                                       ,a@@@^
 *                                     ,#@@@P
 *                                   ,@)@@@L
 *                                 ,@"#@@P
 *                               ,#^ #@@`
 *                             ,#^  @@^
 *                            aP   #K
 *        p_                 |@""flW_    ]@@@@@@@@@@@   wJ
 *        `#@@J              |@@@_  "WJa@@@@@@@@@@@^     #@@p
 *          l@@@p            |@@@@@p_a@@@@@@b#@@@F        }W#@p
 *            #@@@@_          "@@@@"#@Bf}]@]@@@P            lpl@@_
 *             "@@@@W_          l@@_    #^s@@P          ,@@  "WJ$@@_
 *               l@@p"W_          lW  s@]@@P          ,@@@     lp}@@@_
 *                '#@@ lW_          ,#}@@P           a@@@@a@@@@p_#p#@@@_
 *                  "@@  "M        a@@@F            ##@@@P" }@@@@@@@#@@@@_
 *                    #@   lp    ,@@^_              #@@"   aF "#@@@@@@@@@@p
 *    s@@@@@ppppp_     |KWWW@@   "            a@@   #@WWWW@     `fBBBfWWWWWW
 *     "@@@@@@@@@@@p aP_  ,@@@             ]@@@^     "W_  #@_
 *       l@@@@$@@@@@@p _a@@@@@           a@@%P         lM  #@p
 *         $@@@}WJlW#@@K|@@@@          a@@}@L J_         lp #@@p
 *          `$@@pl@     #@@"        ,@@@\aF   '@@_         lp"@@@
 *            `$@@J#p  #@"         #@@P]@` _ap #@@W          l@#@@p
 *               l@@}W_          a@@@D#@@@@@@W#@@@@#p          l@@@@_
 *                 "@@#p       a@@@@@@@@@@@Wp   "@@@@            "W@@p
 *                   "W@@_   y@@@@@@@@@@@F   lM___}@@              "f@@_
 *                      ""   '}}}}}____`     ,@""}}#F                 "^
 *                                          a@F  aP
 *                                        ,@@F s@`
 *                                       #@@F]@"
 *                                     /@@@]@"
 *                                   ,@@@@@_
 *                                  a@@@P`
 *                                ,@@@"
 *                               #@P~
 *   
 *   
 */


#ifndef MLPICO_H
    #define MLPICO_H
    #include <math.h>
    #include <stddef.h> /* just for NULL in feedforward_individual */

    #define __MLPICO_VERSION__ "VERSION: 1.0.0"\n
    

    #if defined(ARDUINO)
        #include "Arduino.h"  /* - That gives you access to the standard types and constants of the Arduino language. */
    #else
        #define NATIVE_OR_BARE_MLPICO
        #if defined(__cplusplus)
            #include <cstdint>
        #else /* C */
            #ifndef __STDC__
                #error "Not a standard C compiler."
            #endif
            #if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) || (defined(__has_include) && __has_include(<stdint.h>))
                #include <stdint.h>
            #else /* ##1 https://stackoverflow.com/a/62944893/11465149 */
                typedef signed char int8_t;
                typedef signed short int16_t;
                typedef unsigned char uint8_t;
            #endif
        #endif
    #endif


    /* STR(MSGX) | pragma message */
    #define MSG1
    #define MSG2
    #define MSG3
    #define MSG4
    #define MSG5


    #define DFLOAT_LEN 7
    #define DFLOAT float
    #define IDFLOAT DFLOAT
    #define MEM_SUBSTRATE_MSG /* Memmory substrate message PROGMEM EEPROM etc. */
    #define TYPE_MEMMORY_READ_IDFLOAT(x) (x)
    #define MULTIPLY_BY_INT_IF_QUANTIZATION
    #if defined(MLPICO_OPTIMIZE)
        #if ((MLPICO_OPTIMIZE | 0B01111111) == 0B11111111)
            #undef MSG1
            #undef MEM_SUBSTRATE_MSG
            #if defined(__AVR__) /* NOTE: ##38 */
                #define MSG1 \n- " [0B10000000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] (USE_PROGMEM) (AVR_PROGMEM_LOGIC)."
                #undef TYPE_MEMMORY_READ_IDFLOAT
                #define TYPE_MEMMORY_READ_IDFLOAT(x) pgm_read_float(&x)
                #define AVR_PROGMEM_LOGIC
            #elif defined(NATIVE_OR_BARE_MLPICO)
                #define MSG1 \n- " [0B10000000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] Simply using `const`, (non-AVR) (CONST_MODERN_PROGMEM_LOGIC)."
                #define CONST_MODERN_PROGMEM_LOGIC
            #else
                #define MSG1 \n- " [0B10000000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] Using (non-AVR) (CONST_MODERN_PROGMEM_LOGIC)." 
                #define CONST_MODERN_PROGMEM_LOGIC
            #endif
            /* Prevent non-AVR build issues when PROGMEM is undefined */
            #if !defined(__AVR__) && !defined(PROGMEM)
                #define PROGMEM
            #endif
            #define USE_PROGMEM
            #define MEM_SUBSTRATE_MSG [𝗣𝗥𝗢𝗚𝗠𝗘𝗠]
        #endif

        #if ((MLPICO_OPTIMIZE | 0B10111111) == 0B11111111)
            #include <float.h>
            #if (DBL_MANT_DIG == FLT_MANT_DIG) /* https://stackoverflow.com/questions/8751109 */
                #error "💥 Your device doesn't support 64bit/8byte double-precision !!! | (DBL_MANT_DIG == FLT_MANT_DIG)"
            #endif
            #undef DFLOAT_LEN 
            #undef DFLOAT 
            #define USE_64_BIT_DOUBLE
            #define DFLOAT_LEN 15
            #define DFLOAT double

            #if defined(AVR_PROGMEM_LOGIC) /*  NOTE: that we are safe to undef below here because of ##38 */
                #undef TYPE_MEMMORY_READ_IDFLOAT
                #define TYPE_MEMMORY_READ_IDFLOAT(x) pgm_read_double(&x)
                double pgm_read_double(const double* address) {
                    double result;
                    memcpy_P(&result, address, sizeof(double));
                    return result;
                }
            #endif

            #undef MSG2
            #define MSG2 \n- " [0B01000000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] Using 64bit\8byte double-precision (USE_64_BIT_DOUBLE)."
        #endif  

        #if (((MLPICO_OPTIMIZE | 0B11011111) == 0B11111111) || ((MLPICO_OPTIMIZE | 0B11101111) == 0B11111111)) 
            #if (((MLPICO_OPTIMIZE | 0B11011111) == 0B11111111) && ((MLPICO_OPTIMIZE | 0B11101111) == 0B11111111)) 
                #error "💥 You can't use both int16_t and int8_t! use either 16 or 8."
            #endif
            #if defined(USE_64_BIT_DOUBLE)
                #error "💥 You can't USE_64_BIT_DOUBLE precision with USE_INT_QUANTIZATION."
            #endif
            #undef MSG3
            #undef IDFLOAT
            #undef TYPE_MEMMORY_READ_IDFLOAT 
            #undef MULTIPLY_BY_INT_IF_QUANTIZATION
            #undef CAST_TO_LLONG_IF_NOT_INT_QUANTIZATION
            #if (MLPICO_OPTIMIZE | 0B11011111) == 0B11111111
                #define MSG3 \n- " [0B00100000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] Using int16_t quantization."
                #define IDFLOAT int16_t
                #define TYPE_MEMMORY_READ_IDFLOAT(x) pgm_read_int16_t(&x)
                #define TYPE_MEMMORY_READ_IDFLOAT_NAME pgm_read_int16_t
            #else
                #define MSG3 \n- " [0B00010000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] Using int8_t quantization."
                #define IDFLOAT int8_t
                #define TYPE_MEMMORY_READ_IDFLOAT(x) pgm_read_int8_t(&x)
                #define TYPE_MEMMORY_READ_IDFLOAT_NAME pgm_read_int8_t
            #endif
            #define USE_INT_QUANTIZATION

            #if defined(AVR_PROGMEM_LOGIC)
                IDFLOAT TYPE_MEMMORY_READ_IDFLOAT_NAME(const IDFLOAT* address) {
                    IDFLOAT result;
                    memcpy_P(&result, address, sizeof(IDFLOAT));
                    return result;
                }
            #else /* we redefine macro-logic back to the default */
                #undef TYPE_MEMMORY_READ_IDFLOAT_NAME
                #undef TYPE_MEMMORY_READ_IDFLOAT
                #define TYPE_MEMMORY_READ_IDFLOAT(x) (x)
            #endif

            /* FLOAT RANGE = (100.0) - (-100.0) | MAX - MIN */
            /* INT   RANGE = (32767) - (-32768) | MAX - MIN */
            #if (MLPICO_OPTIMIZE | 0B11011111) == 0B11111111
                #if !defined(Q_FLOAT_RANGE)
                    #define Q_FLOAT_RANGE 200.0
                #endif
                #define Q_INT_RANGE 65535
            #else
                #if !defined(Q_FLOAT_RANGE)
                    #define Q_FLOAT_RANGE 51.0
                #endif
                #define Q_INT_RANGE 255
            #endif
            #define MULTIPLY_BY_INT_IF_QUANTIZATION * ((DFLOAT)Q_FLOAT_RANGE/Q_INT_RANGE)

            /*
            #undef DEFAULT_LEARNING_RATE_OF_WEIGHTS
            #undef DEFAULT_LEARNING_RATE_OF_BIASES
            #define DEFAULT_LEARNING_RATE_OF_WEIGHTS ((IDFLOAT)((Q_INT_RANGE * 0.33)/Q_FLOAT_RANGE))
            #define DEFAULT_LEARNING_RATE_OF_BIASES  ((IDFLOAT)((Q_INT_RANGE * 0.11)/Q_FLOAT_RANGE))
            */
        #endif

        #if ((MLPICO_OPTIMIZE | 0B11110111) == 0B11111111)
            #undef MSG4
            #undef OPTIONAL_BIAS
            #define OPTIONAL_BIAS(x)
            #define MSG4 \n- " [0B00001000] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] Biases are disabled (NO_BIAS)."
            #define HAS_NO_BIAS 1
            #define NO_BIAS
        #endif

        #if ((MLPICO_OPTIMIZE | 0B11111011) == 0B11111111)
            #if defined(NO_BIAS)
                #error "💥 You can't have both NO_BIAS and MULTIPLE_BIASES_PER_LAYER."
            #endif
            #undef MSG5
            #define MSG5 \n- " [0B00000100] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] You are using (MULTIPLE_BIASES_PER_LAYER)."
            #define MULTIPLE_BIASES_PER_LAYER
            #undef OPTIONAL_MULTI_BIAS
            #define OPTIONAL_MULTI_BIAS(x) , x
        #endif
    #endif


    /* Keyword-based NN optimizations */
    /** Check if the NN has only a SINGLE_BIAS_PER_LAYER **/
    #if !defined(NO_BIAS) && !defined(MULTIPLE_BIASES_PER_LAYER)
        #define SINGLE_BIAS_PER_LAYER
        #undef MSG5
        #define MSG5 \n- " [0B0000XX00] [𝗥𝗲𝗺𝗶𝗻𝗱𝗲𝗿] (default) You are using a (SINGLE_BIAS_PER_LAYER)."
    #endif



    #define ACT1  0
    #define ACT2  0
    #define ACT3  0
    #define ACT4  0
    #define ACT5  0
    #define ACT6  0
    #define ACT7  0
    #define ACT8  0
    #define ACT9  0
    #define ACT10 0
    #define ACT11 0
    #define ACT12 0
    #define ACT13 0
    #define ACT14 0

    /* STR(AX) | pragma message | AN = Activation Name | ALL_A = All | CA = Custom Activation */
    #define AN_1
    #define AN_2
    #define AN_3
    #define AN_4
    #define AN_5
    #define AN_6
    #define AN_7
    #define AN_8
    #define AN_9
    #define AN_10
    #define AN_11
    #define AN_12
    #define AN_13
    #define AN_14
    #define ALL_A


    /* Switch cases for ACTIVATION__PER_LAYER */
    #define CASE_RETURN_SIGMOID(x)
    #define CASE_RETURN_TANH(x)
    #define CASE_RETURN_RELU(x)
    #define CASE_RETURN_LEAKYRELU(x)
    #define CASE_RETURN_ELU(x)
    #define CASE_RETURN_SELU(x)
    #define CASE_RETURN_SOFTMAX(x)
    #define CASE_RETURN_IDENTITY(x)
    #define CASE_RETURN_BINARYSTEP(x)
    #define CASE_RETURN_SOFTPLUS(x)
    #define CASE_RETURN_SILU(x)
    #define CASE_RETURN_GELU(x)
    #define CASE_RETURN_MISH(x)
    #define CASE_RETURN_GAUSSIAN(x)


    #if defined(SIGMOID) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_SIGMOID
        #undef AN_1
        #undef ACT1
        #undef SIGMOID
        #define ACT1 1
        #define ACTIVATION 
        #define ACTIVATION_FUNCTION sigmoid
        #define CASE_RETURN_SIGMOID(x) case idx_sigmoid: return sigmoid(x)
        #define SIGMOID Sigmoid
        #define AN_1 |> Sigmoid 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT sigmoid(const DFLOAT x){ return 1 / (1 + exp(-x)); }
        #endif
    #endif
    #if defined(TANH) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_TANH
        #undef AN_2
        #undef ACT2
        #undef TANH
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT2 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION tanh_
        #define CASE_RETURN_TANH(x) case idx_tanh: return tanh_(x)
        #define TANH Tanh
        #define AN_2 |> Tanh 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT tanh_ (const DFLOAT x) { return (exp(2*x) - 1) / (exp(2*x) + 1); }
        #endif
    #endif
    #if defined(RELU) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_RELU
        #undef AN_3
        #undef ACT3
        #undef RELU
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT3 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION relu
        #define CASE_RETURN_RELU(x) case idx_relu: return relu(x)
        #define SUPPORTS_CLIPPING /* i mean  "supports" / usually-needs  ? */
        #define RELU ReLU
        #define AN_3 |> ReLU 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT relu (const DFLOAT x) { return (x > 0) ? x : 0; }
        #endif
    #endif
    #if defined(LEAKYRELU) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_LEAKYRELU
        #undef AN_4
        #undef ACT4
        #undef LEAKYRELU
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT4 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION leakyrelu
        #define CASE_RETURN_LEAKYRELU(x) case idx_leakyrelu: return leakyrelu(x)
        #define SUPPORTS_CLIPPING /* i mean  "supports" / usually-needs  ? */
        #define LEAKYRELU LeakyReLU
        #define AN_4 |> LeakyReLU 
        extern DFLOAT AlphaLeaky;
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT AlphaLeaky = 0.01;
            DFLOAT leakyrelu (const DFLOAT x) { return (x > 0) ? x : AlphaLeaky * x; }
        #endif
    #endif
    #if defined(ELU) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_ELU
        #undef AN_5
        #undef ACT5
        #undef ELU
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT5 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION elu
        #define CASE_RETURN_ELU(x) case idx_elu: return elu(x)
        #define SUPPORTS_CLIPPING /* i mean  "supports" / usually-needs  ? */
        #define ELU ELU
        #define AN_5 |> ELU 
        extern DFLOAT AlphaELU;
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT AlphaELU = 1.0;
            DFLOAT elu (const DFLOAT x) { return (x > 0) ? x : AlphaELU  * (exp(x) - 1); }
        #endif
    #endif
    #if defined(SELU) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_SELU
        #undef AN_6
        #undef ACT6
        #undef SELU
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT6 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION selu
        #define CASE_RETURN_SELU(x) case idx_selu: return selu(x)
        #define SELU SELU
        #define AN_6 |> SELU 
        extern DFLOAT alpha_selu;
        extern DFLOAT lamda_selu;
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT alpha_selu = 1.67326324;
            DFLOAT lamda_selu = 1.05070098;
            DFLOAT selu (const DFLOAT x){ return (x >= 0) ? x * lamda_selu : lamda_selu * alpha_selu * (exp(x) - 1);}
        #endif
    #endif
    #if defined(SOFTMAX) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #error "💥 Softmax not supported yet!"
        #undef CASE_RETURN_SOFTMAX
        #undef AN_7
        #undef ACT7
        #undef SOFTMAX
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT7 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION softmax
        #define CASE_RETURN_SOFTMAX(x) case idx_softmax: return softmax(x)
        #define SOFTMAX Softmax
        #define AN_7 |> Softmax 
    #endif
    #if defined(IDENTITY) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_IDENTITY
        #undef AN_8
        #undef ACT8
        #undef IDENTITY
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT8 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION identity
        #define CASE_RETURN_IDENTITY(x) case idx_identity: return identity(x)
        #define IDENTITY Identity
        #define AN_8 |> Identity 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT identity (const DFLOAT x) { return x; }
        #endif
    #endif
    #if defined(BINARYSTEP) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_BINARYSTEP
        #undef AN_9
        #undef ACT9
        #undef BINARYSTEP
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT9 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION binarystep
        #define CASE_RETURN_BINARYSTEP(x) case idx_binarystep: return binarystep(x)
        #define BINARYSTEP BinaryStep
        #define AN_9 |> BinaryStep 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT binarystep (const DFLOAT x) { return (x < 0) ? 0 : 1; }
        #endif
    #endif
    #if defined(SOFTPLUS) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_SOFTPLUS
        #undef AN_10
        #undef ACT10
        #undef SOFTPLUS
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT10 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION softplus
        #define CASE_RETURN_SOFTPLUS(x) case idx_softplus: return softplus(x)
        #define SOFTPLUS Softplus
        #define AN_10 |> Softplus 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT softplus (const DFLOAT x) { return log(1 + exp(x)); }
        #endif
    #endif
    #if defined(SILU) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_SILU
        #undef AN_11
        #undef ACT11
        #undef SILU
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT11 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION silu
        #define CASE_RETURN_SILU(x) case idx_silu: return silu(x)
        #define SILU SiLU
        #define AN_11 |> SiLU 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT silu (const DFLOAT x) { return x / (1 + exp(-x)); }
        #endif
    #endif
    #if defined(GELU) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_GELU
        #undef AN_12
        #undef ACT12
        #undef GELU
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT12 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION gelu
        #define CASE_RETURN_GELU(x) case idx_gelu: return gelu(x)
        #define GELU GELU
        #define AN_12 |> GELU 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT gelu (const DFLOAT x) { return (1/2) * x * (1 + erf(x / sqrt(x))); }
        #endif
    #endif
    #if defined(MISH) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_MISH
        #undef AN_13
        #undef ACT13
        #undef MISH
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT13 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION mish
        #define CASE_RETURN_MISH(x) case idx_mish: return mish(x)
        #define MISH Mish
        #define AN_13 |> Mish 
        #if defined(MLPICO_IMPLEMENTATION)
            #if !defined(TANH)
                DFLOAT tanh_ (const DFLOAT x) { return (exp(2*x) - 1) / (exp(2*x) + 1); }
            #endif
            DFLOAT mish (const DFLOAT x) { return x * tanh(log(1 + exp(x))); }
        #endif
    #endif
    #if defined(GAUSSIAN) && (defined(ACTIVATION__PER_LAYER) || !defined(ACTIVATION))
        #undef CASE_RETURN_GAUSSIAN
        #undef AN_14
        #undef ACT14
        #undef GAUSSIAN
        #undef ACTIVATION
        #undef ACTIVATION_FUNCTION
        #define ACT14 1
        #define ACTIVATION
        #define ACTIVATION_FUNCTION gaussian
        #define CASE_RETURN_GAUSSIAN(x) case idx_gaussian: return gaussian(x)
        #define GAUSSIAN Gaussian
        #define AN_14 |> Gaussian 
        #if defined(MLPICO_IMPLEMENTATION)
            DFLOAT gaussian (const DFLOAT x) { return exp(-(x*x)); }
        #endif
    #endif


    /* vvv Meaning that if not exclusivly defined one (or more) ACTIVATION function */
    #if !defined(ACTIVATION)
        #if defined(ACTIVATION__PER_LAYER)
            /* ACTIVATE ALL FUNCTIONS*/
            #error "💥 ALL_ACTIVATION_FUNCTIONS not supported yet! Please #define the ones you need."
            #define ALL_ACTIVATION_FUNCTIONS
            #undef ALL_A
            #define ALL_A |> "(ALL_ACTIVATION_FUNCTIONS)"
            #undef NUM_OF_USED_ACTIVATION_FUNCTIONS
            #define NUM_OF_USED_ACTIVATION_FUNCTIONS (13) 
            /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ + CACT1 + CACT2 + CACT3 + CACT4 + CACT5) */
            /* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ they are 14 with sotmax but removed */
        #else
            /* ENABLE DEFAULT ACTIVATION FUNCTION */
            /* i will also create a mechanism to show #error if more than one is defined with B opperations?*/
            #undef AN_1
            /* #undef SIGMOID */
            #define ACTIVATION /* Sigmoid default but for more than one you must declare it */
            #define ACTIVATION_FUNCTION sigmoid
            /* not needed #define CASE_RETURN_SIGMOID case idx_sigmoid: return sigmoid */
            #define SIGMOID Sigmoid
            #define AN_1 |> Sigmoid 
            #if defined(MLPICO_IMPLEMENTATION)
                DFLOAT sigmoid(const DFLOAT x){ return 1 / (1 + exp(-x));}
            #endif
        #endif
    #endif

    #define ACTIVATE_WITH(actname,value) actname(value)

    #if defined(ACTIVATION__PER_LAYER)
        #undef ACTIVATION_FUNCTION
        #undef ACTIVATE_WITH
        #define ACTIVATE_WITH(actname_idx,value) __activation_with(actname_idx,value)
        #define ACTIVATION_FUNCTION mlp->acts[n-1]
    #endif


    #if !defined(SUPPRESS_PRAGMA_MESSAGE)
        #define STR_HELPER(x) #x
        #define STR(x) STR_HELPER(x)

        #define INFORMATION \n MLPico __MLPICO_VERSION__ MSG1 MSG2 MSG3 MSG4 MSG5 \n\n 𝗨𝗦𝗜𝗡𝗚 MEM_SUBSTRATE_MSG [ƒx] ALL_A AN_1 AN_2 AN_3 AN_4 AN_5 AN_6 AN_7 AN_8 AN_9 AN_10 AN_11 AN_12 AN_13 AN_14 /* CSTA CA1 CA2 CA3 CA4 CA5 */
        #pragma message( STR(INFORMATION) )
    #endif


    enum
    {
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(SIGMOID)
            idx_sigmoid,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(TANH)
            idx_tanh,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(RELU)
            idx_relu,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(LEAKYRELU)
            idx_leakyrelu,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(ELU)
            idx_elu,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(SELU)
            idx_selu,
        #endif
        /*
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(SOFTMAX)
            idx_softmaxsum,
        #endif
        */
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(IDENTITY)
            idx_identity,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(BINARYSTEP)
            idx_binarystep,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(SOFTPLUS)
            idx_softplus,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(SILU)
            idx_silu,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(GELU)
            idx_gelu,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(MISH)
            idx_mish,
        #endif
        #if defined(ALL_ACTIVATION_FUNCTIONS) || defined(GAUSSIAN)
            idx_gaussian,
        #endif

        /*
        #if defined(CUSTOM_AF1)
            idx_ ## CUSTOM_AF1,
        #endif
        #if defined(CUSTOM_AF2)
            idx_ ## CUSTOM_AF2,
        #endif
        #if defined(CUSTOM_AF3)
            idx_ ## CUSTOM_AF3,
        #endif
        #if defined(CUSTOM_AF4)
            idx_ ## CUSTOM_AF4,
        #endif
        #if defined(CUSTOM_AF5)
            idx_ ## CUSTOM_AF5,
        #endif
        */
    };


    typedef struct
    {
        DFLOAT* io_buf;
        #if defined(ACTIVATION__PER_LAYER)
            const uint8_t *acts;
        #endif
        #if !defined(NO_BIAS)
            const IDFLOAT* biases;
        #endif
        const IDFLOAT* weights;
        const unsigned int* layers;
        const unsigned int largest_input;
        const uint8_t number_of_layers;
    }
    MLPico;


    /* Main Functions */
    DFLOAT*   feedforward           (const MLPico* mlp);
    DFLOAT* __feedforward           (const MLPico* mlp, unsigned int n, unsigned int nij, unsigned int ni);
    DFLOAT*   feedforward_individual(const MLPico* mlp, const DFLOAT _input, unsigned int _j);
    DFLOAT  __activation_with       (const uint8_t activation_idx, const DFLOAT output);
    /* TODO: void      mlpico_print          (const MLPico* mlp); */


    /* Activation Functions */
    DFLOAT sigmoid    (const DFLOAT x);
    DFLOAT tanh_      (const DFLOAT x); /* prevent conflicts with math.h */
    DFLOAT relu       (const DFLOAT x);
    DFLOAT leakyrelu  (const DFLOAT x);
    DFLOAT elu        (const DFLOAT x);
    DFLOAT selu       (const DFLOAT x);
    DFLOAT identity   (const DFLOAT x);
    DFLOAT binarystep (const DFLOAT x);
    DFLOAT softplus   (const DFLOAT x);
    DFLOAT silu       (const DFLOAT x);
    DFLOAT gelu       (const DFLOAT x);
    DFLOAT mish       (const DFLOAT x);
    DFLOAT gaussian   (const DFLOAT x);


    #if defined(MLPICO_IMPLEMENTATION)

        DFLOAT __activation_with(const uint8_t activation_idx, const DFLOAT output){
            switch (activation_idx) {
                CASE_RETURN_SIGMOID    (output);
                CASE_RETURN_TANH       (output);
                CASE_RETURN_RELU       (output);
                CASE_RETURN_LEAKYRELU  (output);
                CASE_RETURN_ELU        (output);
                CASE_RETURN_SELU       (output);
                CASE_RETURN_SOFTMAX    (output); /* <--- Error: 💥 not implemented yet */
                CASE_RETURN_IDENTITY   (output);
                CASE_RETURN_BINARYSTEP (output);
                CASE_RETURN_SOFTPLUS   (output);
                CASE_RETURN_SILU       (output);
                CASE_RETURN_GELU       (output);
                CASE_RETURN_MISH       (output);
                CASE_RETURN_GAUSSIAN   (output);
            }
        }


        DFLOAT* __feedforward(const MLPico* mlp, unsigned int n, unsigned int nij, unsigned int ni){
            unsigned int i = 0, j = 0;
            uint8_t shift = 1; /* "bool" | typedef for c89 see #1 */
            DFLOAT *input, *output;

            /* `do{...}while()` instead, should reduce sketch size i think */
            for (; n < mlp->number_of_layers; ++n) {

                output = (mlp->io_buf + ( shift * mlp->largest_input));
                input  = (mlp->io_buf + (!shift * mlp->largest_input));

                for (i = 0; i < mlp->layers[n]; i++){

                    #if defined(NO_BIAS)
                        output[i] = 0;
                    #elif defined(MULTIPLE_BIASES_PER_LAYER)
                        output[i] = TYPE_MEMMORY_READ_IDFLOAT(mlp->biases[ni++]);
                    #else /* Single bias */
                        output[i] = TYPE_MEMMORY_READ_IDFLOAT(mlp->biases[n-1]);
                    #endif

                    for (j = 0; j < mlp->layers[n-1]; j++)
                        output[i] += input[j] * TYPE_MEMMORY_READ_IDFLOAT(mlp->weights[nij++]);
                    output[i] = ACTIVATE_WITH(ACTIVATION_FUNCTION, (output[i] MULTIPLY_BY_INT_IF_QUANTIZATION));
                }

                shift = !shift;
            }
            return output;
        }


        DFLOAT* feedforward(const MLPico* mlp){
            return __feedforward(mlp, 1, 0, 0); /* Compilers are smart enough to inline this when needed :P */
        }


        /* WARN: you need to call feedforward_individual with _j = 0 the first time */
        DFLOAT* feedforward_individual(const MLPico* mlp, const DFLOAT _input, const unsigned int _j){
            unsigned int n = 1, n0_ij = 0, i = 0, j = 0; /* dummy n for ACTIVATION__PER_LAYER and mlp->acts[n-1] */
            DFLOAT *output = mlp->io_buf;
            
            for (i = 0; i < mlp->layers[1]; i++){
                if (!_j)
                    #if defined(NO_BIAS)
                        output[i] = 0;
                    #elif defined(MULTIPLE_BIASES_PER_LAYER)
                        output[i] = TYPE_MEMMORY_READ_IDFLOAT(mlp->biases[i]);
                    #else /* Single bias */
                        output[i] = TYPE_MEMMORY_READ_IDFLOAT(*mlp->biases);
                    #endif

                output[i] += _input * TYPE_MEMMORY_READ_IDFLOAT(mlp->weights[n0_ij+_j]);
                n0_ij += mlp->layers[0];

                if ( _j == mlp->layers[0] - 1)
                    output[i] = ACTIVATE_WITH(ACTIVATION_FUNCTION, (output[i] MULTIPLY_BY_INT_IF_QUANTIZATION));
            }

            if ( _j == mlp->layers[0] - 1)
                return __feedforward(mlp, 2, n0_ij, i);

            return NULL; 
        }


        /* TODO: mlpico_print(MLPico *mlp)
        *  void mlpico_print(const MLPico* mlp){
        *      unsigned int i = 0 , j = 0;
        *  }
        */

    #endif /* MLPICO_IMPLEMENTATION */
#endif /* MLPICO_H */
