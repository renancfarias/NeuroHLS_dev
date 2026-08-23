#pragma once

#include "bit_type.h"

// A flat tile can cross a row boundary when LANES does not divide WIDTH.
// Keeping such tiles inside one row while LANES <= WIDTH makes cyclic
// partitioning match the accesses issued by the unrolled lane loop.  Wider
// tiles retain the contiguous flat traversal because the width dimension is
// then completely partitioned.
template <int ROWS, int WIDTH, int LANES>
struct TimeDrivenRowTiles {
    static_assert(ROWS > 0, "time-driven tensors require at least one row");
    static_assert(WIDTH > 0, "time-driven tensors require a positive width");
    static_assert(LANES > 0, "time-driven tensors require at least one lane");

    static const bool ROW_ALIGNED = LANES <= WIDTH;
    static const int TILES_PER_ROW = (WIDTH + LANES - 1) / LANES;
    static const int COUNT = ROW_ALIGNED
        ? ROWS * TILES_PER_ROW
        : (ROWS * WIDTH + LANES - 1) / LANES;

    static int flat_index(int tile, int lane) {
        #pragma HLS INLINE
        if (ROW_ALIGNED) {
            const int row = tile / TILES_PER_ROW;
            const int column =
                (tile % TILES_PER_ROW) * LANES + lane;
            return row * WIDTH + column;
        }
        return tile * LANES + lane;
    }

    static bool valid(int tile, int lane) {
        #pragma HLS INLINE
        if (ROW_ALIGNED) {
            return (tile % TILES_PER_ROW) * LANES + lane < WIDTH;
        }
        return tile * LANES + lane < ROWS * WIDTH;
    }
};

template <int PLANES, int HEIGHT, int WIDTH, int LANES>
struct TimeDrivenTensorTiles {
    static_assert(PLANES > 0, "time-driven tensors require a plane");
    static_assert(HEIGHT > 0, "time-driven tensors require a positive height");
    static_assert(WIDTH > 0, "time-driven tensors require a positive width");
    static_assert(LANES > 0, "time-driven tensors require at least one lane");

    static const int PLANE_SIZE = HEIGHT * WIDTH;
    static const bool ROW_ALIGNED = LANES <= WIDTH;
    static const bool PLANE_ALIGNED = !ROW_ALIGNED && LANES <= PLANE_SIZE;
    static const int TILES_PER_ROW = (WIDTH + LANES - 1) / LANES;
    static const int TILES_PER_PLANE =
        (PLANE_SIZE + LANES - 1) / LANES;
    static const int COUNT = ROW_ALIGNED
        ? PLANES * HEIGHT * TILES_PER_ROW
        : (PLANE_ALIGNED
            ? PLANES * TILES_PER_PLANE
            : (PLANES * PLANE_SIZE + LANES - 1) / LANES);

    static int flat_index(int tile, int lane) {
        #pragma HLS INLINE
        if (ROW_ALIGNED) {
            const int row = tile / TILES_PER_ROW;
            const int column =
                (tile % TILES_PER_ROW) * LANES + lane;
            return row * WIDTH + column;
        }
        if (PLANE_ALIGNED) {
            const int plane = tile / TILES_PER_PLANE;
            const int offset =
                (tile % TILES_PER_PLANE) * LANES + lane;
            return plane * PLANE_SIZE + offset;
        }
        return tile * LANES + lane;
    }

    static bool valid(int tile, int lane) {
        #pragma HLS INLINE
        if (ROW_ALIGNED) {
            return (tile % TILES_PER_ROW) * LANES + lane < WIDTH;
        }
        if (PLANE_ALIGNED) {
            return (tile % TILES_PER_PLANE) * LANES + lane < PLANE_SIZE;
        }
        return tile * LANES + lane < PLANES * PLANE_SIZE;
    }
};

// template<int n_neurons, int unroll_factor, typename potential_type>
// void time_driven_LIF(potential_type potentials[n_neurons], bit_t output[n_neurons])
// {
//     #pragma HLS INLINE
//     #pragma HLS BIND_OP variable=potentials op=mul impl=dsp

//     leaky_fire_time_driven_apply_decay:
//     for (int n = 0; n < n_neurons; n++)
//     {
//         #pragma HLS UNROLL factor=unroll_factor
//         potentials[n] *= layer::decay;
//     }

//     leaky_fire_time_driven_check_threshold:
//     for (int n = 0; n < n_neurons; n++)
//     {
//         #pragma HLS UNROLL factor=unroll_factor

//         if (potentials[n] >= layer::threshold)
//         {
//             output[n] = 1;
//             potentials[n] -= layer::threshold; ///// POR ENQUANTO, SUPORTE APENAS PARA SUBTRACT EM CASO DE FIRE
//         }
//         else
//         {
//             output[n] = 0;
//         }
//     }
// }

// =========================================================
// PERCENT-PARALLEL REUSE KERNELS
// =========================================================
//
// The time-driven generator supplies one PROCESSING_ELEMENTS argument for
// each reduction primitive.  It is the exact integer U resolved from the
// normalized p contract.  The outer loop is the reuse schedule and the inner
// loop is fully unrolled, so every group describes up to U static operations.
// The guarded tail handles W % U without changing the hardware architecture.

template<
    int PROCESSING_ELEMENTS = 1,
    bool HAS_BIAS = false,
    int N_INPUTS = 1,
    int N_OUTPUTS = 1,
    typename input_type = float,
    typename result_type = input_type,
    typename params_type = input_type>
void DenseReuseImpl(
    const input_type (&input)[N_INPUTS],
    result_type (&result)[N_OUTPUTS],
    const params_type (&weights)[N_OUTPUTS][N_INPUTS],
    const params_type *bias)
{
    static_assert(PROCESSING_ELEMENTS > 0,
                  "DenseReuse requires at least one processing element");
    const int INPUT_BANKS = PROCESSING_ELEMENTS < N_INPUTS
        ? PROCESSING_ELEMENTS : N_INPUTS;
    const int OUTPUT_BANKS = PROCESSING_ELEMENTS < N_OUTPUTS
        ? PROCESSING_ELEMENTS : N_OUTPUTS;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=OUTPUT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=OUTPUT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=INPUT_BANKS dim=2

    dense_reuse_initialize:
    for (int output_index = 0; output_index < N_OUTPUTS; ++output_index) {
        #pragma HLS PIPELINE II=1
        result[output_index] = HAS_BIAS
            ? result_type(bias[output_index]) : result_type(0);
    }

    // Os itens de um mesmo grupo devem atingir saidas distintas.  Com
    // input_index variando mais rapido, os U PEs desenrolados acumulariam
    // todos em result[output_index] -- o mesmo banco -- e o HLS serializa a
    // leitura-modificacao-escrita, elevando o II para U+2 e anulando o
    // paralelismo.  Enumerar com o indice de saida variando mais rapido nao
    // basta: o banco de um PE e output_index % OUTPUT_BANKS, e com
    // output_index = operation % N_OUTPUTS esse valor so fica constante por PE
    // quando PROCESSING_ELEMENTS divide N_OUTPUTS.  Isso vale na MLP
    // (128 saidas, U potencia de dois) e falha no grafo recorrente (38 saidas,
    // U=7), onde o HLS passa a construir um mux entre todos os bancos e o
    // conflito de recurso volta.  Percorrendo a dimensao de saida arredondada
    // para cima ate um multiplo de PROCESSING_ELEMENTS, output_index %
    // OUTPUT_BANKS == pe por construcao, e as posicoes de preenchimento caem
    // no mesmo guarda que ja tratava a cauda.  Quando U divide N_OUTPUTS o
    // passo e igual a N_OUTPUTS e a enumeracao fica identica a anterior.
    // Cada PE fica com uma faixa fixa de saidas e acumula em registrador.
    // Enumerar o dominio achatado com o indice de saida variando mais rapido
    // nao basta: o banco de um PE e output_index % OUTPUT_BANKS, e o mesmo
    // acumulador volta a ser tocado a cada ceil(N_OUTPUTS/U) iteracoes.  Na
    // MLP essa distancia e 2 e o HLS a absorve; no laco recorrente (38 saidas,
    // U=7) ela e 6 contra uma profundidade de 17, uma recorrencia real que
    // eleva o II para 4.  Com o acumulador em registrador a recorrencia passa
    // a ser a soma de um ciclo, e o resultado volta a memoria uma vez por
    // faixa.
    const int OUTPUT_TILES =
        (N_OUTPUTS + PROCESSING_ELEMENTS - 1) / PROCESSING_ELEMENTS;
    dense_reuse_tiles:
    for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
        result_type accumulators[PROCESSING_ELEMENTS];
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1

        dense_reuse_load:
        for (int pe = 0; pe < PROCESSING_ELEMENTS; ++pe) {
            #pragma HLS UNROLL
            const int output_index = tile * PROCESSING_ELEMENTS + pe;
            accumulators[pe] = output_index < N_OUTPUTS
                ? result[output_index]
                : result_type(0);
        }

        dense_reuse_groups:
        for (int input_index = 0; input_index < N_INPUTS; ++input_index) {
            #pragma HLS PIPELINE II=1
            dense_reuse_processing_elements:
            for (int pe = 0; pe < PROCESSING_ELEMENTS; ++pe) {
                #pragma HLS UNROLL
                const int output_index = tile * PROCESSING_ELEMENTS + pe;
                if (output_index < N_OUTPUTS) {
                    accumulators[pe] += result_type(
                        weights[output_index][input_index] * input[input_index]
                    );
                }
            }
        }

        dense_reuse_store:
        for (int pe = 0; pe < PROCESSING_ELEMENTS; ++pe) {
            #pragma HLS UNROLL
            const int output_index = tile * PROCESSING_ELEMENTS + pe;
            if (output_index < N_OUTPUTS) {
                result[output_index] = accumulators[pe];
            }
        }
    }
}

template<
    int PROCESSING_ELEMENTS = 1,
    int N_INPUTS = 1, int N_OUTPUTS = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void LinearReuse(
    const input_type (&input)[N_INPUTS],
    result_type (&result)[N_OUTPUTS],
    const params_type (&weights)[N_OUTPUTS][N_INPUTS])
{
    DenseReuseImpl<PROCESSING_ELEMENTS, false>(
        input, result, weights, (const params_type *)0
    );
}

template<
    int PROCESSING_ELEMENTS = 1,
    int N_INPUTS = 1, int N_OUTPUTS = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void AffineReuse(
    const input_type (&input)[N_INPUTS],
    result_type (&result)[N_OUTPUTS],
    const params_type (&weights)[N_OUTPUTS][N_INPUTS],
    const params_type (&bias)[N_OUTPUTS])
{
    const int BIAS_BANKS = PROCESSING_ELEMENTS < N_OUTPUTS
        ? PROCESSING_ELEMENTS : N_OUTPUTS;
    #pragma HLS ARRAY_PARTITION variable=bias cyclic factor=BIAS_BANKS dim=1
    DenseReuseImpl<PROCESSING_ELEMENTS, true>(input, result, weights, bias);
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int PROCESSING_ELEMENTS = 1,
    bool AVERAGE = false,
    int CHANNELS = 1, int IN_H = 1, int IN_W = 1,
    typename input_type = float, typename result_type = input_type>
void Pool2dReuseImpl(
    const input_type (&input)[CHANNELS][IN_H][IN_W],
    result_type (&output)[CHANNELS]
        [(IN_H + 2 * P_H - K_H) / S_H + 1]
        [(IN_W + 2 * P_W - K_W) / S_W + 1])
{
    static_assert(PROCESSING_ELEMENTS > 0,
                  "Pool2dReuse requires at least one processing element");
    const int OUT_H = (IN_H + 2 * P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - K_W) / S_W + 1;
    const int OUTPUTS = CHANNELS * OUT_H * OUT_W;
    const int TERMS_PER_OUTPUT = K_H * K_W;
    const int TOTAL_WORK_ITEMS = OUTPUTS * TERMS_PER_OUTPUT;
    const int CHANNEL_BANKS = PROCESSING_ELEMENTS < CHANNELS
        ? PROCESSING_ELEMENTS : CHANNELS;
    const int HEIGHT_BANKS = PROCESSING_ELEMENTS < IN_H
        ? PROCESSING_ELEMENTS : IN_H;
    const int WIDTH_BANKS = PROCESSING_ELEMENTS < IN_W
        ? PROCESSING_ELEMENTS : IN_W;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=CHANNEL_BANKS dim=1

    pool_reuse_initialize:
    for (int output_index = 0; output_index < OUTPUTS; ++output_index) {
        #pragma HLS PIPELINE II=1
        const int channel = output_index / (OUT_H * OUT_W);
        const int spatial = output_index % (OUT_H * OUT_W);
        output[channel][spatial / OUT_W][spatial % OUT_W] = result_type(0);
    }

    pool_reuse_groups:
    for (int reuse = 0; reuse < TOTAL_WORK_ITEMS;
         reuse += PROCESSING_ELEMENTS) {
        #pragma HLS PIPELINE II=1
        pool_reuse_processing_elements:
        for (int pe = 0; pe < PROCESSING_ELEMENTS; ++pe) {
            #pragma HLS UNROLL
            const int operation = reuse + pe;
            if (operation < TOTAL_WORK_ITEMS) {
                const int output_index = operation / TERMS_PER_OUTPUT;
                const int term = operation % TERMS_PER_OUTPUT;
                const int channel = output_index / (OUT_H * OUT_W);
                const int spatial = output_index % (OUT_H * OUT_W);
                const int out_h = spatial / OUT_W;
                const int out_w = spatial % OUT_W;
                const int kernel_h = term / K_W;
                const int kernel_w = term % K_W;
                const int in_h = out_h * S_H + kernel_h - P_H;
                const int in_w = out_w * S_W + kernel_w - P_W;
                if (in_h >= 0 && in_h < IN_H && in_w >= 0 && in_w < IN_W) {
                    output[channel][out_h][out_w] += result_type(
                        input[channel][in_h][in_w]
                    );
                }
            }
        }
    }

    if (AVERAGE) {
        pool_reuse_average:
        for (int output_index = 0; output_index < OUTPUTS; ++output_index) {
            #pragma HLS PIPELINE II=1
            const int channel = output_index / (OUT_H * OUT_W);
            const int spatial = output_index % (OUT_H * OUT_W);
            const int out_h = spatial / OUT_W;
            const int out_w = spatial % OUT_W;
            output[channel][out_h][out_w] = result_type(
                output[channel][out_h][out_w] / result_type(TERMS_PER_OUTPUT)
            );
        }
    }
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int PROCESSING_ELEMENTS = 1,
    int CHANNELS = 1, int IN_H = 1, int IN_W = 1,
    typename input_type = float, typename result_type = input_type>
void SumPool2dReuse(
    const input_type (&input)[CHANNELS][IN_H][IN_W],
    result_type (&output)[CHANNELS]
        [(IN_H + 2 * P_H - K_H) / S_H + 1]
        [(IN_W + 2 * P_W - K_W) / S_W + 1])
{
    Pool2dReuseImpl<K_H, K_W, S_H, S_W, P_H, P_W,
                    PROCESSING_ELEMENTS, false>(input, output);
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int PROCESSING_ELEMENTS = 1,
    int CHANNELS = 1, int IN_H = 1, int IN_W = 1,
    typename input_type = float, typename result_type = input_type>
void AvgPool2dReuse(
    const input_type (&input)[CHANNELS][IN_H][IN_W],
    result_type (&output)[CHANNELS]
        [(IN_H + 2 * P_H - K_H) / S_H + 1]
        [(IN_W + 2 * P_W - K_W) / S_W + 1])
{
    Pool2dReuseImpl<K_H, K_W, S_H, S_W, P_H, P_W,
                    PROCESSING_ELEMENTS, true>(input, output);
}

template<
    int K, int S, int P, int D, int GROUPS,
    int PROCESSING_ELEMENTS = 1,
    bool HAS_BIAS = false,
    int C_IN = 1, int IN_W = 1, int C_OUT = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void Conv1dReuseImpl(
    const input_type (&input)[C_IN][IN_W],
    result_type (&output)[C_OUT]
        [(IN_W + 2 * P - (D * (K - 1) + 1)) / S + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K],
    const params_type *bias)
{
    static_assert(PROCESSING_ELEMENTS > 0,
                  "Conv1dReuse requires at least one processing element");
    const int OUT_W = (IN_W + 2 * P - (D * (K - 1) + 1)) / S + 1;
    const int C_IN_GROUP = C_IN / GROUPS;
    const int C_OUT_GROUP = C_OUT / GROUPS;
    const int OUTPUTS = C_OUT * OUT_W;
    const int TERMS_PER_OUTPUT = C_IN_GROUP * K;
    const int TOTAL_WORK_ITEMS = OUTPUTS * TERMS_PER_OUTPUT;
    const int INPUT_CHANNEL_BANKS = PROCESSING_ELEMENTS < C_IN
        ? PROCESSING_ELEMENTS : C_IN;
    const int INPUT_WIDTH_BANKS = PROCESSING_ELEMENTS < IN_W
        ? PROCESSING_ELEMENTS : IN_W;
    const int OUTPUT_CHANNEL_BANKS = PROCESSING_ELEMENTS < C_OUT
        ? PROCESSING_ELEMENTS : C_OUT;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=OUTPUT_CHANNEL_BANKS dim=1

    conv1d_reuse_initialize:
    for (int output_index = 0; output_index < OUTPUTS; ++output_index) {
        #pragma HLS PIPELINE II=1
        const int output_channel = output_index / OUT_W;
        const int out_w = output_index % OUT_W;
        output[output_channel][out_w] = HAS_BIAS
            ? result_type(bias[output_channel]) : result_type(0);
    }

    conv1d_reuse_groups:
    for (int reuse = 0; reuse < TOTAL_WORK_ITEMS;
         reuse += PROCESSING_ELEMENTS) {
        #pragma HLS PIPELINE II=1
        conv1d_reuse_processing_elements:
        for (int pe = 0; pe < PROCESSING_ELEMENTS; ++pe) {
            #pragma HLS UNROLL
            const int operation = reuse + pe;
            if (operation < TOTAL_WORK_ITEMS) {
                const int output_index = operation / TERMS_PER_OUTPUT;
                const int term = operation % TERMS_PER_OUTPUT;
                const int output_channel = output_index / OUT_W;
                const int out_w = output_index % OUT_W;
                const int input_channel_offset = term / K;
                const int kernel_index = term % K;
                const int group = output_channel / C_OUT_GROUP;
                const int input_channel = group * C_IN_GROUP + input_channel_offset;
                const int input_w = out_w * S + kernel_index * D - P;
                if (input_w >= 0 && input_w < IN_W) {
                    output[output_channel][out_w] += result_type(
                        input[input_channel][input_w]
                        * weights[output_channel][input_channel_offset][kernel_index]
                    );
                }
            }
        }
    }
}

template<
    int K, int S, int P, int D, int GROUPS,
    int PROCESSING_ELEMENTS = 1,
    int C_IN = 1, int IN_W = 1, int C_OUT = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void Conv1dReuse(
    const input_type (&input)[C_IN][IN_W],
    result_type (&output)[C_OUT]
        [(IN_W + 2 * P - (D * (K - 1) + 1)) / S + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K],
    const params_type (&bias)[C_OUT])
{
    Conv1dReuseImpl<K, S, P, D, GROUPS, PROCESSING_ELEMENTS, true>(
        input, output, weights, bias
    );
}

template<
    int K, int S, int P, int D, int GROUPS,
    int PROCESSING_ELEMENTS = 1,
    int C_IN = 1, int IN_W = 1, int C_OUT = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void Conv1dReuse(
    const input_type (&input)[C_IN][IN_W],
    result_type (&output)[C_OUT]
        [(IN_W + 2 * P - (D * (K - 1) + 1)) / S + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K])
{
    Conv1dReuseImpl<K, S, P, D, GROUPS, PROCESSING_ELEMENTS, false>(
        input, output, weights, (const params_type *)0
    );
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int D_H, int D_W, int GROUPS, int PROCESSING_ELEMENTS = 1,
    bool HAS_BIAS = false,
    int C_IN = 1, int IN_H = 1, int IN_W = 1, int C_OUT = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void Conv2dReuseImpl(
    const input_type (&input)[C_IN][IN_H][IN_W],
    result_type (&output)[C_OUT]
        [(IN_H + 2 * P_H - (D_H * (K_H - 1) + 1)) / S_H + 1]
        [(IN_W + 2 * P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type *bias)
{
    static_assert(PROCESSING_ELEMENTS > 0,
                  "Conv2dReuse requires at least one processing element");
    const int OUT_H = (IN_H + 2 * P_H - (D_H * (K_H - 1) + 1)) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - (D_W * (K_W - 1) + 1)) / S_W + 1;
    const int C_IN_GROUP = C_IN / GROUPS;
    const int C_OUT_GROUP = C_OUT / GROUPS;
    const int OUTPUTS = C_OUT * OUT_H * OUT_W;
    const int TERMS_PER_OUTPUT = C_IN_GROUP * K_H * K_W;
    const int TOTAL_WORK_ITEMS = OUTPUTS * TERMS_PER_OUTPUT;
    const int INPUT_CHANNEL_BANKS = PROCESSING_ELEMENTS < C_IN
        ? PROCESSING_ELEMENTS : C_IN;
    const int INPUT_HEIGHT_BANKS = PROCESSING_ELEMENTS < IN_H
        ? PROCESSING_ELEMENTS : IN_H;
    const int INPUT_WIDTH_BANKS = PROCESSING_ELEMENTS < IN_W
        ? PROCESSING_ELEMENTS : IN_W;
    const int OUTPUT_CHANNEL_BANKS = PROCESSING_ELEMENTS < C_OUT
        ? PROCESSING_ELEMENTS : C_OUT;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUTPUT_CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=OUTPUT_CHANNEL_BANKS dim=1

    conv2d_reuse_initialize:
    for (int output_index = 0; output_index < OUTPUTS; ++output_index) {
        #pragma HLS PIPELINE II=1
        const int output_channel = output_index / (OUT_H * OUT_W);
        const int spatial = output_index % (OUT_H * OUT_W);
        output[output_channel][spatial / OUT_W][spatial % OUT_W] = HAS_BIAS
            ? result_type(bias[output_channel]) : result_type(0);
    }

    conv2d_reuse_groups:
    for (int reuse = 0; reuse < TOTAL_WORK_ITEMS;
         reuse += PROCESSING_ELEMENTS) {
        #pragma HLS PIPELINE II=1
        conv2d_reuse_processing_elements:
        for (int pe = 0; pe < PROCESSING_ELEMENTS; ++pe) {
            #pragma HLS UNROLL
            const int operation = reuse + pe;
            if (operation < TOTAL_WORK_ITEMS) {
                const int output_index = operation / TERMS_PER_OUTPUT;
                const int term = operation % TERMS_PER_OUTPUT;
                const int output_channel = output_index / (OUT_H * OUT_W);
                const int spatial = output_index % (OUT_H * OUT_W);
                const int out_h = spatial / OUT_W;
                const int out_w = spatial % OUT_W;
                const int input_channel_offset = term / (K_H * K_W);
                const int kernel_spatial = term % (K_H * K_W);
                const int kernel_h = kernel_spatial / K_W;
                const int kernel_w = kernel_spatial % K_W;
                const int group = output_channel / C_OUT_GROUP;
                const int input_channel = group * C_IN_GROUP + input_channel_offset;
                const int input_h = out_h * S_H + kernel_h * D_H - P_H;
                const int input_w = out_w * S_W + kernel_w * D_W - P_W;
                if (input_h >= 0 && input_h < IN_H &&
                    input_w >= 0 && input_w < IN_W) {
                    output[output_channel][out_h][out_w] += result_type(
                        input[input_channel][input_h][input_w]
                        * weights[output_channel][input_channel_offset]
                            [kernel_h][kernel_w]
                    );
                }
            }
        }
    }
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int D_H, int D_W, int GROUPS, int PROCESSING_ELEMENTS = 1,
    int C_IN = 1, int IN_H = 1, int IN_W = 1, int C_OUT = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void Conv2dReuse(
    const input_type (&input)[C_IN][IN_H][IN_W],
    result_type (&output)[C_OUT]
        [(IN_H + 2 * P_H - (D_H * (K_H - 1) + 1)) / S_H + 1]
        [(IN_W + 2 * P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type (&bias)[C_OUT])
{
    Conv2dReuseImpl<K_H, K_W, S_H, S_W, P_H, P_W, D_H, D_W, GROUPS,
                    PROCESSING_ELEMENTS, true>(input, output, weights, bias);
}

template<
    int K_H, int K_W, int S_H, int S_W, int P_H, int P_W,
    int D_H, int D_W, int GROUPS, int PROCESSING_ELEMENTS = 1,
    int C_IN = 1, int IN_H = 1, int IN_W = 1, int C_OUT = 1,
    typename input_type = float, typename result_type = input_type,
    typename params_type = input_type>
void Conv2dReuse(
    const input_type (&input)[C_IN][IN_H][IN_W],
    result_type (&output)[C_OUT]
        [(IN_H + 2 * P_H - (D_H * (K_H - 1) + 1)) / S_H + 1]
        [(IN_W + 2 * P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W])
{
    Conv2dReuseImpl<K_H, K_W, S_H, S_W, P_H, P_W, D_H, D_W, GROUPS,
                    PROCESSING_ELEMENTS, false>(
        input, output, weights, (const params_type *)0
    );
}

// Legacy lane-oriented primitives retained only for source compatibility with
// archived generated projects and direct primitive tests.  The current
// time-driven generator selects the percent-parallel reuse kernels above for
// reduction operators and derives the elementwise lane count from the same
// `p` contract.  New code must not expose independent output/reduction lanes.
template<int LANES = 1, typename input_type, int size>
void Merge(input_type (&receiver)[size], input_type (&other)[size])
{
    static_assert(LANES > 0, "Merge requires at least one lane");
    #pragma HLS ARRAY_PARTITION variable=receiver cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=other cyclic factor=LANES dim=1

    merge_1d_tiles:
    for (int base = 0; base < size; base += LANES) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=receiver inter false
        merge_1d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = base + lane;
            if (index < size) {
                receiver[index] += other[index];
            }
        }
    }
}

template<int LANES = 1, typename input_type, int channels, int width>
void Merge(input_type (&receiver)[channels][width],
           input_type (&other)[channels][width])
{
    static_assert(LANES > 0, "Merge requires at least one lane");
    const int WIDTH_BANKS = LANES < width ? LANES : width;
    const int CHANNEL_BANKS_RAW = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < channels
        ? CHANNEL_BANKS_RAW : channels;
    #pragma HLS ARRAY_PARTITION variable=receiver cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=receiver cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=other cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=other cyclic factor=WIDTH_BANKS dim=2

    typedef TimeDrivenRowTiles<channels, width, LANES> Tiles;
    merge_2d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=receiver inter false
        merge_2d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int ch = index / width;
                const int w = index % width;
                receiver[ch][w] += other[ch][w];
            }
        }
    }
}

template<int LANES = 1, typename input_type, int channels, int height, int width>
void Merge(input_type (&receiver)[channels][height][width],
           input_type (&other)[channels][height][width])
{
    static_assert(LANES > 0, "Merge requires at least one lane");
    const int WIDTH_BANKS = LANES < width ? LANES : width;
    const int AFTER_WIDTH = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = AFTER_WIDTH < height ? AFTER_WIDTH : height;
    const int CHANNEL_BANKS_RAW =
        (AFTER_WIDTH + HEIGHT_BANKS - 1) / HEIGHT_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < channels
        ? CHANNEL_BANKS_RAW : channels;
    #pragma HLS ARRAY_PARTITION variable=receiver cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=receiver cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=receiver cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=other cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=other cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=other cyclic factor=WIDTH_BANKS dim=3

    typedef TimeDrivenTensorTiles<channels, height, width, LANES> Tiles;
    merge_3d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=receiver inter false
        merge_3d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int ch = index / (height * width);
                const int spatial = index % (height * width);
                const int h = spatial / width;
                const int w = spatial % width;
                receiver[ch][h][w] += other[ch][h][w];
            }
        }
    }
}

template<int size, int LANES = 1,
         typename input_type, typename result_type>
void Flatten(input_type (&input)[size], result_type result[size])
{
    static_assert(LANES > 0, "Flatten requires at least one lane");
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=LANES dim=1

    flatten_1d_tiles:
    for (int base = 0; base < size; base += LANES) {
        #pragma HLS PIPELINE II=1
        flatten_1d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = base + lane;
            if (index < size) {
                result[index] = input[index];
            }
        }
    }
}

template<int height, int width, int LANES = 1,
         typename input_type, typename result_type>
void Flatten(input_type (&matrix)[height][width],
             result_type result[height * width])
{
    static_assert(LANES > 0, "Flatten requires at least one lane");
    const int WIDTH_BANKS = LANES < width ? LANES : width;
    const int HEIGHT_BANKS_RAW =
        (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = HEIGHT_BANKS_RAW < height
        ? HEIGHT_BANKS_RAW : height;
    #pragma HLS ARRAY_PARTITION variable=matrix cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=matrix cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=LANES dim=1

    typedef TimeDrivenRowTiles<height, width, LANES> Tiles;
    flatten_2d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        flatten_2d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                result[index] = matrix[index / width][index % width];
            }
        }
    }
}

template<int channels, int height, int width, int LANES = 1,
         typename input_type, typename result_type>
void Flatten(input_type (&matrix)[channels][height][width],
             result_type result[channels * height * width])
{
    static_assert(LANES > 0, "Flatten requires at least one lane");
    const int WIDTH_BANKS = LANES < width ? LANES : width;
    const int AFTER_WIDTH = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = AFTER_WIDTH < height ? AFTER_WIDTH : height;
    const int CHANNEL_BANKS_RAW =
        (AFTER_WIDTH + HEIGHT_BANKS - 1) / HEIGHT_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < channels
        ? CHANNEL_BANKS_RAW : channels;
    #pragma HLS ARRAY_PARTITION variable=matrix cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=matrix cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=matrix cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=LANES dim=1

    typedef TimeDrivenTensorTiles<channels, height, width, LANES> Tiles;
    flatten_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        flatten_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int ch = index / (height * width);
                const int spatial = index % (height * width);
                const int h = spatial / width;
                const int w = spatial % width;
                result[index] = matrix[ch][h][w];
            }
        }
    }
}

template<
    int OUTPUT_LANES,
    int REDUCTION_LANES,
    bool HAS_BIAS,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void DenseImpl(const input_type (&input)[n_inputs],
               result_type (&result)[n_neurons],
               const params_type (&weights)[n_neurons][n_inputs],
               const params_type *bias)
{
    static_assert(OUTPUT_LANES > 0, "Dense requires output lanes");
    static_assert(REDUCTION_LANES > 0, "Dense requires reduction lanes");
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=REDUCTION_LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=OUTPUT_LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=OUTPUT_LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=REDUCTION_LANES dim=2

    dense_output_tiles:
    for (int output_base = 0; output_base < n_neurons;
         output_base += OUTPUT_LANES) {
        #pragma HLS PIPELINE off
        result_type accumulators[OUTPUT_LANES];
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1

        dense_initialize_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int neuron = output_base + output_lane;
            accumulators[output_lane] =
                neuron < n_neurons && HAS_BIAS
                    ? result_type(bias[neuron]) : result_type(0);
        }

        dense_reduction_tiles:
        for (int input_base = 0; input_base < n_inputs;
             input_base += REDUCTION_LANES) {
            #pragma HLS PIPELINE II=1
            dense_output_lanes:
            for (int output_lane = 0; output_lane < OUTPUT_LANES;
                 ++output_lane) {
                #pragma HLS UNROLL
                const int neuron = output_base + output_lane;
                result_type block_sum = 0;
                dense_reduction_lanes:
                for (int reduction_lane = 0;
                     reduction_lane < REDUCTION_LANES; ++reduction_lane) {
                    #pragma HLS UNROLL
                    const int input_index = input_base + reduction_lane;
                    if (neuron < n_neurons && input_index < n_inputs) {
                        block_sum += result_type(
                            weights[neuron][input_index] * input[input_index]
                        );
                    }
                }
                if (neuron < n_neurons) {
                    accumulators[output_lane] += block_sum;
                }
            }
        }

        dense_store_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int neuron = output_base + output_lane;
            if (neuron < n_neurons) {
                result[neuron] = accumulators[output_lane];
            }
        }
    }
}

template<
    int OUTPUT_LANES = 1,
    int REDUCTION_LANES = 1,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void Linear(const input_type (&input)[n_inputs],
            result_type (&result)[n_neurons],
            const params_type (&weights)[n_neurons][n_inputs])
{
    DenseImpl<OUTPUT_LANES, REDUCTION_LANES, false>(
        input, result, weights, (const params_type *)0
    );
}

template<
    int OUTPUT_LANES = 1,
    int REDUCTION_LANES = 1,
    int n_inputs,
    int n_neurons,
    typename input_type,
    typename result_type,
    typename params_type>
void Affine(const input_type (&input)[n_inputs],
            result_type (&result)[n_neurons],
            const params_type (&weights)[n_neurons][n_inputs],
            const params_type (&bias)[n_neurons])
{
    #pragma HLS ARRAY_PARTITION variable=bias cyclic factor=OUTPUT_LANES dim=1
    DenseImpl<OUTPUT_LANES, REDUCTION_LANES, true>(
        input, result, weights, bias
    );
}

template<int size, int LANES = 1,
         typename input_type, typename result_type, typename params_type>
void Scale(const input_type (&input)[size],
           result_type (&result)[size],
           const params_type (&scale)[size])
{
    static_assert(LANES > 0, "Scale requires at least one lane");
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=LANES dim=1

    scale_1d_tiles:
    for (int base = 0; base < size; base += LANES) {
        #pragma HLS PIPELINE II=1
        scale_1d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = base + lane;
            if (index < size) {
                result[index] = input[index] * scale[index];
            }
        }
    }
}

template<int height, int width, int LANES = 1,
         typename input_type, typename result_type, typename params_type>
void Scale(const input_type (&input)[height][width],
           result_type (&result)[height][width],
           const params_type (&scale)[height][width])
{
    static_assert(LANES > 0, "Scale requires at least one lane");
    const int WIDTH_BANKS = LANES < width ? LANES : width;
    const int HEIGHT_BANKS_RAW = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = HEIGHT_BANKS_RAW < height
        ? HEIGHT_BANKS_RAW : height;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=WIDTH_BANKS dim=2

    typedef TimeDrivenRowTiles<height, width, LANES> Tiles;
    scale_2d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        scale_2d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int h = index / width;
                const int w = index % width;
                result[h][w] = input[h][w] * scale[h][w];
            }
        }
    }
}

template<int channels, int height, int width, int LANES = 1,
         typename input_type, typename result_type, typename params_type>
void Scale(const input_type (&input)[channels][height][width],
           result_type (&result)[channels][height][width],
           const params_type (&scale)[channels][height][width])
{
    static_assert(LANES > 0, "Scale requires at least one lane");
    const int WIDTH_BANKS = LANES < width ? LANES : width;
    const int AFTER_WIDTH = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = AFTER_WIDTH < height ? AFTER_WIDTH : height;
    const int CHANNEL_BANKS_RAW =
        (AFTER_WIDTH + HEIGHT_BANKS - 1) / HEIGHT_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < channels
        ? CHANNEL_BANKS_RAW : channels;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=scale cyclic factor=WIDTH_BANKS dim=3

    typedef TimeDrivenTensorTiles<channels, height, width, LANES> Tiles;
    scale_3d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        scale_3d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int c = index / (height * width);
                const int spatial = index % (height * width);
                const int h = spatial / width;
                const int w = spatial % width;
                result[c][h][w] = input[c][h][w] * scale[c][h][w];
            }
        }
    }
}

template <
    int K_H,  int K_W,
    int S_H,  int S_W,
    int P_H,  int P_W,
    int IN_H, int IN_W,
    typename input_type,
    typename result_type>
void SumPool1d(
    const input_type (&input)[IN_H][IN_W],
    result_type output[(IN_H + 2*P_H - K_H) / S_H + 1][(IN_W + 2*P_W - K_W) / S_W + 1])
{
    // Constantes de dimensão de saída
    const int OUT_H = (IN_H + 2 * P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - K_W) / S_W + 1;

    // Loop Vertical da Saída
    for (int i = 0; i < OUT_H; ++i) {
        
        // Loop Horizontal da Saída
        for (int j = 0; j < OUT_W; ++j) {
            
            result_type sum = 0;

            // --- Janela do Kernel (Retangular) ---
            
            // Loop Vertical do Kernel
            for (int ki = 0; ki < K_H; ++ki) {
                
                // Loop Horizontal do Kernel
                for (int kj = 0; kj < K_W; ++kj) {
                    
                    // Lógica de Endereçamento com Padding Virtual
                    int r_idx = (i * S_H) + ki - P_H;
                    int c_idx = (j * S_W) + kj - P_W;

                    // Verificação de Borda (Boundary Check)
                    if (r_idx >= 0 && r_idx < IN_H && c_idx >= 0 && c_idx < IN_W) {
                        sum += input[r_idx][c_idx];
                    }
                }
            }
            output[i][j] = sum;
        }
    }
}

template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int OUTPUT_LANES,
    int REDUCTION_LANES,
    bool AVERAGE,
    int CHANNELS,
    int IN_H, int IN_W,
    typename input_type,
    typename result_type>
void Pool2dImpl(
    const input_type (&input)[CHANNELS][IN_H][IN_W],
    result_type (&output)[CHANNELS]
        [(IN_H + 2*P_H - K_H) / S_H + 1]
        [(IN_W + 2*P_W - K_W) / S_W + 1])
{
    static_assert(OUTPUT_LANES > 0, "Pooling requires output lanes");
    static_assert(REDUCTION_LANES > 0, "Pooling requires reduction lanes");
    const int OUT_H = (IN_H + 2 * P_H - K_H) / S_H + 1;
    const int OUT_W = (IN_W + 2 * P_W - K_W) / S_W + 1;
    const int REDUCTION_COUNT = K_H * K_W;
    typedef TimeDrivenTensorTiles<CHANNELS, OUT_H, OUT_W,
                                  OUTPUT_LANES> OutputTiles;
    typedef TimeDrivenRowTiles<K_H, K_W,
                               REDUCTION_LANES> ReductionTiles;

    const int OUT_W_BANKS = OUTPUT_LANES < OUT_W ? OUTPUT_LANES : OUT_W;
    const int AFTER_OUT_W =
        (OUTPUT_LANES + OUT_W_BANKS - 1) / OUT_W_BANKS;
    const int OUT_H_BANKS = AFTER_OUT_W < OUT_H ? AFTER_OUT_W : OUT_H;
    const int OUT_C_BANKS_RAW =
        (AFTER_OUT_W + OUT_H_BANKS - 1) / OUT_H_BANKS;
    const int OUT_C_BANKS = OUT_C_BANKS_RAW < CHANNELS
        ? OUT_C_BANKS_RAW : CHANNELS;
    const int RED_W_BANKS = REDUCTION_LANES < K_W
        ? REDUCTION_LANES : K_W;
    const int AFTER_RED_W =
        (REDUCTION_LANES + RED_W_BANKS - 1) / RED_W_BANKS;
    const int RED_H_BANKS = AFTER_RED_W < K_H ? AFTER_RED_W : K_H;
    const int INPUT_W_TARGET = OUT_W_BANKS > RED_W_BANKS
        ? OUT_W_BANKS : RED_W_BANKS;
    const int INPUT_H_TARGET = OUT_H_BANKS > RED_H_BANKS
        ? OUT_H_BANKS : RED_H_BANKS;
    const int INPUT_W_BANKS = INPUT_W_TARGET < IN_W ? INPUT_W_TARGET : IN_W;
    const int INPUT_H_BANKS = INPUT_H_TARGET < IN_H ? INPUT_H_TARGET : IN_H;

    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=OUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_H_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_W_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_H_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_W_BANKS dim=3

    pool_output_tiles:
    for (int output_tile = 0; output_tile < OutputTiles::COUNT;
         ++output_tile) {
        #pragma HLS PIPELINE off
        result_type accumulators[OUTPUT_LANES];
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1

        pool_initialize_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            accumulators[output_lane] = result_type(0);
        }

        pool_reduction_tiles:
        for (int reduction_tile = 0;
             reduction_tile < ReductionTiles::COUNT; ++reduction_tile) {
            #pragma HLS PIPELINE II=1
            pool_output_lanes:
            for (int output_lane = 0; output_lane < OUTPUT_LANES;
                 ++output_lane) {
                #pragma HLS UNROLL
                const int output_index =
                    OutputTiles::flat_index(output_tile, output_lane);
                result_type block_sum = 0;
                pool_reduction_lanes:
                for (int reduction_lane = 0;
                     reduction_lane < REDUCTION_LANES; ++reduction_lane) {
                    #pragma HLS UNROLL
                    const int reduction_index = ReductionTiles::flat_index(
                        reduction_tile, reduction_lane
                    );
                    if (OutputTiles::valid(output_tile, output_lane) &&
                        ReductionTiles::valid(
                            reduction_tile, reduction_lane)) {
                        const int c = output_index / (OUT_H * OUT_W);
                        const int spatial = output_index % (OUT_H * OUT_W);
                        const int oh = spatial / OUT_W;
                        const int ow = spatial % OUT_W;
                        const int kh = reduction_index / K_W;
                        const int kw = reduction_index % K_W;
                        const int in_h = oh * S_H + kh - P_H;
                        const int in_w = ow * S_W + kw - P_W;
                        if (in_h >= 0 && in_h < IN_H &&
                            in_w >= 0 && in_w < IN_W) {
                            block_sum += result_type(input[c][in_h][in_w]);
                        }
                    }
                }
                if (OutputTiles::valid(output_tile, output_lane)) {
                    accumulators[output_lane] += block_sum;
                }
            }
        }

        pool_store_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int output_index =
                OutputTiles::flat_index(output_tile, output_lane);
            if (OutputTiles::valid(output_tile, output_lane)) {
                const int c = output_index / (OUT_H * OUT_W);
                const int spatial = output_index % (OUT_H * OUT_W);
                const int oh = spatial / OUT_W;
                const int ow = spatial % OUT_W;
                // The division widens ap_fixed arithmetic.  Keep the two
                // assignments separate and explicitly narrow the averaged
                // value; a conditional expression between the widened result
                // and result_type is ambiguous to the Clang front-end used by
                // Vitis HLS 2025.2.
                if (AVERAGE) {
                    output[c][oh][ow] = result_type(
                        accumulators[output_lane] /
                        result_type(REDUCTION_COUNT)
                    );
                } else {
                    output[c][oh][ow] = accumulators[output_lane];
                }
            }
        }
    }
}

template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int OUTPUT_LANES = 1,
    int REDUCTION_LANES = 1,
    int CHANNELS,
    int IN_H, int IN_W,
    typename input_type,
    typename result_type>
void SumPool2d(
    const input_type (&input)[CHANNELS][IN_H][IN_W],
    result_type (&output)[CHANNELS]
        [(IN_H + 2*P_H - K_H) / S_H + 1]
        [(IN_W + 2*P_W - K_W) / S_W + 1])
{
    Pool2dImpl<K_H, K_W, S_H, S_W, P_H, P_W,
               OUTPUT_LANES, REDUCTION_LANES, false>(input, output);
}

// NIR average pooling counts zero-padded positions in the fixed divisor.
template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int OUTPUT_LANES = 1,
    int REDUCTION_LANES = 1,
    int CHANNELS,
    int IN_H, int IN_W,
    typename input_type,
    typename result_type>
void AvgPool2d(
    const input_type (&input)[CHANNELS][IN_H][IN_W],
    result_type (&output)[CHANNELS]
        [(IN_H + 2*P_H - K_H) / S_H + 1]
        [(IN_W + 2*P_W - K_W) / S_W + 1])
{
    Pool2dImpl<K_H, K_W, S_H, S_W, P_H, P_W,
               OUTPUT_LANES, REDUCTION_LANES, true>(input, output);
}

template <
    int K, int S, int P, int D, int GROUPS,
    int OUTPUT_LANES, int REDUCTION_LANES, bool HAS_BIAS,
    int C_IN, int IN_W, int C_OUT,
    typename input_type, typename result_type, typename params_type>
void Conv1dImpl(
    const input_type (&input)[C_IN][IN_W],
    result_type (&output)[C_OUT][(IN_W + 2*P - (D * (K - 1) + 1)) / S + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K],
    const params_type *bias)
{
    static_assert(OUTPUT_LANES > 0, "Conv1d requires output lanes");
    static_assert(REDUCTION_LANES > 0, "Conv1d requires reduction lanes");
    const int OUT_W = (IN_W + 2*P - (D * (K - 1) + 1)) / S + 1;
    const int C_IN_GROUP = C_IN / GROUPS;
    const int C_OUT_GROUP = C_OUT / GROUPS;
    typedef TimeDrivenRowTiles<C_OUT, OUT_W,
                               OUTPUT_LANES> OutputTiles;
    typedef TimeDrivenRowTiles<C_IN_GROUP, K,
                               REDUCTION_LANES> ReductionTiles;

    const int OUT_W_BANKS = OUTPUT_LANES < OUT_W ? OUTPUT_LANES : OUT_W;
    const int OUT_C_BANKS_RAW =
        (OUTPUT_LANES + OUT_W_BANKS - 1) / OUT_W_BANKS;
    const int OUT_C_BANKS = OUT_C_BANKS_RAW < C_OUT
        ? OUT_C_BANKS_RAW : C_OUT;
    const int RED_K_BANKS = REDUCTION_LANES < K
        ? REDUCTION_LANES : K;
    const int RED_C_BANKS_RAW =
        (REDUCTION_LANES + RED_K_BANKS - 1) / RED_K_BANKS;
    const int RED_C_BANKS = RED_C_BANKS_RAW < C_IN_GROUP
        ? RED_C_BANKS_RAW : C_IN_GROUP;
    const int INPUT_C_TARGET = OUT_C_BANKS > RED_C_BANKS
        ? OUT_C_BANKS : RED_C_BANKS;
    const int INPUT_C_BANKS = INPUT_C_TARGET < C_IN
        ? INPUT_C_TARGET : C_IN;
    const int INPUT_W_TARGET = OUT_W_BANKS > RED_K_BANKS
        ? OUT_W_BANKS : RED_K_BANKS;
    const int INPUT_W_BANKS = INPUT_W_TARGET < IN_W
        ? INPUT_W_TARGET : IN_W;

    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_W_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_W_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=OUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=RED_C_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=RED_K_BANKS dim=3

    conv1d_output_tiles:
    for (int output_tile = 0; output_tile < OutputTiles::COUNT;
         ++output_tile) {
        #pragma HLS PIPELINE off
        result_type accumulators[OUTPUT_LANES];
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1

        conv1d_initialize_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int output_index =
                OutputTiles::flat_index(output_tile, output_lane);
            const int oc = output_index / OUT_W;
            accumulators[output_lane] =
                OutputTiles::valid(output_tile, output_lane) && HAS_BIAS
                    ? result_type(bias[oc]) : result_type(0);
        }

        conv1d_reduction_tiles:
        for (int reduction_tile = 0;
             reduction_tile < ReductionTiles::COUNT; ++reduction_tile) {
            #pragma HLS PIPELINE II=1
            conv1d_output_lanes:
            for (int output_lane = 0; output_lane < OUTPUT_LANES;
                 ++output_lane) {
                #pragma HLS UNROLL
                const int output_index =
                    OutputTiles::flat_index(output_tile, output_lane);
                result_type block_sum = 0;
                conv1d_reduction_lanes:
                for (int reduction_lane = 0;
                     reduction_lane < REDUCTION_LANES; ++reduction_lane) {
                    #pragma HLS UNROLL
                    const int reduction_index = ReductionTiles::flat_index(
                        reduction_tile, reduction_lane
                    );
                    if (OutputTiles::valid(output_tile, output_lane) &&
                        ReductionTiles::valid(
                            reduction_tile, reduction_lane)) {
                        const int oc = output_index / OUT_W;
                        const int ow = output_index % OUT_W;
                        const int ic_offset = reduction_index / K;
                        const int k = reduction_index % K;
                        const int group_id = oc / C_OUT_GROUP;
                        const int ic = group_id * C_IN_GROUP + ic_offset;
                        const int in_w = ow * S + k * D - P;
                        if (in_w >= 0 && in_w < IN_W) {
                            block_sum += result_type(
                                input[ic][in_w] *
                                weights[oc][ic_offset][k]
                            );
                        }
                    }
                }
                if (OutputTiles::valid(output_tile, output_lane)) {
                    accumulators[output_lane] += block_sum;
                }
            }
        }

        conv1d_store_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int output_index =
                OutputTiles::flat_index(output_tile, output_lane);
            if (OutputTiles::valid(output_tile, output_lane)) {
                const int oc = output_index / OUT_W;
                const int ow = output_index % OUT_W;
                output[oc][ow] = accumulators[output_lane];
            }
        }
    }
}

template <
    int K, int S, int P, int D, int GROUPS,
    int OUTPUT_LANES = 1, int REDUCTION_LANES = 1,
    int C_IN, int IN_W, int C_OUT,
    typename input_type, typename result_type, typename params_type>
void Conv1d(
    const input_type (&input)[C_IN][IN_W],
    result_type (&output)[C_OUT][(IN_W + 2*P - (D * (K - 1) + 1)) / S + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K],
    const params_type (&bias)[C_OUT])
{
    const int BIAS_BANKS = OUTPUT_LANES < C_OUT ? OUTPUT_LANES : C_OUT;
    #pragma HLS ARRAY_PARTITION variable=bias cyclic factor=BIAS_BANKS dim=1
    Conv1dImpl<K, S, P, D, GROUPS,
               OUTPUT_LANES, REDUCTION_LANES, true>(
        input, output, weights, bias
    );
}

template <
    int K, int S, int P, int D, int GROUPS,
    int OUTPUT_LANES = 1, int REDUCTION_LANES = 1,
    int C_IN, int IN_W, int C_OUT,
    typename input_type, typename result_type, typename params_type>
void Conv1d(
    const input_type (&input)[C_IN][IN_W],
    result_type (&output)[C_OUT][(IN_W + 2*P - (D * (K - 1) + 1)) / S + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K])
{
    Conv1dImpl<K, S, P, D, GROUPS,
               OUTPUT_LANES, REDUCTION_LANES, false>(
        input, output, weights, (const params_type *)0);
}

/**
 * Conv2D Genérica para Vitis HLS
 * ------------------------------------------------------------------
 * Suporta: Weights, Bias, Stride, Padding, Dilation, Groups.
 * * input_type: Tipo dos dados (input/output)
 * params_type: Tipo dos pesos e bias
 * * Dimensões (Template Parameters para Síntese Estática):
 * C_IN, H_IN, W_IN: Dimensões de Entrada
 * C_OUT: Canais de Saída
 * K_H, K_W: Tamanho do Kernel
 * S_H, S_W: Stride (Passo)
 * P_H, P_W: Padding (Zero padding)
 * D_H, D_W: Dilation (Espaçamento entre elementos do kernel)
 * GROUPS: Número de grupos (1 = Conv Normal, C_IN = Depthwise)
 */
template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int D_H, int D_W,
    int GROUPS,
    int OUTPUT_LANES, int REDUCTION_LANES, bool HAS_BIAS,
    int C_IN, int H_IN, int W_IN,
    int C_OUT,
    typename input_type,
    typename result_type,
    typename params_type>
void Conv2dImpl(
    const input_type (&input)[C_IN][H_IN][W_IN],
    result_type (&output)[C_OUT][(H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1][(W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type *bias)
{
    static_assert(OUTPUT_LANES > 0, "Conv2d requires output lanes");
    static_assert(REDUCTION_LANES > 0, "Conv2d requires reduction lanes");
    const int H_OUT = (H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1;
    const int W_OUT = (W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1;
    const int C_IN_GROUP = C_IN / GROUPS;
    const int C_OUT_GROUP = C_OUT / GROUPS;
    const int KERNEL_AREA = K_H * K_W;
    typedef TimeDrivenTensorTiles<C_OUT, H_OUT, W_OUT,
                                  OUTPUT_LANES> OutputTiles;
    typedef TimeDrivenTensorTiles<C_IN_GROUP, K_H, K_W,
                                  REDUCTION_LANES> ReductionTiles;

    const int OUT_W_BANKS = OUTPUT_LANES < W_OUT
        ? OUTPUT_LANES : W_OUT;
    const int AFTER_OUT_W =
        (OUTPUT_LANES + OUT_W_BANKS - 1) / OUT_W_BANKS;
    const int OUT_H_BANKS = AFTER_OUT_W < H_OUT
        ? AFTER_OUT_W : H_OUT;
    const int OUT_C_BANKS_RAW =
        (AFTER_OUT_W + OUT_H_BANKS - 1) / OUT_H_BANKS;
    const int OUT_C_BANKS = OUT_C_BANKS_RAW < C_OUT
        ? OUT_C_BANKS_RAW : C_OUT;

    const int RED_KW_BANKS = REDUCTION_LANES < K_W
        ? REDUCTION_LANES : K_W;
    const int AFTER_RED_KW =
        (REDUCTION_LANES + RED_KW_BANKS - 1) / RED_KW_BANKS;
    const int RED_KH_BANKS = AFTER_RED_KW < K_H
        ? AFTER_RED_KW : K_H;
    const int RED_IC_BANKS_RAW =
        (AFTER_RED_KW + RED_KH_BANKS - 1) / RED_KH_BANKS;
    const int RED_IC_BANKS = RED_IC_BANKS_RAW < C_IN_GROUP
        ? RED_IC_BANKS_RAW : C_IN_GROUP;

    const int INPUT_C_TARGET = OUT_C_BANKS > RED_IC_BANKS
        ? OUT_C_BANKS : RED_IC_BANKS;
    const int INPUT_C_BANKS = INPUT_C_TARGET < C_IN
        ? INPUT_C_TARGET : C_IN;
    const int INPUT_H_TARGET = OUT_H_BANKS > RED_KH_BANKS
        ? OUT_H_BANKS : RED_KH_BANKS;
    const int INPUT_H_BANKS = INPUT_H_TARGET < H_IN
        ? INPUT_H_TARGET : H_IN;
    const int INPUT_W_TARGET = OUT_W_BANKS > RED_KW_BANKS
        ? OUT_W_BANKS : RED_KW_BANKS;
    const int INPUT_W_BANKS = INPUT_W_TARGET < W_IN
        ? INPUT_W_TARGET : W_IN;

    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_H_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=INPUT_W_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_H_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=OUT_W_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=OUT_C_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=RED_IC_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=RED_KH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=RED_KW_BANKS dim=4

    conv2d_output_tiles:
    for (int output_tile = 0; output_tile < OutputTiles::COUNT;
         ++output_tile) {
        #pragma HLS PIPELINE off
        result_type accumulators[OUTPUT_LANES];
        #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=1

        conv2d_initialize_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int output_index =
                OutputTiles::flat_index(output_tile, output_lane);
            const int oc = output_index / (H_OUT * W_OUT);
            accumulators[output_lane] =
                OutputTiles::valid(output_tile, output_lane) && HAS_BIAS
                    ? result_type(bias[oc]) : result_type(0);
        }

        conv2d_reduction_tiles:
        for (int reduction_tile = 0;
             reduction_tile < ReductionTiles::COUNT; ++reduction_tile) {
            #pragma HLS PIPELINE II=1
            conv2d_output_lanes:
            for (int output_lane = 0; output_lane < OUTPUT_LANES;
                 ++output_lane) {
                #pragma HLS UNROLL
                const int output_index =
                    OutputTiles::flat_index(output_tile, output_lane);
                result_type block_sum = 0;
                conv2d_reduction_lanes:
                for (int reduction_lane = 0;
                     reduction_lane < REDUCTION_LANES; ++reduction_lane) {
                    #pragma HLS UNROLL
                    const int reduction_index = ReductionTiles::flat_index(
                        reduction_tile, reduction_lane
                    );
                    if (OutputTiles::valid(output_tile, output_lane) &&
                        ReductionTiles::valid(
                            reduction_tile, reduction_lane)) {
                        const int oc = output_index / (H_OUT * W_OUT);
                        const int output_spatial =
                            output_index % (H_OUT * W_OUT);
                        const int oh = output_spatial / W_OUT;
                        const int ow = output_spatial % W_OUT;
                        const int ic_offset = reduction_index / KERNEL_AREA;
                        const int kernel_index =
                            reduction_index % KERNEL_AREA;
                        const int kh = kernel_index / K_W;
                        const int kw = kernel_index % K_W;
                        const int group_id = oc / C_OUT_GROUP;
                        const int ic = group_id * C_IN_GROUP + ic_offset;
                        const int in_h = oh * S_H + kh * D_H - P_H;
                        const int in_w = ow * S_W + kw * D_W - P_W;
                        if (in_h >= 0 && in_h < H_IN &&
                            in_w >= 0 && in_w < W_IN) {
                            block_sum += result_type(
                                input[ic][in_h][in_w] *
                                weights[oc][ic_offset][kh][kw]
                            );
                        }
                    }
                }
                if (OutputTiles::valid(output_tile, output_lane)) {
                    accumulators[output_lane] += block_sum;
                }
            }
        }

        conv2d_store_lanes:
        for (int output_lane = 0; output_lane < OUTPUT_LANES; ++output_lane) {
            #pragma HLS UNROLL
            const int output_index =
                OutputTiles::flat_index(output_tile, output_lane);
            if (OutputTiles::valid(output_tile, output_lane)) {
                const int oc = output_index / (H_OUT * W_OUT);
                const int output_spatial = output_index % (H_OUT * W_OUT);
                const int oh = output_spatial / W_OUT;
                const int ow = output_spatial % W_OUT;
                output[oc][oh][ow] = accumulators[output_lane];
            }
        }
    }
}

template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int D_H, int D_W,
    int GROUPS,
    int OUTPUT_LANES = 1, int REDUCTION_LANES = 1,
    int C_IN, int H_IN, int W_IN,
    int C_OUT,
    typename input_type,
    typename result_type,
    typename params_type>
void Conv2d(
    const input_type (&input)[C_IN][H_IN][W_IN],
    result_type (&output)[C_OUT][(H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1][(W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W],
    const params_type (&bias)[C_OUT])
{
    const int BIAS_BANKS = OUTPUT_LANES < C_OUT ? OUTPUT_LANES : C_OUT;
    #pragma HLS ARRAY_PARTITION variable=bias cyclic factor=BIAS_BANKS dim=1
    Conv2dImpl<K_H, K_W, S_H, S_W, P_H, P_W, D_H, D_W, GROUPS,
               OUTPUT_LANES, REDUCTION_LANES, true>(
        input, output, weights, bias
    );
}

template <
    int K_H, int K_W,
    int S_H, int S_W,
    int P_H, int P_W,
    int D_H, int D_W,
    int GROUPS,
    int OUTPUT_LANES = 1, int REDUCTION_LANES = 1,
    int C_IN, int H_IN, int W_IN,
    int C_OUT,
    typename input_type,
    typename result_type,
    typename params_type>
void Conv2d(
    const input_type (&input)[C_IN][H_IN][W_IN],
    result_type (&output)[C_OUT][(H_IN + 2*P_H - (D_H * (K_H - 1) + 1)) / S_H + 1][(W_IN + 2*P_W - (D_W * (K_W - 1) + 1)) / S_W + 1],
    const params_type (&weights)[C_OUT][C_IN / GROUPS][K_H][K_W])
{
    Conv2dImpl<K_H, K_W, S_H, S_W, P_H, P_W, D_H, D_W, GROUPS,
               OUTPUT_LANES, REDUCTION_LANES, false>(
        input, output, weights, (const params_type *)0);
}

template <int IN_CHANNELS, int IN_H, int IN_W, int LANES = 1,
          typename input_type, typename state_type, typename params_type>
void IF(
    const input_type (&input)[IN_CHANNELS][IN_H][IN_W],
    bit_t output[IN_CHANNELS][IN_H][IN_W],
    state_type membrane_potential[IN_CHANNELS][IN_H][IN_W],

    const params_type R[IN_CHANNELS][IN_H][IN_W],
    const params_type threshold[IN_CHANNELS][IN_H][IN_W],
    const params_type v_reset[IN_CHANNELS][IN_H][IN_W],
    bool reset_potentials)
{
    static_assert(LANES > 0, "IF requires at least one lane");
    const int WIDTH_BANKS = LANES < IN_W ? LANES : IN_W;
    const int AFTER_WIDTH = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = AFTER_WIDTH < IN_H ? AFTER_WIDTH : IN_H;
    const int CHANNEL_BANKS_RAW =
        (AFTER_WIDTH + HEIGHT_BANKS - 1) / HEIGHT_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < IN_CHANNELS
        ? CHANNEL_BANKS_RAW : IN_CHANNELS;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=membrane_potential cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=membrane_potential cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=membrane_potential cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=threshold cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=threshold cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=threshold cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=WIDTH_BANKS dim=3

    typedef TimeDrivenTensorTiles<IN_CHANNELS, IN_H, IN_W, LANES> Tiles;
    if_3d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        if_3d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int ch = index / (IN_H * IN_W);
                const int spatial = index % (IN_H * IN_W);
                const int h = spatial / IN_W;
                const int w = spatial % IN_W;
                if (reset_potentials)
                {
                    membrane_potential[ch][h][w] = 0;
                }
                
                membrane_potential[ch][h][w] += input[ch][h][w] * R[ch][h][w];
    
                if (membrane_potential[ch][h][w] >= threshold[ch][h][w])
                {
                    output[ch][h][w] = 1;
                    membrane_potential[ch][h][w] = v_reset[ch][h][w];
                }
                else
                {
                    output[ch][h][w] = 0;
                }
            }
        }
    }
}

template <int IN_H, int IN_W, int LANES = 1,
          typename input_type, typename state_type, typename params_type>
void IF(
    const input_type (&input)[IN_H][IN_W],
    bit_t output[IN_H][IN_W],
    state_type membrane_potential[IN_H][IN_W],

    const params_type R[IN_H][IN_W],
    const params_type threshold[IN_H][IN_W],
    const params_type v_reset[IN_H][IN_W],
    bool reset_potentials)
{
    static_assert(LANES > 0, "IF requires at least one lane");
    const int WIDTH_BANKS = LANES < IN_W ? LANES : IN_W;
    const int HEIGHT_BANKS_RAW =
        (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = HEIGHT_BANKS_RAW < IN_H
        ? HEIGHT_BANKS_RAW : IN_H;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=membrane_potential cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=membrane_potential cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=threshold cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=threshold cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=WIDTH_BANKS dim=2

    typedef TimeDrivenRowTiles<IN_H, IN_W, LANES> Tiles;
    if_2d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        if_2d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int h = index / IN_W;
                const int w = index % IN_W;
                if (reset_potentials) {
                    membrane_potential[h][w] = 0;
                }

                membrane_potential[h][w] += input[h][w] * R[h][w];

                if (membrane_potential[h][w] >= threshold[h][w]) {
                    output[h][w] = 1;
                    membrane_potential[h][w] = v_reset[h][w];
                } else {
                    output[h][w] = 0;
                }
            }
        }
    }
}

template <int NEURONS, int LANES = 1,
          typename input_type, typename state_type, typename params_type>
void IF(
    const input_type (&input)[NEURONS],
    bit_t output[NEURONS],
    state_type membrane_potential[NEURONS],

    const params_type R[NEURONS],
    const params_type threshold[NEURONS],
    const params_type v_reset[NEURONS],
    bool reset_potentials)
{
    static_assert(LANES > 0, "IF requires at least one lane");
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=membrane_potential cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=threshold cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=LANES dim=1

    if_1d_tiles:
    for (int base = 0; base < NEURONS; base += LANES) {
        #pragma HLS PIPELINE II=1
        if_1d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int n = base + lane;
            if (n < NEURONS) {
                if (reset_potentials) {
                    membrane_potential[n] = 0;
                }

                membrane_potential[n] += input[n] * R[n];

                if (membrane_potential[n] >= threshold[n]) {
                    output[n] = 1;
                    membrane_potential[n] = v_reset[n];
                } else {
                    output[n] = 0;
                }
            }
        }
    }
}

// =========================================================
// KERNELS MATEMÁTICOS (Lógica de 1 Neurônio)
// =========================================================

/**
 * Integrator Kernel - Calcula a dinâmica de um único neurônio
 * v_new = v_old + (Input * R)
 */
template <typename input_type, typename T_PARAM>
void integrator_kernel(input_type input, T_PARAM R, input_type& v_state) {
    #pragma HLS INLINE
    v_state = v_state + (input * R);
}

/**
 * Integrate-and-Fire Kernel - Integrador com threshold
 */
template <typename input_type, typename T_PARAM>
void if_kernel(input_type input, T_PARAM R, T_PARAM threshold, input_type v_reset, 
               input_type& v_state, bool& spike) {
    #pragma HLS INLINE
    
    // 1. Integração
    v_state = v_state + (input * R);
    
    // 2. Disparo
    if (v_state >= threshold) {
        spike = true;
        v_state = v_reset;
    } else {
        spike = false;
    }
}

/**
 * Leaky Integrator Kernel - Dinâmica com vazamento
 * tau * dv/dt = (v_leak - v) + R*I
 */
template <typename input_type>
void li_kernel(input_type input, input_type tau, input_type R, input_type v_leak, input_type dt,
               input_type& v_state) {
    #pragma HLS INLINE
    
    // Discretização de Euler: v_new = v_old + (dt/tau) * ((v_leak - v_old) + R*Input)
    input_type leak_current = v_leak - v_state;
    input_type input_current = R * input;
    input_type dv = (dt / tau) * (leak_current + input_current);
    
    v_state = v_state + dv;
}

/**
 * LIF Kernel - Leaky Integrator com threshold
 */
template <typename input_type>
void lif_kernel(input_type input, input_type tau, input_type R, input_type v_leak, input_type dt,
                input_type v_threshold, input_type v_reset, input_type& v_state, bool& spike) {
    #pragma HLS INLINE
    
    // 1. Integração com vazamento
    input_type leak_current = v_leak - v_state;
    input_type input_current = R * input;
    input_type dv = (dt / tau) * (leak_current + input_current);
    
    v_state = v_state + dv;
    
    // 2. Disparo e Reset
    if (v_state >= v_threshold) {
        spike = true;
        v_state = v_reset;
    } else {
        spike = false;
    }
}

/**
 * CubaLI Kernel - Current-Based Leaky Integrator
 * Modelo de dois estágios: sinapse (u) + membrana (v)
 */
template <typename input_type>
void cuba_li_kernel(input_type input, input_type tau_syn, input_type w_in, input_type tau_mem, 
                    input_type R, input_type v_leak, input_type dt, input_type& u_state, input_type& v_state) {
    #pragma HLS INLINE
    
    // ESTÁGIO 1: Dinâmica da Sinapse (u)
    // tau_syn * du/dt = -u + w_in * input
    input_type leak_u = 0 - u_state;  // Sinapse decai para zero
    input_type input_u = w_in * input;
    input_type du = (dt / tau_syn) * (leak_u + input_u);
    u_state = u_state + du;
    
    // ESTÁGIO 2: Dinâmica da Membrana (v)
    // tau_mem * dv/dt = (v_leak - v) + R * u
    input_type leak_v = v_leak - v_state;
    input_type input_v = R * u_state;
    input_type dv = (dt / tau_mem) * (leak_v + input_v);
    v_state = v_state + dv;
}

/**
 * CubaLIF Kernel - Current-Based LIF
 * CubaLI + threshold e reset na membrana
 */
template <typename input_type, typename W_DATA>
void cuba_lif_kernel(input_type input, W_DATA tau_syn, W_DATA w_in, W_DATA tau_mem,
                     W_DATA R, W_DATA v_leak, W_DATA dt, W_DATA v_threshold, W_DATA v_reset,
                     W_DATA& u_state, W_DATA& v_state, bit_t& spike,
                     bool reset_by_subtraction = false) {
    #pragma HLS INLINE
    
    // ESTÁGIO 1: Dinâmica da Sinapse (u)
    input_type leak_u = 0 - u_state;
    input_type input_u = w_in * input;
    input_type du = (dt / tau_syn) * (leak_u + input_u);
    u_state = u_state + du;
    
    // ESTÁGIO 2: Dinâmica da Membrana (v)
    input_type leak_v = v_leak - v_state;
    input_type input_v = R * u_state;
    input_type dv = (dt / tau_mem) * (leak_v + input_v);
    v_state = v_state + dv;
    
    // ESTÁGIO 3: Disparo e Reset (apenas na membrana)
    if (v_state >= v_threshold) {
        spike = 1;
        if (reset_by_subtraction) {
            v_state -= v_threshold;
        } else {
            v_state = v_reset;
        }
        // Nota: u_state NÃO é resetado em modelos CuBa
    } else {
        spike = 0;
    }
}

/**
 * CubaLIF kernel with precomputed Euler coefficients.
 *
 * alpha_syn = dt / tau_syn
 * beta_mem  = dt / tau_mem
 *
 * The next states remain in DYNAMICS_DATA until after the threshold
 * comparison, preventing an early fixed-point wrap from suppressing spikes.
 */
template <
    typename DYNAMICS_DATA,
    typename input_type,
    typename ALPHA_DATA,
    typename BETA_DATA,
    typename W_DATA
>
void cuba_lif_decay_kernel(
    input_type input,
    ALPHA_DATA alpha_syn,
    BETA_DATA beta_mem,
    W_DATA w_in,
    W_DATA R,
    W_DATA v_leak,
    W_DATA v_threshold,
    W_DATA v_reset,
    W_DATA& u_state,
    W_DATA& v_state,
    bit_t& spike,
    bool reset_by_subtraction = false
) {
    #pragma HLS INLINE

    DYNAMICS_DATA leak_u =
        DYNAMICS_DATA(0) - DYNAMICS_DATA(u_state);
    DYNAMICS_DATA input_u =
        DYNAMICS_DATA(w_in) * DYNAMICS_DATA(input);
    DYNAMICS_DATA u_next =
        DYNAMICS_DATA(u_state)
        + DYNAMICS_DATA(alpha_syn) * (leak_u + input_u);

    DYNAMICS_DATA leak_v =
        DYNAMICS_DATA(v_leak) - DYNAMICS_DATA(v_state);
    DYNAMICS_DATA input_v =
        DYNAMICS_DATA(R) * u_next;
    DYNAMICS_DATA v_next =
        DYNAMICS_DATA(v_state)
        + DYNAMICS_DATA(beta_mem) * (leak_v + input_v);

    if (v_next >= DYNAMICS_DATA(v_threshold)) {
        spike = 1;
        if (reset_by_subtraction) {
            v_next -= DYNAMICS_DATA(v_threshold);
        } else {
            v_next = DYNAMICS_DATA(v_reset);
        }
    } else {
        spike = 0;
    }

    u_state = W_DATA(u_next);
    v_state = W_DATA(v_next);
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - INTEGRATE-AND-FIRE
// =========================================================

/**
 * Integrate-and-Fire 1D (Dense/FC Layers)
 */
template <typename input_type, typename T_PARAM, int N>
void integrate_and_fire(
    const input_type (&input)[N],
    const T_PARAM R[N],
    const T_PARAM threshold[N],
    const input_type v_reset,
    input_type membrane_potential[N],
    bool output_spikes[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        if_kernel(input[i], R[i], threshold[i], v_reset, 
                  membrane_potential[i], output_spikes[i]);
    }
}

/**
 * Integrate-and-Fire 2D (Conv1D ou Imagens P&B)
 */
template <
    typename input_type, 
    typename T_PARAM,
    int H, int W
>
void integrate_and_fire(
    const input_type input[H][W],          // Entrada (Corrente)
    const T_PARAM R[H][W],             // Matriz R (Resistência/Ganho)
    const T_PARAM threshold[H][W],     // Matriz T (Limiares)
    const input_type v_reset,              // Valor de Reset (geralmente 0)
    input_type membrane_potential[H][W],   // ESTADO: Memória da voltagem (Read/Write)
    bool output_spikes[H][W]           // Saída: 1 (Spike) ou 0 (Silêncio)
) {
    // Diretivas de Interface para HLS (opcional, mas recomendado)
    // #pragma HLS ARRAY_PARTITION variable=R cyclic factor=4 dim=2
    
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            if_kernel(input[i][j], R[i][j], threshold[i][j], v_reset,
                      membrane_potential[i][j], output_spikes[i][j]);
        }
    }
}

/**
 * Integrate-and-Fire 3D (Conv2D com Canais)
 */
template <typename input_type, typename T_PARAM, int CH, int H, int W>
void integrate_and_fire(
    const input_type input[CH][H][W],
    const T_PARAM R[CH][H][W],
    const T_PARAM threshold[CH][H][W],
    const input_type v_reset,
    input_type membrane_potential[CH][H][W],
    bool output_spikes[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                if_kernel(input[c][i][j], R[c][i][j], threshold[c][i][j], v_reset,
                          membrane_potential[c][i][j], output_spikes[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - INTEGRATOR
// =========================================================

/**
 * Integrator 1D (Dense/FC Layers)
 */
template <typename input_type, typename T_PARAM, int N>
void integrator(
    const input_type input[N],
    const T_PARAM R[N],
    input_type voltage_state[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        integrator_kernel(input[i], R[i], voltage_state[i]);
    }
}

/**
 * Integrator 2D (Conv1D ou Imagens P&B)
 */
template <
    typename input_type,
    typename T_PARAM,
    int H, int W
>
void integrator(
    const input_type input[H][W],         // Corrente de Entrada (I)
    const T_PARAM R[H][W],            // Resistência (R) - Matriz por elemento
    input_type voltage_state[H][W]        // Estado da Voltagem (v) - Memória (In/Out)
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            integrator_kernel(input[i][j], R[i][j], voltage_state[i][j]);
        }
    }
}

/**
 * Integrator 3D (Conv2D com Canais)
 */
template <typename input_type, typename T_PARAM, int CH, int H, int W>
void integrator(
    const input_type input[CH][H][W],
    const T_PARAM R[CH][H][W],
    input_type voltage_state[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                integrator_kernel(input[c][i][j], R[c][i][j], voltage_state[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - LEAKY INTEGRATOR
// =========================================================

/**
 * Leaky Integrator 1D (Dense/FC Layers)
 */
template <typename input_type, int N>
void leaky_integrator(
    const input_type input[N],
    const input_type tau[N],
    const input_type R[N],
    const input_type v_leak[N],
    const input_type dt,
    input_type v_state[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        li_kernel(input[i], tau[i], R[i], v_leak[i], dt, v_state[i]);
    }
}

/**
 * Leaky Integrator 2D (Conv1D ou Imagens P&B)
 */
template <
    typename input_type,
    int H, int W
>
void leaky_integrator(
    const input_type input[H][W],         // Corrente de Entrada (I)
    const input_type tau[H][W],           // Constante de tempo (tau) [ms]
    const input_type R[H][W],             // Resistência (R) [Ohm]
    const input_type v_leak[H][W],        // Voltagem de Vazamento/Repouso (v_leak) [mV]
    const input_type dt,                  // Passo de tempo da simulação [ms]
    input_type v_state[H][W]              // Estado da Voltagem (v) - Memória In/Out
) {
    // Diretivas HLS para paralelismo
    // #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=4 dim=2
    
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            li_kernel(input[i][j], tau[i][j], R[i][j], v_leak[i][j], dt, v_state[i][j]);
        }
    }
}

/**
 * Leaky Integrator 3D (Conv2D com Canais)
 */
template <typename input_type, int CH, int H, int W>
void leaky_integrator(
    const input_type input[CH][H][W],
    const input_type tau[CH][H][W],
    const input_type R[CH][H][W],
    const input_type v_leak[CH][H][W],
    const input_type dt,
    input_type v_state[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                li_kernel(input[c][i][j], tau[c][i][j], R[c][i][j], 
                          v_leak[c][i][j], dt, v_state[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - LIF (Leaky Integrate-and-Fire)
// =========================================================

template <typename input_type, typename state_type,
          typename temporal_type, typename params_type>
void lif_compat_kernel(
    input_type input,
    bit_t& spike,
    state_type& v_state,
    temporal_type tau,
    params_type R,
    params_type v_leak,
    params_type v_threshold,
    params_type v_reset,
    bool reset_potentials,
    bool reset_by_subtraction = false)
{
    #pragma HLS INLINE
    if (reset_potentials) {
        v_state = state_type(v_reset);
    }
    const temporal_type leak_current = temporal_type(v_leak)
        - temporal_type(v_state);
    const temporal_type input_current = temporal_type(R)
        * temporal_type(input);
    const temporal_type min_tau = temporal_type(1.0 / 16777216.0);
    const temporal_type safe_tau = tau > temporal_type(0)
        ? tau : min_tau;
    const temporal_type dv = (temporal_type(0.0001) / safe_tau)
        * (leak_current + input_current);
    v_state = v_state + state_type(dv);
    if (v_state >= state_type(v_threshold)) {
        spike = 1;
        if (reset_by_subtraction) {
            v_state = v_state - state_type(v_threshold);
        } else {
            v_state = state_type(v_reset);
        }
    } else {
        spike = 0;
    }
}

/**
 * LIF 1D (Dense/FC Layers)
 */
template <typename input_type, int N>
void lif_neuron(
    const input_type input[N],
    const input_type tau[N],
    const input_type R[N],
    const input_type v_leak[N],
    const input_type dt,
    const input_type v_threshold[N],
    const input_type v_reset,
    input_type v_state[N],
    bool spikes_out[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        lif_kernel(input[i], tau[i], R[i], v_leak[i], dt,
                   v_threshold[i], v_reset, v_state[i], spikes_out[i]);
    }
}

// Compatibility wrapper used by the model generator for time-driven LIF.
// The generated model stores reset values per neuron and omits dt from the
// call; NMNIST models use the simulation step of 1e-4 s.
template <int N, int LANES = 1,
          typename input_type, typename state_type,
          typename temporal_type, typename params_type>
void LIF(
    const input_type (&input)[N],
    bit_t (&spikes_out)[N],
    state_type (&v_state)[N],
    const temporal_type (&tau)[N],
    const params_type (&R)[N],
    const params_type (&v_leak)[N],
    const params_type (&v_threshold)[N],
    const params_type (&v_reset)[N],
    bool reset_potentials,
    bool reset_by_subtraction = false
) {
    static_assert(LANES > 0, "LIF requires at least one lane");
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=tau cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=LANES dim=1

    lif_compat_tiles: for (int base = 0; base < N; base += LANES) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=v_state inter false
        lif_compat_lanes: for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int i = base + lane;
            if (i < N) {
                lif_compat_kernel(
                    input[i], spikes_out[i], v_state[i], tau[i], R[i],
                    v_leak[i], v_threshold[i], v_reset[i], reset_potentials,
                    reset_by_subtraction
                );
            }
        }
    }
}

template <int H, int W, int LANES = 1,
          typename input_type, typename state_type,
          typename temporal_type, typename params_type>
void LIF(
    const input_type (&input)[H][W],
    bit_t (&spikes_out)[H][W],
    state_type (&v_state)[H][W],
    const temporal_type (&tau)[H][W],
    const params_type (&R)[H][W],
    const params_type (&v_leak)[H][W],
    const params_type (&v_threshold)[H][W],
    const params_type (&v_reset)[H][W],
    bool reset_potentials,
    bool reset_by_subtraction = false)
{
    static_assert(LANES > 0, "LIF requires at least one lane");
    const int WIDTH_BANKS = LANES < W ? LANES : W;
    const int HEIGHT_BANKS_RAW =
        (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = HEIGHT_BANKS_RAW < H
        ? HEIGHT_BANKS_RAW : H;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=tau cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=tau cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=WIDTH_BANKS dim=2

    typedef TimeDrivenRowTiles<H, W, LANES> Tiles;
    lif_2d_tiles: for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=v_state inter false
        lif_2d_lanes: for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int h = index / W;
                const int w = index % W;
                lif_compat_kernel(
                    input[h][w], spikes_out[h][w], v_state[h][w],
                    tau[h][w], R[h][w], v_leak[h][w],
                    v_threshold[h][w], v_reset[h][w], reset_potentials,
                    reset_by_subtraction
                );
            }
        }
    }
}

template <int C, int H, int W, int LANES = 1,
          typename input_type, typename state_type,
          typename temporal_type, typename params_type>
void LIF(
    const input_type (&input)[C][H][W],
    bit_t (&spikes_out)[C][H][W],
    state_type (&v_state)[C][H][W],
    const temporal_type (&tau)[C][H][W],
    const params_type (&R)[C][H][W],
    const params_type (&v_leak)[C][H][W],
    const params_type (&v_threshold)[C][H][W],
    const params_type (&v_reset)[C][H][W],
    bool reset_potentials,
    bool reset_by_subtraction = false)
{
    static_assert(LANES > 0, "LIF requires at least one lane");
    const int WIDTH_BANKS = LANES < W ? LANES : W;
    const int AFTER_WIDTH = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = AFTER_WIDTH < H ? AFTER_WIDTH : H;
    const int CHANNEL_BANKS_RAW =
        (AFTER_WIDTH + HEIGHT_BANKS - 1) / HEIGHT_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < C
        ? CHANNEL_BANKS_RAW : C;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=tau cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=tau cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=tau cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=R cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=WIDTH_BANKS dim=3

    typedef TimeDrivenTensorTiles<C, H, W, LANES> Tiles;
    lif_3d_tiles: for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=v_state inter false
        lif_3d_lanes: for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int c = index / (H * W);
                const int spatial = index % (H * W);
                const int h = spatial / W;
                const int w = spatial % W;
                lif_compat_kernel(
                    input[c][h][w], spikes_out[c][h][w],
                    v_state[c][h][w], tau[c][h][w], R[c][h][w],
                    v_leak[c][h][w], v_threshold[c][h][w],
                    v_reset[c][h][w], reset_potentials,
                    reset_by_subtraction
                );
            }
        }
    }
}

/**
 * LIF 2D (Conv1D ou Imagens P&B)
 */
template <
    typename input_type,
    int H, int W
>
void lif_neuron(
    // Entradas da Dinâmica (passadas para o LI)
    const input_type input[H][W],
    const input_type tau[H][W],
    const input_type R[H][W],
    const input_type v_leak[H][W],
    const input_type dt,
    
    // Parâmetros de Disparo (Novos)
    const input_type v_threshold[H][W],  // Limiar de disparo
    const input_type v_reset,            // Valor para onde a voltagem vai após spike
    
    // Estados (IO)
    input_type v_state[H][W],            // Memória da Voltagem
    bool spikes_out[H][W]            // Saída de Spikes (1 ou 0)
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            lif_kernel(input[i][j], tau[i][j], R[i][j], v_leak[i][j], dt,
                       v_threshold[i][j], v_reset, v_state[i][j], spikes_out[i][j]);
        }
    }
}

/**
 * LIF 3D (Conv2D com Canais)
 */
template <typename input_type, int CH, int H, int W>
void lif_neuron(
    const input_type input[CH][H][W],
    const input_type tau[CH][H][W],
    const input_type R[CH][H][W],
    const input_type v_leak[CH][H][W],
    const input_type dt,
    const input_type v_threshold[CH][H][W],
    const input_type v_reset,
    input_type v_state[CH][H][W],
    bool spikes_out[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                lif_kernel(input[c][i][j], tau[c][i][j], R[c][i][j], 
                           v_leak[c][i][j], dt, v_threshold[c][i][j], v_reset,
                           v_state[c][i][j], spikes_out[c][i][j]);
            }
        }
    }
}

/**
 * Linear Scale (Helper Function)
 * ----------------------------------------------------------------
 * Aplica escala elemento-por-elemento: output[i] = input[i] * weights[i]
 * Usada como estágio inicial em modelos Current-Based.
 */
template <typename input_type, int DIM>
void linear_scale(const input_type input[DIM], const input_type weights[DIM], input_type output[DIM]) {
    #pragma HLS PIPELINE II=1
    scale_loop: for(int i=0; i<DIM; i++) {
        output[i] = input[i] * weights[i];
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - CUBA-LI
// =========================================================

/**
 * CubaLI 1D (Dense/FC Layers)
 */
template <typename input_type, int N>
void cuba_li_neuron(
    const input_type input[N],
    const input_type tau_syn[N],
    const input_type w_in[N],
    const input_type tau_mem[N],
    const input_type R[N],
    const input_type v_leak[N],
    const input_type dt,
    input_type u_state[N],
    input_type v_state[N]
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        cuba_li_kernel(input[i], tau_syn[i], w_in[i], tau_mem[i], 
                       R[i], v_leak[i], dt, u_state[i], v_state[i]);
    }
}

/**
 * CubaLI 2D (Conv1D ou Imagens P&B)
 */
template <
    typename input_type,
    int H, int W
>
void cuba_li_neuron(
    const input_type input[H][W],
    const input_type tau_syn[H][W],
    const input_type w_in[H][W],
    const input_type tau_mem[H][W],
    const input_type R[H][W],
    const input_type v_leak[H][W],
    const input_type dt,
    input_type u_state[H][W],
    input_type v_state[H][W]
) {
    loop_h: for(int i = 0; i < H; i++) {
        loop_w: for(int j = 0; j < W; j++) {
            #pragma HLS PIPELINE II=1
            cuba_li_kernel(input[i][j], tau_syn[i][j], w_in[i][j], tau_mem[i][j],
                           R[i][j], v_leak[i][j], dt, u_state[i][j], v_state[i][j]);
        }
    }
}

/**
 * CubaLI 3D (Conv2D com Canais)
 */
template <typename input_type, int CH, int H, int W>
void cuba_li_neuron(
    const input_type input[CH][H][W],
    const input_type tau_syn[CH][H][W],
    const input_type w_in[CH][H][W],
    const input_type tau_mem[CH][H][W],
    const input_type R[CH][H][W],
    const input_type v_leak[CH][H][W],
    const input_type dt,
    input_type u_state[CH][H][W],
    input_type v_state[CH][H][W]
) {
    loop_ch: for(int c = 0; c < CH; c++) {
        loop_h: for(int i = 0; i < H; i++) {
            loop_w: for(int j = 0; j < W; j++) {
                #pragma HLS PIPELINE II=1
                cuba_li_kernel(input[c][i][j], tau_syn[c][i][j], w_in[c][i][j], 
                               tau_mem[c][i][j], R[c][i][j], v_leak[c][i][j], dt,
                               u_state[c][i][j], v_state[c][i][j]);
            }
        }
    }
}

// =========================================================
// WRAPPERS DE DIMENSIONALIDADE - CUBA-LIF
// =========================================================

/**
 * CubaLIF 1D (Dense/FC Layers)
 */
template <typename input_type, typename W_DATA, int N>
void CubaLIF(
    const input_type (&input)[N],
    bit_t spikes_out[N],

    const W_DATA tau_syn[N],
    const W_DATA tau_mem[N],
    const W_DATA R_mem[N],
    const W_DATA v_leak[N],
    const W_DATA v_threshold[N],
    const W_DATA v_reset[N],
    const W_DATA w_in[N],

    W_DATA u_state[N],
    W_DATA v_state[N],
    const W_DATA dt,
    bool reset_potentials,
    bool reset_by_subtraction = false
) {
    loop_1d: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        if (reset_potentials) {
            u_state[i] = 0;
            v_state[i] = 0;
        }

        cuba_lif_kernel(input[i], tau_syn[i], w_in[i], tau_mem[i], R_mem[i],
                        v_leak[i], dt, v_threshold[i], v_reset[i],
                        u_state[i], v_state[i], spikes_out[i],
                        reset_by_subtraction);
    }
}

/**
 * CubaLIF 1D using precomputed time-driven decay coefficients.
 */
template <
    typename DYNAMICS_DATA,
    int LANES = 1,
    typename input_type,
    typename ALPHA_DATA,
    typename BETA_DATA,
    typename W_DATA,
    int N
>
void CubaLIF(
    const input_type (&input)[N],
    bit_t spikes_out[N],

    const ALPHA_DATA alpha_syn[N],
    const BETA_DATA beta_mem[N],
    const W_DATA R_mem[N],
    const W_DATA v_leak[N],
    const W_DATA v_threshold[N],
    const W_DATA v_reset[N],
    const W_DATA w_in[N],

    W_DATA u_state[N],
    W_DATA v_state[N],
    bool reset_potentials,
    bool reset_by_subtraction = false
) {
    static_assert(LANES > 0, "CubaLIF requires at least one lane");
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=alpha_syn cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=beta_mem cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=R_mem cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=w_in cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=u_state cyclic factor=LANES dim=1
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=LANES dim=1

    loop_decay_tiles: for(int base = 0; base < N; base += LANES) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=u_state inter false
        #pragma HLS DEPENDENCE variable=v_state inter false
        loop_decay_lanes: for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int i = base + lane;
            if (i < N) {
                // O ramo de reset escrevia em u_state[i] e v_state[i] antes de
                // o kernel ler e reescrever os mesmos enderecos: tres acessos
                // por iteracao no mesmo banco, o que fixa o II em 2.  Trazendo
                // o estado para registradores a iteracao faz uma leitura e uma
                // escrita.
                W_DATA u = reset_potentials ? W_DATA(0) : u_state[i];
                W_DATA v = reset_potentials ? W_DATA(0) : v_state[i];

                cuba_lif_decay_kernel<DYNAMICS_DATA>(
                    input[i], alpha_syn[i], beta_mem[i], w_in[i], R_mem[i],
                    v_leak[i], v_threshold[i], v_reset[i], u,
                    v, spikes_out[i], reset_by_subtraction
                );

                u_state[i] = u;
                v_state[i] = v;
            }
        }
    }
}

/**
 * CubaLIF 2D using precomputed time-driven decay coefficients.
 */
template <
    typename DYNAMICS_DATA,
    int LANES = 1,
    typename input_type,
    typename ALPHA_DATA,
    typename BETA_DATA,
    typename W_DATA,
    int H,
    int W
>
void CubaLIF(
    const input_type (&input)[H][W],
    bit_t spikes_out[H][W],

    const ALPHA_DATA alpha_syn[H][W],
    const BETA_DATA beta_mem[H][W],
    const W_DATA R_mem[H][W],
    const W_DATA v_leak[H][W],
    const W_DATA v_threshold[H][W],
    const W_DATA v_reset[H][W],
    const W_DATA w_in[H][W],

    W_DATA u_state[H][W],
    W_DATA v_state[H][W],
    bool reset_potentials,
    bool reset_by_subtraction = false
) {
    static_assert(LANES > 0, "CubaLIF requires at least one lane");
    const int WIDTH_BANKS = LANES < W ? LANES : W;
    const int HEIGHT_BANKS_RAW =
        (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = HEIGHT_BANKS_RAW < H
        ? HEIGHT_BANKS_RAW : H;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=alpha_syn cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=alpha_syn cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=beta_mem cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=beta_mem cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=R_mem cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=R_mem cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=w_in cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=w_in cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=u_state cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=u_state cyclic factor=WIDTH_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=HEIGHT_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=WIDTH_BANKS dim=2

    typedef TimeDrivenRowTiles<H, W, LANES> Tiles;
    cuba_lif_decay_2d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=u_state inter false
        #pragma HLS DEPENDENCE variable=v_state inter false
        cuba_lif_decay_2d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int h = index / W;
                const int w = index % W;
                if (reset_potentials) {
                    u_state[h][w] = 0;
                    v_state[h][w] = 0;
                }
                cuba_lif_decay_kernel<DYNAMICS_DATA>(
                    input[h][w], alpha_syn[h][w], beta_mem[h][w],
                    w_in[h][w], R_mem[h][w], v_leak[h][w],
                    v_threshold[h][w], v_reset[h][w], u_state[h][w],
                    v_state[h][w], spikes_out[h][w], reset_by_subtraction
                );
            }
        }
    }
}

/**
 * CubaLIF 3D using precomputed time-driven decay coefficients.
 */
template <
    typename DYNAMICS_DATA,
    int LANES = 1,
    typename input_type,
    typename ALPHA_DATA,
    typename BETA_DATA,
    typename W_DATA,
    int C,
    int H,
    int W
>
void CubaLIF(
    const input_type (&input)[C][H][W],
    bit_t spikes_out[C][H][W],

    const ALPHA_DATA alpha_syn[C][H][W],
    const BETA_DATA beta_mem[C][H][W],
    const W_DATA R_mem[C][H][W],
    const W_DATA v_leak[C][H][W],
    const W_DATA v_threshold[C][H][W],
    const W_DATA v_reset[C][H][W],
    const W_DATA w_in[C][H][W],

    W_DATA u_state[C][H][W],
    W_DATA v_state[C][H][W],
    bool reset_potentials,
    bool reset_by_subtraction = false
) {
    static_assert(LANES > 0, "CubaLIF requires at least one lane");
    const int WIDTH_BANKS = LANES < W ? LANES : W;
    const int AFTER_WIDTH = (LANES + WIDTH_BANKS - 1) / WIDTH_BANKS;
    const int HEIGHT_BANKS = AFTER_WIDTH < H ? AFTER_WIDTH : H;
    const int CHANNEL_BANKS_RAW =
        (AFTER_WIDTH + HEIGHT_BANKS - 1) / HEIGHT_BANKS;
    const int CHANNEL_BANKS = CHANNEL_BANKS_RAW < C
        ? CHANNEL_BANKS_RAW : C;
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=spikes_out cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=alpha_syn cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=alpha_syn cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=alpha_syn cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=beta_mem cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=beta_mem cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=beta_mem cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=R_mem cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=R_mem cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=R_mem cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_leak cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_threshold cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_reset cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=w_in cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=w_in cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=w_in cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=u_state cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=u_state cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=u_state cyclic factor=WIDTH_BANKS dim=3
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=CHANNEL_BANKS dim=1
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=HEIGHT_BANKS dim=2
    #pragma HLS ARRAY_PARTITION variable=v_state cyclic factor=WIDTH_BANKS dim=3

    typedef TimeDrivenTensorTiles<C, H, W, LANES> Tiles;
    cuba_lif_decay_3d_tiles:
    for (int tile = 0; tile < Tiles::COUNT; ++tile) {
        #pragma HLS PIPELINE II=1
        // Cada iteracao acessa um indice distinto, entao a
        // leitura-modificacao-escrita do estado nao carrega dependencia
        // entre iteracoes; sem isto o HLS a assume e fixa o II em 2.
        #pragma HLS DEPENDENCE variable=u_state inter false
        #pragma HLS DEPENDENCE variable=v_state inter false
        cuba_lif_decay_3d_lanes:
        for (int lane = 0; lane < LANES; ++lane) {
            #pragma HLS UNROLL
            const int index = Tiles::flat_index(tile, lane);
            if (Tiles::valid(tile, lane)) {
                const int c = index / (H * W);
                const int spatial = index % (H * W);
                const int h = spatial / W;
                const int w = spatial % W;
                if (reset_potentials) {
                    u_state[c][h][w] = 0;
                    v_state[c][h][w] = 0;
                }
                cuba_lif_decay_kernel<DYNAMICS_DATA>(
                    input[c][h][w], alpha_syn[c][h][w],
                    beta_mem[c][h][w], w_in[c][h][w], R_mem[c][h][w],
                    v_leak[c][h][w], v_threshold[c][h][w],
                    v_reset[c][h][w], u_state[c][h][w],
                    v_state[c][h][w], spikes_out[c][h][w],
                    reset_by_subtraction
                );
            }
        }
    }
}

// /**
//  * CubaLIF 2D (Conv1D ou Imagens P&B)
//  */
// template <
//     typename T_DATA,
//     int H, int W
// >
// void CubaLIF(
//     const T_DATA input[H][W],
//     const T_DATA w_in[H][W],
//     const T_DATA tau_syn[H][W],
//     const T_DATA tau_mem[H][W],
//     const T_DATA R_mem[H][W],
//     const T_DATA v_leak[H][W],
//     const T_DATA v_threshold[H][W],
//     const T_DATA v_reset,
//     const T_DATA dt,
//     T_DATA u_state[H][W],
//     T_DATA v_state[H][W],
//     bool spikes_out[H][W]
// ) {
//     loop_h: for(int i = 0; i < H; i++) {
//         loop_w: for(int j = 0; j < W; j++) {
//             #pragma HLS PIPELINE II=1
//             cuba_lif_kernel(input[i][j], tau_syn[i][j], w_in[i][j], tau_mem[i][j],
//                             R_mem[i][j], v_leak[i][j], dt, v_threshold[i][j], v_reset,
//                             u_state[i][j], v_state[i][j], spikes_out[i][j]);
//         }
//     }
// }

// /**
//  * CubaLIF 3D (Conv2D com Canais)
//  */
// template <typename T_DATA, int CH, int H, int W>
// void CubaLIF(
//     const T_DATA input[CH][H][W],
//     const T_DATA w_in[CH][H][W],
//     const T_DATA tau_syn[CH][H][W],
//     const T_DATA tau_mem[CH][H][W],
//     const T_DATA R_mem[CH][H][W],
//     const T_DATA v_leak[CH][H][W],
//     const T_DATA v_threshold[CH][H][W],
//     const T_DATA v_reset,
//     const T_DATA dt,
//     T_DATA u_state[CH][H][W],
//     T_DATA v_state[CH][H][W],
//     bool spikes_out[CH][H][W]
// ) {
//     loop_ch: for(int c = 0; c < CH; c++) {
//         loop_h: for(int i = 0; i < H; i++) {
//             loop_w: for(int j = 0; j < W; j++) {
//                 #pragma HLS PIPELINE II=1
//                 cuba_lif_kernel(input[c][i][j], tau_syn[c][i][j], w_in[c][i][j],
//                                 tau_mem[c][i][j], R_mem[c][i][j], v_leak[c][i][j], dt,
//                                 v_threshold[c][i][j], v_reset, u_state[c][i][j],
//                                 v_state[c][i][j], spikes_out[c][i][j]);
//             }
//         }
//     }
// }
