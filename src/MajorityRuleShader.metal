#include <metal_stdlib>
using namespace metal;

// ALTERADO: Assinatura do Kernel
kernel void countConsensusConfigs(
    device const int   *allMatrices    [[ buffer(0) ]], // ALTERADO: agora são todas as matrizes
    device const int   *configs        [[ buffer(1) ]],
    device atomic_uint *results        [[ buffer(2) ]], // ALTERADO: agora é um array de resultados
    constant uint      &nodeCount      [[ buffer(3) ]],
    constant uint      &numConfigs     [[ buffer(4) ]],
    constant uint      &numSteps       [[ buffer(5) ]],
    constant uint      &numGraphs      [[ buffer(6) ]], // NOVO: número total de grafos
    uint                gid            [[ thread_position_in_grid ]]
)
{
    // NOVO: Calcula o índice do grafo e da configuração para esta thread
    uint graph_idx = gid / numConfigs;
    uint config_idx = gid % numConfigs;

    if (graph_idx >= numGraphs) return; // Proteção contra threads extras

    // buffers locais para estado atual e próximo
    thread uint cur[64];
    thread uint nxt[64];

    // 1. carrega configuração inicial e conta 1s e 0s
    // ALTERADO: Usa config_idx para encontrar a base da configuração
    uint base_config = config_idx * nodeCount;
    uint ones = 0;
    for (uint i = 0; i < nodeCount; ++i) {
        uint v = (uint)configs[base_config + i];
        cur[i] = v;
        ones += v;
    }
    uint zeros = nodeCount - ones;
    if (ones == zeros) return;
    uint initMajority = (ones > zeros) ? 1u : 0u;

    // NOVO: Calcula o início da matriz correta no buffer gigante
    uint matrix_offset = graph_idx * nodeCount * nodeCount;

    // 2. dinâmica de Regra da Maioria (vizinhança incoming)
    for (uint step = 0; step < numSteps; ++step) {
        bool same = true;
        for (uint row = 0; row < nodeCount; ++row) {
            uint sum = 0, deg = 0;
            for (uint col = 0; col < nodeCount; ++col) {
                // ALTERADO: Usa o offset da matriz para ler o dado correto
                uint e = (uint)allMatrices[matrix_offset + col * nodeCount + row];
                deg += e;
                sum += e * cur[col];
            }
            uint out;
            if (deg == 0) {
                out = cur[row];
            } else if (sum * 2 > deg) {
                out = 1u;
            } else if (sum * 2 < deg) {
                out = 0u;
            } else {
                out = cur[row];
            }
            nxt[row] = out;
            if (out != cur[row]) same = false;
        }
        for (uint i = 0; i < nodeCount; ++i) {
            cur[i] = nxt[i];
        }
        if (same) break;
    }

    // 3. teste de consenso unânime + valor corresponde à maioria inicial
    for (uint i = 1; i < nodeCount; ++i) {
        if (cur[i] != cur[0]) return;
    }
    if (cur[0] != initMajority) return;

    // ALTERADO: incrementa o contador atômico na posição correta do array de resultados
    atomic_fetch_add_explicit(&results[graph_idx], 1u, memory_order_relaxed);
}