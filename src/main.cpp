// src/main.cpp

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <omp.h>
#include "load_data.hpp"

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


// === Gera todas as matrizes de adjacência para n <= 5 ===
uint8_t ***generate_matrices(int n, size_t *out_count, size_t sample_size = 1000000) {
    if (n <= 5) {
        size_t total = 1ULL << (n * n);
        *out_count = total;
        uint8_t ***matrices = new uint8_t **[total];

#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < total; k++) {
            uint8_t **matrix = new uint8_t *[n];
            for (int i = 0; i < n; i++)
                matrix[i] = new uint8_t[n];

            for (int i = 0; i < n * n; i++) {
                int row = i / n;
                int col = i % n;
                matrix[row][col] = (k >> (n * n - 1 - i)) & 1;
            }
            matrices[k] = matrix;
        }
        return matrices;
    }
    *out_count = sample_size;
    return nullptr;
}


// === Gera todas as configurações binárias possíveis de n bits ===
std::vector<int> generateAllBinaryConfigs(uint32_t n) {
    uint32_t totalConfigs = 1u << n;
    std::vector<int> configs;
    configs.reserve((size_t)totalConfigs * n);

    for (uint32_t i = 0; i < totalConfigs; ++i)
        for (uint32_t bit = 0; bit < n; ++bit)
            configs.push_back((i >> (n - 1 - bit)) & 1);

    return configs;
}


// === Kernel GPU com os dois estágios ===
void gpuConsensusSimulationBatch(MTL::Device* device,
                                 const std::vector<int>& allMatricesFlat,
                                 const std::vector<int>& configs,
                                 std::vector<uint32_t>& results,
                                 std::vector<uint32_t>& stableFlags,
                                 uint32_t nodeCount,
                                 uint32_t numConfigs,
                                 uint32_t numGraphs,
                                 uint32_t numSteps) {
    using namespace MTL;
    using namespace NS;

    assert(device);

    auto commandQueue = device->newCommandQueue();
    NS::Error* error = nullptr;
    auto lib = device->newLibrary(NS::String::string("build/default.metallib", NS::UTF8StringEncoding), &error);
    if (!lib) {
        std::cerr << "Erro ao carregar Metal library: " << error->localizedDescription()->utf8String() << "\n";
        return;
    }

    auto f_count = lib->newFunction(NS::String::string("countConsensusConfigs", NS::UTF8StringEncoding));
    auto f_mark  = lib->newFunction(NS::String::string("markStableGraphs", NS::UTF8StringEncoding));
    auto p_count = device->newComputePipelineState(f_count, &error);
    auto p_mark  = device->newComputePipelineState(f_mark, &error);

    size_t allMatricesSize = sizeof(int) * allMatricesFlat.size();
    size_t configsSize     = sizeof(int) * configs.size();
    size_t resultsSize     = sizeof(uint32_t) * numGraphs;
    size_t stableSize      = sizeof(uint32_t) * numGraphs;

    auto allMatricesBuf = device->newBuffer(allMatricesSize, ResourceStorageModeShared);
    auto configsBuf     = device->newBuffer(configsSize, ResourceStorageModeShared);
    auto resultsBuf     = device->newBuffer(resultsSize, ResourceStorageModeShared);
    auto stableBuf      = device->newBuffer(stableSize, ResourceStorageModeShared);

    memcpy(allMatricesBuf->contents(), allMatricesFlat.data(), allMatricesSize);
    memcpy(configsBuf->contents(), configs.data(), configsSize);
    memset(resultsBuf->contents(), 0, resultsSize);
    memset(stableBuf->contents(),  0, stableSize);

    auto commandBuffer = commandQueue->commandBuffer();

    // === Kernel 1 ===
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(p_count);
    encoder->setBuffer(allMatricesBuf, 0, 0);
    encoder->setBuffer(configsBuf,     0, 1);
    encoder->setBuffer(resultsBuf,     0, 2);
    encoder->setBytes(&nodeCount,  sizeof(uint32_t), 3);
    encoder->setBytes(&numConfigs, sizeof(uint32_t), 4);
    encoder->setBytes(&numSteps,   sizeof(uint32_t), 5);
    encoder->setBytes(&numGraphs,  sizeof(uint32_t), 6);

    uint32_t totalThreads = numGraphs * numConfigs;
    uint32_t groupSize = std::min<uint32_t>(
            static_cast<uint32_t>(p_count->maxTotalThreadsPerThreadgroup()), 256u);
    encoder->dispatchThreads(MTL::Size(totalThreads, 1, 1), MTL::Size(groupSize, 1, 1));
    encoder->endEncoding();

    // === Kernel 2 ===
    auto encoder2 = commandBuffer->computeCommandEncoder();
    encoder2->setComputePipelineState(p_mark);
    encoder2->setBuffer(resultsBuf, 0, 0);
    encoder2->setBuffer(stableBuf,  0, 1);
    encoder2->setBytes(&numConfigs, sizeof(uint32_t), 2);
    encoder2->setBytes(&numGraphs,  sizeof(uint32_t), 3);

    uint32_t groupSize2 = std::min<uint32_t>(static_cast<uint32_t> (p_mark->maxTotalThreadsPerThreadgroup()), 256u);
    encoder2->dispatchThreads(MTL::Size(numGraphs, 1, 1), MTL::Size(groupSize2, 1, 1));
    encoder2->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    memcpy(results.data(), resultsBuf->contents(), resultsSize);
    memcpy(stableFlags.data(), stableBuf->contents(), stableSize);

    allMatricesBuf->release();
    configsBuf->release();
    resultsBuf->release();
    stableBuf->release();
    p_count->release();
    p_mark->release();
    f_count->release();
    f_mark->release();
    lib->release();
    commandQueue->release();
    commandBuffer->release();
}


// === MAIN ===
int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal não é suportado neste dispositivo.\n";
        return 1;
    }

    uint32_t nodeCount;
    std::cout << "Digite o valor de N (número de nós): ";
    std::cin >> nodeCount;

    uint32_t numSteps = (1u << nodeCount) + 1u;
    std::vector<int> configs = generateAllBinaryConfigs(nodeCount);
    uint32_t numConfigs = configs.size() / nodeCount;

    std::string filename = "./data/UniqueGraphs_n" + std::to_string(nodeCount) + ".bin";
    std::vector<std::vector<int>> allMatrices = loadAdjacencyMatrices(filename, nodeCount);
    uint32_t numGraphs = allMatrices.size();

    std::cout << "\n🔢 Total de grafos carregados: " << numGraphs << "\n";
    std::cout << "⚙️  Total de configurações por grafo: " << numConfigs << "\n\n";

    std::vector<int> allMatricesFlat;
    allMatricesFlat.reserve(numGraphs * nodeCount * nodeCount);
    for (const auto& matrix : allMatrices)
        allMatricesFlat.insert(allMatricesFlat.end(), matrix.begin(), matrix.end());

    std::vector<uint32_t> results(numGraphs);
    std::vector<uint32_t> stableFlags(numGraphs);

    std::cout << "🚀 Enviando " << numGraphs << " grafos para GPU...\n";
    auto start = std::chrono::high_resolution_clock::now();
    gpuConsensusSimulationBatch(device, allMatricesFlat, configs, results, stableFlags,
                                nodeCount, numConfigs, numGraphs, numSteps);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "✅ GPU concluído em "
              << std::chrono::duration<double>(end - start).count() << " s\n\n";

    std::vector<uint32_t> distribution(numConfigs + 1, 0);
    uint32_t totalEstaveis = 0;

    std::ofstream allGraphsCSV("./data/results/all_graphs_" + std::to_string(nodeCount) + "n.csv");
    if (!allGraphsCSV) {
        std::cerr << "Erro ao criar CSV!\n";
        return 1;
    }

    std::cout << "📊 Processando resultados...\n";
    for (size_t i = 0; i < numGraphs; ++i) {
        uint32_t convergentes = results[i];
        uint32_t isStable = stableFlags[i];
        distribution[convergentes]++;
        if (isStable) totalEstaveis++;

        const std::vector<int>& matrix = allMatrices[i];
        allGraphsCSV << "[";
        for (uint32_t row = 0; row < nodeCount; ++row) {
            allGraphsCSV << "[";
            for (uint32_t col = 0; col < nodeCount; ++col) {
                size_t index = row * nodeCount + col;
                allGraphsCSV << matrix[index];
                if (col < nodeCount - 1)
                    allGraphsCSV << ",";
            }
            allGraphsCSV << "]";
            if (row < nodeCount - 1)
                allGraphsCSV << ",";
        }
        allGraphsCSV << "]," << (isStable ? 1 : 0) << "\n";
    }

    allGraphsCSV.close();
    std::cout << "✔️ CSV salvo em ./data/results/all_graphs_" << nodeCount << "n.csv\n";
    std::cout << "📊 Total estáveis: " << totalEstaveis << " de " << numGraphs << "\n";

    std::ofstream freqCSV("./data/frequency_table_" + std::to_string(nodeCount) + "n.csv");
    freqCSV << "Quantity,Frequency\n";
    for (uint32_t k = 0; k <= numConfigs; ++k)
        if (distribution[k] > 0)
            freqCSV << k << "," << distribution[k] << "\n";
    freqCSV.close();

    std::cout << "✔️ Tabela de frequências salva.\n";

    device->release();
    return 0;
}
