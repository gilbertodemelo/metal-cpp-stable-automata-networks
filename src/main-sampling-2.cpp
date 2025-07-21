// src/main.cpp

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include "load_data.hpp"  // ✅ Nosso loader

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// === Gera todas as configurações binárias possíveis de n bits ===
std::vector<int> generateAllBinaryConfigs(uint32_t n) {
    uint32_t totalConfigs = 1u << n;  // 2^n
    std::vector<int> configs;
    configs.reserve((size_t)totalConfigs * n);

    for (uint32_t i = 0; i < totalConfigs; ++i) {
        for (uint32_t bit = 0; bit < n; ++bit) {
            int value = (i >> (n - 1 - bit)) & 1;
            configs.push_back(value);
        }
    }

    return configs;
}

// === Função para rodar o kernel de consenso ===
void gpuConsensusSimulationBatch(MTL::Device* device,
                                 const std::vector<int>& allMatricesFlat, // NOVO
                                 const std::vector<int>& configs,
                                 std::vector<uint32_t>& results, // NOVO: para receber os resultados
                                 uint32_t nodeCount,
                                 uint32_t numConfigs,
                                 uint32_t numGraphs, // NOVO
                                 uint32_t numSteps) {

    using namespace MTL;
    using namespace NS;

    assert(device);

    CommandQueue* commandQueue = device->newCommandQueue();
    Library* library = device->newDefaultLibrary();
    if (!library) {
        std::cerr << "Erro: não foi possível carregar a Metal library!" << std::endl;
        return;
    }

    Function* function = library->newFunction(
        String::string("countConsensusConfigs", UTF8StringEncoding)
    );
    NS::Error* error = nullptr;
    ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
    if (!pipeline) {
        std::cerr << "Erro ao criar o pipeline: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return;
    }

    // ALTERADO: Tamanhos dos buffers para conter TODOS os dados
    size_t allMatricesSize = sizeof(int) * allMatricesFlat.size();
    size_t configsSize = sizeof(int) * configs.size();
    size_t resultsSize = sizeof(uint32_t) * numGraphs;

    // ALTERADO: Nomes dos buffers para refletir o conteúdo
    Buffer* allMatricesBuf = device->newBuffer(allMatricesSize, ResourceStorageModeShared);
    Buffer* configsBuf = device->newBuffer(configsSize, ResourceStorageModeShared);
    Buffer* resultsBuf = device->newBuffer(resultsSize, ResourceStorageModeShared); // Era countBuf

    memcpy(allMatricesBuf->contents(), allMatricesFlat.data(), allMatricesSize);
    memcpy(configsBuf->contents(), configs.data(), configsSize);
    memset(resultsBuf->contents(), 0, resultsSize); // Zera o buffer de resultados

    CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipeline);
    // ALTERADO: Mapeamento dos buffers
    encoder->setBuffer(allMatricesBuf, 0, 0); // Todas as matrizes no buffer 0
    encoder->setBuffer(configsBuf,     0, 1);
    encoder->setBuffer(resultsBuf,     0, 2); // Buffer de resultados no slot 2

    // ALTERADO: Passando os novos parâmetros para o kernel
    encoder->setBytes(&nodeCount,    sizeof(uint32_t), 3);
    encoder->setBytes(&numConfigs,   sizeof(uint32_t), 4);
    encoder->setBytes(&numSteps,     sizeof(uint32_t), 5);
    encoder->setBytes(&numGraphs,    sizeof(uint32_t), 6); // NOVO: Passa o número de grafos

    // ALTERADO: O grid agora é muito maior para cobrir todos os grafos e configs
    uint32_t totalThreads = numGraphs * numConfigs;
    uint32_t maxThreads = pipeline->maxTotalThreadsPerThreadgroup();
    uint32_t groupSize = std::min(maxThreads, 256u); // Pode aumentar um pouco o groupSize
    
    MTL::Size gridSize(totalThreads, 1, 1);
    MTL::Size threadGroupSize(groupSize, 1, 1);
    encoder->dispatchThreads(gridSize, threadGroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted(); // Espera UMA VEZ por todo o trabalho

    // NOVO: Copia o array de resultados de volta para o vetor da CPU
    memcpy(results.data(), resultsBuf->contents(), resultsSize);

    // Limpeza
    allMatricesBuf->release();
    configsBuf->release();
    resultsBuf->release();
    pipeline->release();
    function->release();
    library->release();
    commandQueue->release();
    commandBuffer->release(); // commandBuffer e encoder não precisam ser liberados
}

// === MAIN ===
// === MAIN ===
int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal não é suportado neste dispositivo." << std::endl;
        return 1;
    }

    uint32_t nodeCount;
    std::cout << "Digite o valor de N (número de nós): ";
    std::cin >> nodeCount;

    uint32_t numSteps = (1u << nodeCount) + 1u;

    std::vector<int> configs = generateAllBinaryConfigs(nodeCount);
    uint32_t numConfigs = static_cast<uint32_t>(configs.size() / nodeCount);

    std::string filename = "./data/UniqueGraphs_n" + std::to_string(nodeCount) + "_sample.bin";
    std::vector<std::vector<int>> allMatrices = loadAdjacencyMatrices(filename, nodeCount);
    uint32_t numGraphs = allMatrices.size(); // NOVO

    std::cout << "\n🔢 Total de grafos carregados: " << numGraphs << "\n";
    std::cout << "⚙️  Total de configurações possíveis por grafo: " << numConfigs << "\n\n";

    // NOVO: Achatando todas as matrizes em um único vetor
    std::vector<int> allMatricesFlat;
    allMatricesFlat.reserve(numGraphs * nodeCount * nodeCount);
    for (const auto& matrix : allMatrices) {
        allMatricesFlat.insert(allMatricesFlat.end(), matrix.begin(), matrix.end());
    }

    // NOVO: Vetor para guardar os resultados de todos os grafos
    std::vector<uint32_t> results(numGraphs);
    
    // ALTERADO: Chamada única à GPU
    std::cout << "🚀 Enviando todos os " << numGraphs << " grafos para a GPU de uma vez...\n";
    auto start = std::chrono::high_resolution_clock::now();
    gpuConsensusSimulationBatch(
        device, allMatricesFlat, configs, results,
        nodeCount, numConfigs, numGraphs, numSteps
    );
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "✅ Processamento GPU concluído.\n";
    std::cout << "⏱️  Tempo total GPU: "
              << std::chrono::duration<double>(end - start).count() << " s\n\n";


    // ALTERADO: O loop agora serve apenas para processar os resultados já calculados
    std::vector<uint32_t> distribution(numConfigs + 1, 0);
    uint32_t totalEstaveis = 0;
    
    std::ofstream allGraphsCSV("./data/results/all_graphs_" + std::to_string(nodeCount) + "n_sample.csv");
    if (!allGraphsCSV) {
        std::cerr << "Erro ao criar o arquivo de todos os grafos!\n";
        return 1;
    }

    std::cout << "📊 Processando resultados...\n";
    for (size_t i = 0; i < numGraphs; ++i) {
        uint32_t convergentes = results[i]; // Pega o resultado do vetor

        distribution[convergentes]++;
        if (convergentes == numConfigs) {
            totalEstaveis++;
        }

        const std::vector<int>& matrix = allMatrices[i];
        // ... (o código para salvar em CSV não muda) ...
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
        allGraphsCSV << "]," << ((convergentes == numConfigs) ? 1 : 0) << "\n";
    }

    allGraphsCSV.close();
    std::cout << "✔️ Arquivo com todos os grafos salvo em: ./data/results/all_graphs_"
              << nodeCount << "n.csv\n";
    
    // ... (o resto do código para imprimir a distribuição e salvar a tabela não muda) ...
    std::cout << "=============================" << std::endl;
    std::cout << "📊 Total de grafos totalmente estáveis: "
              << totalEstaveis << " de " << numGraphs << std::endl;
    std::cout << "🧠 Critério: convergência com todas as " << numConfigs << " configurações iniciais" << std::endl;

    std::cout << "\n📊 Distribuição de convergência:\nQuantity,Frequency\n";
    for (uint32_t x = 0; x <= numConfigs; ++x) {
        uint32_t freq = distribution[x];
        if (freq == 0) continue;
        std::cout << x << "," << freq << std::endl;
    }

    {
        std::string name = "./data/frequency_table_" + std::to_string(nodeCount) + "n.csv";
        std::ofstream csv(name);
        if (!csv) {
            std::cerr << "Erro ao criar arquivo CSV\n";
        } else {
            csv << "Quantity,Frequency\n";
            for (uint32_t k = 0; k <= numConfigs; ++k) {
                csv << k << "," << distribution[k] << "\n";
            }
            csv.close();
            std::cout << "✔️ Tabela de frequências salva em frequency_table.csv\n";
        }
    }

    device->release();
    return 0;
}