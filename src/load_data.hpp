#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

inline std::vector<std::vector<int>> loadAdjacencyMatrices(const std::string& filename, int matrixSize) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<std::vector<int>> matrices;

    if (!file) {
        std::cerr << "Erro ao abrir o arquivo binário: " << filename << std::endl;
        return matrices;
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    int matrixLen = matrixSize * matrixSize;
    if (fileSize % matrixLen != 0) {
        std::cerr << "Erro: Tamanho do arquivo não é múltiplo do tamanho da matriz." << std::endl;
        return matrices;
    }

    int numMatrices = fileSize / matrixLen;
    std::vector<char> buffer(fileSize);

    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Erro ao ler o arquivo binário." << std::endl;
        return matrices;
    }

    for (int i = 0; i < numMatrices; ++i) {
        std::vector<int> matrix;
        for (int j = 0; j < matrixLen; ++j) {
            matrix.push_back(static_cast<unsigned char>(buffer[i * matrixLen + j]));
        }
        matrices.push_back(matrix);
    }

    return matrices;
}

#endif // LOAD_DATA_HPP
