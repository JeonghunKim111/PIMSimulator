#ifndef CSC_MATRIX_LOADER_H
#define CSC_MATRIX_LOADER_H
#include <cstdint>
#include <string>
#include <vector>
namespace csc_descriptor {
struct COOEntry { uint32_t row, col; float value; };
struct CSCMatrix { uint32_t rows=0, cols=0; std::vector<uint64_t> col_ptr; std::vector<uint32_t> row_idx; std::vector<float> values; };
CSCMatrix makeCSC(uint32_t rows,uint32_t cols,std::vector<COOEntry> entries);
CSCMatrix loadCSC(const std::string& path);
}
#endif
