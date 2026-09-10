#pragma once
#include <cstdint>
struct SparseLightningIndexerGradKlLossTilingData {
    uint32_t batch, queryRows, keyRows, queryHeads, indexHeads;
    uint32_t queryDim, indexDim, ropeDim, sparseSize, empty, statsLayout;
    uint32_t queryLengthType, keyLengthType, cores;
    float scale;
    uint64_t batchScratch;
};
