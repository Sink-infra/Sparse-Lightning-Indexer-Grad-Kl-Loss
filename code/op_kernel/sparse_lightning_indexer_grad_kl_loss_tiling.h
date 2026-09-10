#pragma once
#include <cstdint>
struct SparseLightningIndexerGradKlLossTilingData {
    uint32_t batch, queryRows, keyRows, queryHeads, indexHeads;
    uint32_t queryDim, indexDim, ropeDim, sparseSize, empty, statsLayout;
    uint32_t queryLengthType, keyLengthType, cores;
    uint32_t fast, fastRecompute;
    float scale;
    uint64_t batchScratch;
};
