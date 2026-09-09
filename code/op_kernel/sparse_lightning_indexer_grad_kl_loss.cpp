// Kernel侧核函数实现
#include "kernel_operator.h"

#include "sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "tiling_key_sparse_lightning_indexer_grad_kl_loss.h"

template <class DT_QUERY>
class KernelSparseLightningIndexerGradKlLoss {
public:
    __aicore__ inline KernelSparseLightningIndexerGradKlLoss() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weights, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_key, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeights, GM_ADDR loss, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_QUERY>
 __global__ __aicore__ void sparse_lightning_indexer_grad_kl_loss(GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex, GM_ADDR weights, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_key, GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeights, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SparseLightningIndexerGradKlLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseLightningIndexerGradKlLossTilingData, tiling_data, tiling);
    KernelSparseLightningIndexerGradKlLoss<DT_QUERY> op;
    op.Init(query, key, queryIndex, keyIndex, weights, sparseIndices, softmaxMax, softmaxSum, queryRope, keyRope, actual_seq_lengths_query, actual_seq_lengths_key, dQueryIndex, dKeyIndex, dWeights, loss, tiling_data.length);
    op.Process();
}
