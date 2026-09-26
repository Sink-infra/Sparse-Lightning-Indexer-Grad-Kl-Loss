#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "../op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace optiling {
using Tiling = SparseLightningIndexerGradKlLossTilingData;
// Mirror of the kernel fast-path layout so the Host can bound UB residency.
static uint64_t FastTake(uint64_t &cursor, uint64_t count) {
    cursor = (cursor + 7) / 8 * 8;
    uint64_t at = cursor;
    cursor += count + 64;
    return at;
}
static uint64_t FastLayoutFloats(const Tiling &t, bool stageT, bool stageW, bool stageQR, bool stageKR, bool tiny) {
    uint64_t rows = static_cast<uint64_t>(t.batch) * t.queryRows;
    uint64_t n1 = t.queryHeads, ni = t.indexHeads;
    uint64_t dq = t.queryDim, dr = t.ropeDim, di = t.indexDim;
    uint64_t kp = (static_cast<uint64_t>(t.sparseSize) + 7) / 8 * 8;
    uint64_t s2 = t.keyRows, batches = t.batch, k = t.sparseSize;
    uint64_t o = 0;
    FastTake(o, rows*n1*dq); FastTake(o, rows*n1*dr); FastTake(o, rows*ni*di);
    uint64_t n1p0 = (n1 + 7) / 8 * 8, nip0 = (ni + 7) / 8 * 8;
    uint64_t wStage = (ni + 15) / 16 * 16;
    FastTake(o, rows*nip0); FastTake(o, rows*n1p0); FastTake(o, rows*n1p0); FastTake(o, rows*n1p0);
    FastTake(o, rows*wStage);
    FastTake(o, batches*s2*dq); FastTake(o, batches*s2*dr); FastTake(o, batches*s2*di);
    uint64_t n1p = (n1 + 7) / 8 * 8, nip = (ni + 7) / 8 * 8;
    FastTake(o, kp*n1p);
    if (tiny) {
        // the tiny path keeps only the per-key score, similarity, indicator and
        // scaled-indicator stacks; the fast path's extra product stacks, head
        // copies and block spreads are contracted a key at a time instead
        FastTake(o, kp*nip); FastTake(o, kp*nip); FastTake(o, kp*nip);
    } else {
        FastTake(o, kp*nip); FastTake(o, kp*nip); FastTake(o, kp*nip);
        FastTake(o, kp*nip); FastTake(o, kp*nip); FastTake(o, nip*di); FastTake(o, nip*di);
        FastTake(o, kp*dq); FastTake(o, kp*dr); FastTake(o, kp*di);
    }
    FastTake(o, batches*s2*di); FastTake(o, rows*ni*di); FastTake(o, rows*nip0);
    {
        uint64_t redSpan = (uint64_t)batches*s2*di;
        // the row-split merge stages every core's partial in one burst, so the
        // accumulator slab holds one slot per participating core plus the sum
        FastTake(o, tiny ? (t.cores + 1)*(redSpan + 8) + 16 : redSpan + 3*((redSpan + 15)/16*16) + 16);
    }
    uint64_t wide = s2 > kp ? s2 : kp;
    uint64_t dim = dq > dr ? dq : dr;
    if (di > dim) dim = di;
    uint64_t prodNeed = wide*dim;
    uint64_t split = n1p*dq + 64 + nip*di;
    if (split > prodNeed) prodNeed = split;
    if (kp*nip > prodNeed) prodNeed = kp*nip;
    FastTake(o, prodNeed);
    FastTake(o, 2048);
    FastTake(o, rows*k);
    if (tiny) {
        FastTake(o, 8*kp*nip);
        FastTake(o, 8*kp);
        FastTake(o, n1p*((s2 + 7) / 8 * 8));
    }
    uint64_t stg = (o + 7) / 8 * 8;
    if (stageT) stg = (stg + rows*n1*dq + 7) / 8 * 8;
    if (stageQR) stg = (stg + rows*n1*dr + 7) / 8 * 8;
    if (stageT) stg = (stg + rows*ni*di + 7) / 8 * 8;
    if (stageW) stg = (stg + rows*ni + 7) / 8 * 8;
    if (stageT) stg = (stg + batches*s2*dq + 7) / 8 * 8;
    if (stageKR) stg = (stg + batches*s2*dr + 7) / 8 * 8;
    if (stageT) stg = (stg + batches*s2*di + 7) / 8 * 8;
    return stg + 64;
}
static bool PowerOfTwo(uint32_t n) { return n && !(n & (n - 1)); }
// One vector core per row, and the whole row budget when the shape is smaller:
// the extra blocks only redo the shared transfers but they shorten the critical
// path of the cores that carry a row.
static const uint32_t TinyCores = 8u;
// Workspace floats reserved behind the per-core slots for the tiny path's
// cross-core handshake: one ready word per core plus the launch epoch.
static const uint32_t TinySyncFloats = 64u;
// The tiny path keeps one row of work per vector core and contracts every dot
// with a single broadcast multiply, so it needs the rope block to fold into the
// head rows and a bounded number of selected keys.
static bool TinyEligible(const Tiling &t) {
    if (t.empty || t.statsLayout != 0) return false;
    if (t.queryDim % 8 || t.ropeDim % 8 || t.indexDim % 8) return false;
    if (t.queryDim < 8 || t.ropeDim < 8 || t.indexDim < 8) return false;
    if (t.queryDim > 64 || t.ropeDim > 64 || t.indexDim > 64) return false;
    if (t.ropeDim > t.queryDim) return false;
    if (t.sparseSize < 1 || t.sparseSize > 8) return false;
    if (t.queryHeads < 1 || t.queryHeads > 64) return false;
    if (t.indexHeads < 1 || t.indexHeads > 64) return false;
    if (t.keyRows < 1 || t.keyRows > 64) return false;
    if (static_cast<uint64_t>(t.batch) * t.queryRows > 8) return false;
    return true;
}
static bool FastEligible(const Tiling &t, bool stageT, bool stageW, bool stageQR, bool stageKR) {
    if (t.empty || t.statsLayout != 0) return false;
    if (t.queryDim % 8 || t.ropeDim % 8 || t.indexDim % 8) return false;
    // the tiny algorithm reduces whole rows with a single masked repeat, so each
    // row must fit one 256B block and the head trees need power-of-two widths
    if (t.queryDim < 8 || t.ropeDim < 8 || t.indexDim < 8) return false;
    if (t.queryDim > 64 || t.ropeDim > 64 || t.indexDim > 64) return false;
    if (!PowerOfTwo(t.queryHeads) || !PowerOfTwo(t.indexHeads)) return false;
    if (t.queryHeads < 1 || t.queryHeads > 64 || t.indexHeads < 1 || t.indexHeads > 64) return false;
    if (t.keyRows < 1 || t.keyRows > 64 || t.sparseSize < 1 || t.sparseSize > 64) return false;
    if (static_cast<uint64_t>(t.batch) * t.queryRows > 32) return false;
    return FastLayoutFloats(t, stageT, stageW, stageQR, stageKR, t.tiny != 0) * sizeof(float) <= 168u * 1024u;
}
static bool Floating(ge::DataType type) {
    return type == ge::DT_FLOAT || type == ge::DT_FLOAT16 || type == ge::DT_BF16;
}
static bool Shape(const gert::Tensor *tensor, std::initializer_list<int64_t> dims) {
    if (!tensor || tensor->GetStorageShape().GetDimNum() != dims.size()) return false;
    size_t i = 0;
    for (auto dim : dims) if (tensor->GetStorageShape().GetDim(i++) != dim) return false;
    return true;
}
static bool Length(const gert::Tensor *tensor, uint32_t batch, uint32_t bound, uint32_t &type) {
    type = 0;
    if (!tensor) return true;
    if (!Shape(tensor, {batch})) return false;
    auto dtype = tensor->GetDataType();
    if (dtype != ge::DT_INT32 && dtype != ge::DT_INT64) return false;
    type = dtype == ge::DT_INT32 ? 1 : 2;
    if (tensor->GetPlacement() == gert::kOnHost && tensor->GetData<uint8_t>()) {
        for (uint32_t b = 0; b < batch; ++b) {
            int64_t value = type == 1 ? tensor->GetData<int32_t>()[b] : tensor->GetData<int64_t>()[b];
            if (value < 0 || value > bound) return false;
        }
    }
    return true;
}
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *in[12];
    for (size_t i = 0; i < 10; ++i) {
        in[i] = context->GetRequiredInputTensor(i);
        if (!in[i]) return ge::GRAPH_FAILED;
    }
    in[10] = context->GetOptionalInputTensor(10);
    in[11] = context->GetOptionalInputTensor(11);
    const auto &q = in[0]->GetStorageShape();
    const auto &k = in[1]->GetStorageShape();
    const auto &qi = in[2]->GetStorageShape();
    const auto &si = in[5]->GetStorageShape();
    const auto &qr = in[8]->GetStorageShape();
    if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || qi.GetDimNum() != 4 ||
        qr.GetDimNum() != 4 || (si.GetDimNum() != 3 && si.GetDimNum() != 4)) return ge::GRAPH_FAILED;
    Tiling t{};
    auto bounded = [](int64_t n, int64_t maximum) { return n > 0 && n <= maximum; };
    if (!bounded(q.GetDim(0), 256) || q.GetDim(1) < 0 || q.GetDim(1) > 8192 ||
        !bounded(k.GetDim(1), 524288) || q.GetDim(1) > k.GetDim(1) ||
        !bounded(q.GetDim(2), 128) || !bounded(qi.GetDim(2), 64) ||
        !bounded(q.GetDim(3), 512) || !bounded(qi.GetDim(3), 128) ||
        !bounded(qr.GetDim(3), 64) || !bounded(si.GetDim(si.GetDimNum()-1), 8192)) return ge::GRAPH_FAILED;
    t.batch = q.GetDim(0); t.queryRows = qi.GetDim(1); t.keyRows = k.GetDim(1);
    t.queryHeads = q.GetDim(2); t.indexHeads = qi.GetDim(2);
    t.queryDim = q.GetDim(3); t.indexDim = qi.GetDim(3); t.ropeDim = qr.GetDim(3);
    t.sparseSize = si.GetDim(si.GetDimNum()-1); t.empty = q.GetShapeSize() == 0;
    auto core = in[0]->GetDataType();
    if (!Floating(core) || !Floating(in[8]->GetDataType()) || !Floating(in[9]->GetDataType())) return ge::GRAPH_FAILED;
    for (size_t i : {1U, 2U, 3U}) if (in[i]->GetDataType() != core) return ge::GRAPH_FAILED;
    auto wt = in[4]->GetDataType();
    if ((wt != core && wt != ge::DT_FLOAT) || in[5]->GetDataType() != ge::DT_INT32 ||
        in[6]->GetDataType() != ge::DT_FLOAT || in[7]->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    if (!t.empty) {
        if (!Shape(in[1], {t.batch,t.keyRows,1,t.queryDim}) ||
            !Shape(in[2], {t.batch,q.GetDim(1),t.indexHeads,t.indexDim}) ||
            !Shape(in[3], {t.batch,t.keyRows,1,t.indexDim}) ||
            !Shape(in[4], {t.batch,t.queryRows,t.indexHeads}) ||
            !(Shape(in[5], {t.batch,t.queryRows,1,t.sparseSize}) || Shape(in[5], {t.batch,t.queryRows,t.sparseSize})) ||
            !Shape(in[8], {t.batch,t.queryRows,t.queryHeads,t.ropeDim}) ||
            !Shape(in[9], {t.batch,t.keyRows,1,t.ropeDim})) return ge::GRAPH_FAILED;
        bool linear = Shape(in[6], {t.batch,1,t.queryRows,t.queryHeads}) ||
                      Shape(in[6], {t.batch,t.queryRows,t.queryHeads});
        bool headsFirst = Shape(in[6], {t.batch,t.queryHeads,t.queryRows}) ||
                          Shape(in[6], {t.batch,t.queryHeads,t.queryRows,1});
        if ((!linear && !headsFirst) || in[6]->GetStorageShape() != in[7]->GetStorageShape()) return ge::GRAPH_FAILED;
        t.statsLayout = linear ? 0 : 1;
    }
    if (!Length(in[10], t.batch, t.queryRows, t.queryLengthType) ||
        !Length(in[11], t.batch, t.keyRows, t.keyLengthType)) return ge::GRAPH_FAILED;
    const auto *attrs = context->GetAttrs();
    if (!attrs || !attrs->GetFloat(0) || !attrs->GetStr(1) || !attrs->GetInt(2) ||
        std::strcmp(attrs->GetStr(1), "BSND") != 0 || *attrs->GetInt(2) != 3 ||
        !std::isfinite(*attrs->GetFloat(0))) return ge::GRAPH_FAILED;
    t.scale = *attrs->GetFloat(0);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // The row-split fast path starts one vector core per row and merges the
    // gradient accumulators, so its workspace slot holds one full dKeyIndex.
    uint32_t fastRows = static_cast<uint32_t>(t.batch) * t.queryRows;
    t.cores = fastRows ? fastRows : 1u;
    t.tiny = TinyEligible(t) ? 1u : 0u;
    // The tiny path keeps one row of work per vector core and lets the row owner
    // fold the producers, so the block budget stays fixed: spreading the same work
    // over fewer blocks lengthens every core's span by more than it saves.
    if (t.tiny) t.cores = TinyCores;
    t.fast = FastEligible(t, core != ge::DT_FLOAT, wt != ge::DT_FLOAT,
                          in[8]->GetDataType() != ge::DT_FLOAT, in[9]->GetDataType() != ge::DT_FLOAT) ? 1u : 0u;
    if (!t.fast) t.tiny = 0;
    // The vector-only launch carries no hardware barrier; only the tiny path may
    // split rows across cores, because it synchronizes through the workspace.
    if (t.fast && !t.tiny) t.cores = 1u;
    // FP32 fixtures may carry statistics from a lower-precision forward pass; the
    // accurate reference rebuilds the visible normalizer, so mirror it on device.
    t.fastRecompute = (t.fast && core == ge::DT_FLOAT) ? 1u : 0u;
    if (t.fast) {
        uint64_t dkElems = static_cast<uint64_t>(t.batch) * t.keyRows * t.indexDim;
        // the tiny slot keeps one spare lane for the core's loss share
        t.batchScratch = t.cores > 1 ? (t.tiny ? dkElems + 8 : (dkElems + 15) / 16 * 16) : 0;
        // the merge stages every core's partial in one burst
        context->SetScheduleMode(1);
    } else {
        // Single vector core: the AIV-only launch has no scheduling module, so the
        // general path avoids the cross-core loss reduction entirely.
        t.cores = 1u;
        // Each worker owns a padded scratch region, including the FP32 scatter accumulator.
        uint64_t scratch = static_cast<uint64_t>(t.keyRows) * t.indexDim +
                           static_cast<uint64_t>(t.sparseSize) * (t.indexHeads + 4) + 16;
        if (core == ge::DT_FLOAT) scratch = 2 * (scratch + t.indexHeads * t.indexDim + t.queryHeads) + 64;
        t.batchScratch = (scratch + 15) / 16 * 16;
    }
    *context->GetTilingData<Tiling>() = t;
    uint32_t DT_QUERY = core, DT_WEIGHT = wt, DT_QROPE = in[8]->GetDataType(), DT_KROPE = in[9]->GetDataType();
    // The direct-UB entry saves a fixed per-block cost. Keep it for rows whose
    // sparse dot work is small enough for that cost to matter; larger tiny rows
    // retain the original TPipe entry and event allocation.
    uint64_t tinyWork = static_cast<uint64_t>(t.batch) * t.queryRows * t.sparseSize *
        (static_cast<uint64_t>(t.queryHeads) * (t.queryDim + t.ropeDim) +
         static_cast<uint64_t>(t.indexHeads) * t.indexDim);
    uint32_t STATIC_TINY = t.tiny && tinyWork <= 4096u ? 1u : 0u;
    ASCENDC_TPL_SEL_PARAM(context, DT_QUERY, DT_WEIGHT, DT_QROPE, DT_KROPE, STATIC_TINY);
    context->SetBlockDim(t.cores);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        (t.batchScratch * t.cores + t.cores * 16 + TinySyncFloats) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    for (size_t i = 0; i < 3; ++i) {
        if (!context->GetInputShape(i+2) || !context->GetOutputShape(i)) return GRAPH_FAILED;
        *context->GetOutputShape(i) = *context->GetInputShape(i+2);
    }
    *context->GetOutputShape(3) = gert::Shape{1};
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    for (size_t i = 0; i < 3; ++i) context->SetOutputDataType(i, context->GetInputDataType(i+2));
    context->SetOutputDataType(3, DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SparseLightningIndexerGradKlLoss : public OpDef {
public:
    explicit SparseLightningIndexerGradKlLoss(const char *name) : OpDef(name) {
        std::vector<ge::DataType> types[12];
        for (auto core : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
            for (auto weight : {core, ge::DT_FLOAT}) {
                if (core == ge::DT_FLOAT && types[0].size() && types[0].back() == ge::DT_FLOAT) continue;
                for (auto qr : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
                    if ((core == ge::DT_FLOAT) != (qr == ge::DT_FLOAT)) continue;
                    for (auto kr : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
                        if ((core == ge::DT_FLOAT) != (kr == ge::DT_FLOAT)) continue;
                        for (auto length : {ge::DT_INT32, ge::DT_INT64}) {
                            ge::DataType column[] = {core,core,core,core,weight,ge::DT_INT32,ge::DT_FLOAT,ge::DT_FLOAT,qr,kr,length,length};
                            for (size_t i = 0; i < 12; ++i) types[i].push_back(column[i]);
                        }
                    }
                }
            }
        }
        const char *inputs[] = {"query","key","queryIndex","keyIndex","weights","sparseIndices",
            "softmaxMax","softmaxSum","queryRope","keyRope","actual_seq_lengths_query","actual_seq_lengths_key"};
        std::vector<ge::Format> formats(types[0].size(), ge::FORMAT_ND);
        for (size_t i = 0; i < 12; ++i) this->Input(inputs[i]).ParamType(i < 10 ? REQUIRED : OPTIONAL)
            .DataType(types[i]).Format(formats);
        const char *outputs[] = {"dQueryIndex","dKeyIndex","dWeights","loss"};
        for (size_t i = 0; i < 4; ++i) this->Output(outputs[i]).ParamType(REQUIRED)
            .DataType(types[i == 3 ? 6 : i+2]).Format(formats);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0442f);
        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("deterministic").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SparseLightningIndexerGradKlLoss);
}
