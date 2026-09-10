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
    t.cores = std::max(1U, std::min(t.batch, static_cast<uint32_t>(platform.GetCoreNumAiv())));
    // Each worker owns a padded scratch region, including the FP32 scatter accumulator.
    uint64_t scratch = static_cast<uint64_t>(t.keyRows) * t.indexDim +
                       static_cast<uint64_t>(t.sparseSize) * (t.indexHeads + 4) + 16;
    if (core == ge::DT_FLOAT) scratch = 2 * (scratch + t.indexHeads * t.indexDim + t.queryHeads) + 64;
    t.batchScratch = (scratch + 15) / 16 * 16;
    *context->GetTilingData<Tiling>() = t;
    uint32_t DT_QUERY = core, DT_WEIGHT = wt, DT_QROPE = in[8]->GetDataType(), DT_KROPE = in[9]->GetDataType();
    ASCENDC_TPL_SEL_PARAM(context, DT_QUERY, DT_WEIGHT, DT_QROPE, DT_KROPE);
    context->SetBlockDim(t.cores);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        (t.batchScratch * t.cores + t.cores * 16) * sizeof(float);
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
