#include "base/context_builder/op_tiling_context_builder.h"
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "platform/platform_info.h"
#include "../../code/op_host/sparse_lightning_indexer_grad_kl_loss.cpp"
#include <iostream>
#include <vector>

int main()
{
    fe::PlatFormInfos platform;
    fe::OptionalInfos optional;
    auto &manager = fe::PlatformInfoManager::Instance();
    if (manager.InitializePlatformInfo() != 0 ||
        manager.GetPlatformInfos("Ascend910B3", platform, optional) != 0) return 2;
    // Real Tiling on a metadata-only CANN context; no device or Kernel needed.
    const std::vector<int> scenarios = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 20};
    for (int scenario : scenarios) {
        std::vector<gert::Shape> shapes = {
            {1,1,32,512}, {1,1024,1,512}, {1,1,8,128}, {1,1024,1,128},
            {1,1,8}, {1,1,1,1024}, {1,1,1,32}, {1,1,1,32},
            {1,1,32,64}, {1,1024,1,64}, {1}, {1}};
        std::vector<ge::DataType> types = {ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,
            ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_INT32,ge::DT_FLOAT,ge::DT_FLOAT,
            ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_INT64,ge::DT_INT64};
        int64_t qLength = 1, kLength = 1024, mode = 3;
        const char *layout = "BSND";
        bool expected = scenario <= 3 || scenario == 20;
        if (scenario == 0) shapes[0].SetDim(1, 0);
        if (scenario == 2) types[4] = ge::DT_FLOAT;
        if (scenario == 3) types[8] = ge::DT_BF16;
        if (scenario == 5) types[1] = ge::DT_BF16;
        if (scenario == 6) shapes[1].SetDim(2, 2);
        if (scenario == 7) shapes[2].SetDim(1, 2);
        if (scenario == 8) shapes[5].SetDim(3, 8193);
        if (scenario == 9) qLength = -1;
        if (scenario == 10) kLength = 1025;
        if (scenario == 12) mode = 0;
        if (scenario == 13) layout = "BNSD";
        if (scenario == 14) shapes[7].SetDim(3, 31);
        if (scenario == 15) shapes[9].SetDim(3, 65);
        if (scenario == 20) {
            qLength = 32;
            for (size_t i : {0U, 2U, 4U, 5U, 8U}) shapes[i].SetDim(1, qLength);
            shapes[6] = shapes[7] = gert::Shape{1, 1, qLength, 32};
        }
        std::vector<gert::Tensor> tensors;
        tensors.reserve(16);
        for (size_t i = 0; i < 12; ++i) {
            void *data = i == 10 ? &qLength : (i == 11 ? &kLength : nullptr);
            gert::StorageShape storage;
            storage.MutableOriginShape() = shapes[i];
            storage.MutableStorageShape() = shapes[i];
            tensors.emplace_back(storage,
                gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                gert::kOnHost,
                types[i], data);
        }
        std::vector<gert::Tensor *> inputs, outputs;
        for (auto &tensor : tensors) inputs.push_back(&tensor);
        for (size_t i : {2U, 3U, 4U, 6U}) {
            outputs.push_back(&tensors[i]);
        }
        int unusedCompileInfo = 0;
        gert::OpTilingContextBuilder builder;
        auto workspace = gert::ContinuousVector::Create<size_t>(1);
        auto holder = builder.OpType("SparseLightningIndexerGradKlLoss").OpName("host_contract").IONum(12, 4)
            .AppendAttr(0.0441941738f).AppendAttr(ge::AscendString(layout))
            .AppendAttr(mode).AppendAttr(false).InputTensors(inputs).OutputTensors(outputs)
            .PlatformInfo(&platform).CompileInfo(&unusedCompileInfo)
            .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace.get()))
            .TilingDataSize(sizeof(optiling::Tiling)).Build();
        auto *context = holder.GetContext();
        if (!context) return 3;
        bool passed = optiling::TilingFunc(context) == ge::GRAPH_SUCCESS;
        if (passed != expected) {
            std::cerr << "Host contract scenario " << scenario << " failed\n";
            return 1;
        }
        if (passed) {
            gert::OpInferShapeContextBuilder shapeBuilder;
            gert::OpInferDataTypeContextBuilder dtypeBuilder;
            shapeBuilder.OpType("SparseLightningIndexerGradKlLoss").OpName("infer_shape")
                .IONum(12, 4).InputTensors(inputs);
            dtypeBuilder.OpType("SparseLightningIndexerGradKlLoss").OpName("infer_dtype").IONum(12, 4);
            for (size_t i = 0; i < 12; ++i) {
                dtypeBuilder.InputTensorDesc(i, types[i], ge::FORMAT_ND, ge::FORMAT_ND);
            }
            for (size_t i = 0; i < 4; ++i) {
                shapeBuilder.OutputTensorDesc(i, i == 3 ? ge::DT_FLOAT : types[i + 2],
                                              ge::FORMAT_ND, ge::FORMAT_ND);
                dtypeBuilder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
            }
            auto shapeHolder = shapeBuilder.Build();
            auto dtypeHolder = dtypeBuilder.Build();
            auto *shapeContext = shapeHolder.GetContext();
            auto *dtypeContext = dtypeHolder.GetContext();
            if (!shapeContext || !dtypeContext || ge::InferShape(shapeContext) != ge::GRAPH_SUCCESS ||
                ge::InferDataType(dtypeContext) != ge::GRAPH_SUCCESS) return 5;
            for (size_t i = 0; i < 4; ++i) {
                if (*shapeContext->GetOutputShape(i) != (i == 3 ? gert::Shape{1} : shapes[i + 2]) ||
                    dtypeContext->GetOutputDataType(i) != (i == 3 ? ge::DT_FLOAT : types[i + 2])) return 6;
            }
        }
    }
    std::cout << "PASS Host contract: " << scenarios.size()
              << " input constraints and output shape/dtype scenarios\n";
}
